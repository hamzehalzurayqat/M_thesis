#!/usr/bin/env python3
"""
Crack Detection + SAM2 Segmentation Pipeline

This script:
1. Loads trained RF-DETR model
2. Detects crack bounding boxes
3. Uses RF-DETR detections directly 
4. Samples point prompts from darkest pixels inside each selected crack box
5. Passes each box + point prompts to SAM2 
6. Generates final segmentation masks and visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import argparse
from types import SimpleNamespace
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# RF-DETR imports
import sys
sys.path.insert(0, str(Path(__file__).parent / "rf-detr"))
from rfdetr.models import build_model
from rfdetr.main import populate_args
from rfdetr.util.misc import nested_tensor_from_tensor_list
from rfdetr.datasets.coco import make_coco_transforms, make_coco_transforms_square_div_64

# SAM2 imports
sys.path.insert(0, str(Path(__file__).parent / "sam2"))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for logits: softmax(logits / T)."""

    def __init__(self, init_temp=1.0):
        super().__init__()
        if float(init_temp) <= 0.0:
            raise ValueError("init_temp must be > 0")
        # Optimize in log-space so T stays positive.
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(init_temp))))

    @property
    def temperature(self):
        return torch.exp(self.log_temperature)

    def forward(self, logits):
        return logits / self.temperature.clamp_min(1e-6)

    def predict_proba(self, logits):
        return F.softmax(self.forward(logits), dim=-1)


def _flatten_logits_and_labels_for_nll(logits, labels):
    """Flatten batched logits/labels for NLL. Supports [B,C] or [B,N,C] logits."""
    if logits.ndim == 2:
        flat_logits = logits
    elif logits.ndim == 3:
        flat_logits = logits.reshape(-1, logits.shape[-1])
    else:
        raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")

    if labels.ndim == 1:
        flat_labels = labels
    elif labels.ndim == 2:
        flat_labels = labels.reshape(-1)
    else:
        raise ValueError(f"Unsupported labels shape: {tuple(labels.shape)}")

    if flat_logits.shape[0] != flat_labels.shape[0]:
        raise ValueError(
            "Logits and labels do not align after flattening: "
            f"{flat_logits.shape[0]} vs {flat_labels.shape[0]}"
        )
    return flat_logits, flat_labels


def collect_logits_and_labels_from_val_loader(
    model,
    val_loader,
    device,
    logits_key='pred_logits',
    labels_key='labels',
    images_key='images',
):
    """
    Collect validation logits/labels for temperature fitting.

    Expected batch format:
      batch[images_key] -> model input
      batch[labels_key] -> class targets aligned with logits shape
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch[images_key].to(device, non_blocking=True)
            labels = batch[labels_key].to(device, non_blocking=True)
            outputs = model(images)
            logits = outputs[logits_key]
            all_logits.append(logits.detach())
            all_labels.append(labels.detach())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def fit_temperature_from_logits_labels(
    logits,
    labels,
    device,
    init_temp=1.0,
    max_iter=100,
    lr=0.05,
    ignore_index=-100,
):
    """Fit only temperature T on validation logits by minimizing NLL."""
    scaler = TemperatureScaler(init_temp=init_temp).to(device)

    logits = logits.to(device)
    labels = labels.to(device)
    flat_logits, flat_labels = _flatten_logits_and_labels_for_nll(logits, labels)

    optimizer = torch.optim.LBFGS([scaler.log_temperature], lr=float(lr), max_iter=int(max_iter))

    def closure():
        optimizer.zero_grad()
        scaled_logits = scaler(flat_logits)
        loss = F.cross_entropy(scaled_logits, flat_labels, ignore_index=int(ignore_index))
        loss.backward()
        return loss

    with torch.no_grad():
        nll_before = F.cross_entropy(flat_logits, flat_labels, ignore_index=int(ignore_index)).item()

    optimizer.step(closure)

    with torch.no_grad():
        nll_after = F.cross_entropy(scaler(flat_logits), flat_labels, ignore_index=int(ignore_index)).item()

    stats = {
        'temperature': float(scaler.temperature.item()),
        'nll_before': float(nll_before),
        'nll_after': float(nll_after),
    }
    return scaler, stats


class CrackDetectionPipeline:
    """End-to-end pipeline for crack detection and segmentation"""
    
    def __init__(
        self,
        rfdetr_checkpoint,
        sam2_checkpoint,
        sam2_config,
        device='cpu',
        num_classes=None,
        detection_temperature=3.0,
        multi_scale=False,
    ):
        """
        Initialize the pipeline with RF-DETR and SAM2 models
        
        Args:
            rfdetr_checkpoint: Path to trained RF-DETR checkpoint
            sam2_checkpoint: Path to SAM2 tiny checkpoint
            sam2_config: SAM2 config name (sam2_hiera_t.yaml)
            device: 'cuda' or 'cpu'
            num_classes: Optional RF-DETR class count override. If None, use checkpoint value.
            detection_temperature: Post-hoc temperature for RF-DETR class logits.
            multi_scale: Force RF-DETR multi-scale augmentation flag (use False for eval/inference).
        """
        self.device = torch.device(device)
        self.num_classes_override = int(num_classes) if num_classes is not None else None
        self.multi_scale = bool(multi_scale)
        print(f"Using device: {self.device}")

        # Temperature scaling (post-hoc calibration, model weights unchanged)
        self.temperature_scaler = TemperatureScaler(init_temp=float(detection_temperature)).to(self.device)
        self.temperature_scaler.eval()
        print(f"Using detection temperature T={self.get_detection_temperature():.4f}")
        
        # Load RF-DETR
        print("Loading RF-DETR model...")
        self.rfdetr_model = self._load_rfdetr(rfdetr_checkpoint)
        self.rfdetr_model.eval()

        # Build RF-DETR transform from training config stored in checkpoint
        self.transform = self._build_rfdetr_transform()
        
        # Load SAM2 tiny
        print("Loading SAM2 tiny model...")
        self.sam2_model = build_sam2(
            sam2_config, 
            sam2_checkpoint, 
            device=self.device,
            apply_postprocessing=False
        )
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        
        print("Pipeline initialized successfully!")

    def set_detection_temperature(self, temperature):
        """Set a fixed post-hoc temperature for RF-DETR logits calibration."""
        if float(temperature) <= 0.0:
            raise ValueError("temperature must be > 0")
        with torch.no_grad():
            self.temperature_scaler.log_temperature.copy_(
                torch.log(torch.tensor(float(temperature), device=self.device))
            )

    def get_detection_temperature(self):
        """Return current RF-DETR logits temperature."""
        return float(self.temperature_scaler.temperature.item())

    def fit_detection_temperature_from_logits(self, logits, labels, init_temp=None, max_iter=100, lr=0.05):
        """
        Fit temperature on held-out logits/labels and update pipeline scaler.
        This does not retrain RF-DETR weights.
        """
        if init_temp is None:
            init_temp = self.get_detection_temperature()
        scaler, stats = fit_temperature_from_logits_labels(
            logits=logits,
            labels=labels,
            device=self.device,
            init_temp=float(init_temp),
            max_iter=int(max_iter),
            lr=float(lr),
        )
        self.temperature_scaler = scaler.eval()
        return stats

    def _get_checkpoint_arg(self, name, default=None):
        """Read an argument from checkpoint args (Namespace or dict)."""
        if isinstance(self.rfdetr_args, dict):
            return self.rfdetr_args.get(name, default)
        return getattr(self.rfdetr_args, name, default)

    def _build_rfdetr_transform(self):
        """Create preprocessing transform matching RF-DETR training configuration."""
        resolution = int(self._get_checkpoint_arg('resolution', 640))
        square_resize_div_64 = bool(self._get_checkpoint_arg('square_resize_div_64', False))
        multi_scale = bool(self._get_checkpoint_arg('multi_scale', False))

        if square_resize_div_64:
            transform = make_coco_transforms_square_div_64(
                'val',
                resolution=resolution,
                multi_scale=multi_scale,
            )
            mode = 'square_resize_div_64'
        else:
            transform = make_coco_transforms(
                'val',
                resolution=resolution,
                multi_scale=multi_scale,
            )
            mode = 'standard_val_resize'

        print(
            f"RF-DETR preprocess: {mode} @ resolution={resolution} "
            f"(multi_scale={multi_scale})"
        )
        return transform
    
    def _load_rfdetr(self, checkpoint_path):
        """Load RF-DETR model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('model', None)
        if state_dict is None:
            raise ValueError("Checkpoint does not contain model weights under key 'model'")
        
        # Build model from checkpoint args
        args = checkpoint.get('args', None)
        if args is None:
            raise ValueError("Checkpoint does not contain model arguments")

        # Normalize checkpoint args into a plain dict.
        if isinstance(args, dict):
            normalized_args = dict(args)
        else:
            normalized_args = vars(args).copy()

        if self.num_classes_override is not None:
            normalized_args['num_classes'] = int(self.num_classes_override)
            print(f"Overriding RF-DETR num_classes={self.num_classes_override}")

        # Some checkpoints omit num_classes. Infer from classifier head if needed.
        if 'num_classes' not in normalized_args:
            inferred_num_classes = self._infer_num_classes_from_state_dict(state_dict)
            if inferred_num_classes is None:
                raise ValueError(
                    "Could not determine RF-DETR num_classes from checkpoint args or model weights. "
                    "Pass num_classes explicitly to CrackDetectionPipeline(..., num_classes=...)."
                )
            normalized_args['num_classes'] = int(inferred_num_classes)
            print(f"Inferred RF-DETR num_classes={int(inferred_num_classes)} from checkpoint weights")

        # Fill all missing RF-DETR fields with library defaults.
        defaults = vars(populate_args())
        defaults.update(normalized_args)

        # Force eval/inference behavior for preprocessing (no train-time multi-scale augmentation).
        defaults['multi_scale'] = bool(self.multi_scale)

        # Always bind runtime device to pipeline device.
        defaults['device'] = str(self.device)

        args = SimpleNamespace(**defaults)

        self.rfdetr_args = args
        
        model = build_model(args)
        model.to(self.device)
        
        # Load weights
        model.load_state_dict(state_dict, strict=False)
        
        return model

    @staticmethod
    def _infer_num_classes_from_state_dict(state_dict):
        """Infer class count from RF-DETR class head weights when args.num_classes is missing."""
        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue
            if 'class_embed' not in key or not key.endswith('weight'):
                continue
            if value.ndim != 2 or value.shape[0] <= 1:
                continue
            # class head output is typically [num_classes + 1, hidden_dim], where +1 is no-object.
            return int(value.shape[0] - 1)
        return None
    
    def detect_cracks(self, image_path, confidence_threshold=0.5, nms_iou_threshold=0.5, return_details=False):
        """
        Run RF-DETR inference to detect crack bounding boxes
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence score
            nms_iou_threshold: IoU threshold for NMS (<=0 disables NMS)
            return_details: If True, also return query/class indices per detection
            
        Returns:
            boxes: Detected bounding boxes in xyxy format
            scores: Confidence scores
            image: Original image (numpy array)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_np = np.array(image)
        
        # Apply transform
        transformed, _ = self.transform(image, None)
        
        # Prepare for model
        samples = nested_tensor_from_tensor_list([transformed])
        samples = samples.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.rfdetr_model(samples)
        
        # Process outputs
        pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
        pred_boxes = outputs['pred_boxes'][0]     # [num_queries, 4]
        
        # Get probabilities
        # Temperature scaling preserves argmax class predictions for T > 0, but recalibrates confidence.
        with torch.no_grad():
            scaled_logits = self.temperature_scaler(pred_logits)
            probs = scaled_logits.softmax(-1)[:, :-1]  # Exclude no-object class
            scores, labels = probs.max(-1)
        
        # Filter by confidence
        keep = scores > confidence_threshold
        query_indices = torch.nonzero(keep, as_tuple=False).squeeze(1)
        boxes = pred_boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        
        # Convert from cxcywh (normalized) to xyxy (pixel coordinates)
        h, w = image_np.shape[:2]
        boxes_xyxy = self._box_cxcywh_to_xyxy(boxes, w, h)

        # Apply class-agnostic NMS to suppress duplicate overlapping detections
        if nms_iou_threshold is not None and float(nms_iou_threshold) > 0.0:
            keep_indices = self._nms_torch(boxes_xyxy, scores, float(nms_iou_threshold))
            boxes_xyxy = boxes_xyxy[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
            query_indices = query_indices[keep_indices]

        if return_details:
            return (
                boxes_xyxy.detach().cpu().numpy(),
                scores.detach().cpu().numpy(),
                image_np,
                {
                    'query_indices': query_indices.detach().cpu().numpy(),
                    'labels': labels.detach().cpu().numpy(),
                },
            )
        
        return boxes_xyxy.detach().cpu().numpy(), scores.detach().cpu().numpy(), image_np
    
    @staticmethod
    def _box_cxcywh_to_xyxy(boxes, img_w, img_h):
        """Convert boxes from (cx, cy, w, h) normalized to (x1, y1, x2, y2) pixels"""
        cx, cy, w, h = boxes.unbind(-1)
        
        # Denormalize
        cx = cx * img_w
        cy = cy * img_h
        w = w * img_w
        h = h * img_h
        
        # Convert to xyxy
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _iou_one_to_many(box, boxes):
        """Compute IoU between one xyxy box and many xyxy boxes (torch tensors)."""
        x1 = torch.maximum(box[0], boxes[:, 0])
        y1 = torch.maximum(box[1], boxes[:, 1])
        x2 = torch.minimum(box[2], boxes[:, 2])
        y2 = torch.minimum(box[3], boxes[:, 3])

        inter_w = (x2 - x1).clamp(min=0)
        inter_h = (y2 - y1).clamp(min=0)
        inter = inter_w * inter_h

        area_box = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
        area_boxes = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        union = area_box + area_boxes - inter

        eps = torch.finfo(union.dtype).eps
        return inter / (union + eps)

    @classmethod
    def _nms_torch(cls, boxes_xyxy, scores, iou_threshold):
        """Class-agnostic NMS for xyxy boxes and confidence scores."""
        if boxes_xyxy.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes_xyxy.device)

        order = torch.argsort(scores, descending=True)
        keep = []

        while order.numel() > 0:
            current = order[0]
            keep.append(current)

            if order.numel() == 1:
                break

            remaining = order[1:]
            ious = cls._iou_one_to_many(boxes_xyxy[current], boxes_xyxy[remaining])
            order = remaining[ious <= iou_threshold]

        return torch.stack(keep)

    @staticmethod
    def _clip_box_to_image(box_xyxy, image_shape):
        """Clip box to image bounds and return int coordinates [x1, y1, x2, y2]."""
        h, w = image_shape[:2]
        x1, y1, x2, y2 = box_xyxy
        x1 = int(np.clip(np.floor(x1), 0, w - 1))
        y1 = int(np.clip(np.floor(y1), 0, h - 1))
        x2 = int(np.clip(np.ceil(x2), 0, w - 1))
        y2 = int(np.clip(np.ceil(y2), 0, h - 1))
        return [x1, y1, x2, y2]

    @staticmethod
    def _farthest_point_sample(points_xy, k):
        """Greedy farthest-point sampling for spatially distributed prompts."""
        n = len(points_xy)
        if n <= k:
            return points_xy

        selected = [0]
        min_dists = np.full(n, np.inf, dtype=np.float32)

        for _ in range(1, k):
            last = points_xy[selected[-1]]
            d2 = np.sum((points_xy - last) ** 2, axis=1)
            min_dists = np.minimum(min_dists, d2)
            next_idx = int(np.argmax(min_dists))
            selected.append(next_idx)

        return points_xy[np.array(selected, dtype=np.int32)]

    @staticmethod
    def _line_continuity_score(roi_gray, y, x, line_dark_threshold):
        """Score whether a point lies on a dark, bidirectional line-like structure."""
        h, w = roi_gray.shape
        directions = [
            (1, 0),   # 0°
            (2, 1),   # ~26°
            (1, 1),   # 45°
            (1, 2),   # ~63°
            (0, 1),   # 90°
            (-1, 2),  # ~117°
            (-1, 1),  # 135°
            (-2, 1),  # ~153°
        ]

        best = 0.0
        for dx, dy in directions:
            side_scores = []
            for sign in (-1, 1):
                dark_hits = 0
                valid = 0
                for step in (1, 2):
                    xx = x + sign * dx * step
                    yy = y + sign * dy * step
                    if 0 <= xx < w and 0 <= yy < h:
                        valid += 1
                        if roi_gray[yy, xx] <= line_dark_threshold:
                            dark_hits += 1
                side_scores.append((dark_hits / valid) if valid > 0 else 0.0)

            direction_score = min(side_scores)
            if direction_score > best:
                best = direction_score

        return float(best)

    @staticmethod
    def _boxes_overlap(box_a, box_b):
        """Return True when two xyxy boxes have a positive-area overlap."""
        inter_x1 = max(float(box_a[0]), float(box_b[0]))
        inter_y1 = max(float(box_a[1]), float(box_b[1]))
        inter_x2 = min(float(box_a[2]), float(box_b[2]))
        inter_y2 = min(float(box_a[3]), float(box_b[3]))
        return (inter_x2 - inter_x1) > 0.0 and (inter_y2 - inter_y1) > 0.0

    def select_top_non_overlapping_detections(self, boxes, scores, max_cracks=2, return_indices=False):
        """Select top-scoring detections while enforcing no overlap between boxes."""
        if len(boxes) == 0:
            if return_indices:
                return boxes, scores, np.empty((0,), dtype=np.int32)
            return boxes, scores

        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        order = np.argsort(scores)[::-1]

        selected_indices = []
        for idx in order:
            has_overlap = any(self._boxes_overlap(boxes[idx], boxes[kept_idx]) for kept_idx in selected_indices)
            if not has_overlap:
                selected_indices.append(int(idx))
            if len(selected_indices) >= int(max_cracks):
                break

        if len(selected_indices) == 0:
            if return_indices:
                return (
                    np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.int32),
                )
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        selected_indices = np.asarray(selected_indices, dtype=np.int32)
        if return_indices:
            return boxes[selected_indices], scores[selected_indices], selected_indices
        return boxes[selected_indices], scores[selected_indices]

    def sample_darkest_points(self, image, box_xyxy, num_points=1, darkest_percentile=10.0):
        """
        Sample positive point prompts from dark, neighborhood-dark, line-like pixels inside box.

        Args:
            image: RGB image as numpy array
            box_xyxy: Bounding box [x1, y1, x2, y2]
            num_points: Number of positive point prompts
            darkest_percentile: Candidate threshold percentile (lower = darker)

        Returns:
            point_coords: Nx2 float array in absolute image coordinates (x, y)
            point_labels: N int array (all ones = positive prompts)
        """
        if num_points <= 0:
            return None, None

        x1, y1, x2, y2 = self._clip_box_to_image(box_xyxy, image.shape)
        if x2 <= x1 or y2 <= y1:
            return None, None

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        roi = gray[y1:y2 + 1, x1:x2 + 1]
        if roi.size == 0:
            return None, None

        dark_threshold = np.percentile(roi, darkest_percentile)
        neighborhood_mean = cv2.blur(roi.astype(np.float32), (3, 3))
        neighborhood_threshold = np.percentile(neighborhood_mean, min(30.0, darkest_percentile + 8.0))
        line_dark_threshold = np.percentile(roi, min(35.0, darkest_percentile + 18.0))

        dark_mask = roi <= dark_threshold
        neighborhood_dark_mask = neighborhood_mean <= neighborhood_threshold
        candidate_mask = dark_mask & neighborhood_dark_mask

        candidates_local = np.argwhere(candidate_mask)
        if len(candidates_local) == 0:
            candidates_local = np.argwhere(dark_mask)
        if len(candidates_local) == 0:
            candidates_local = np.argwhere(np.ones_like(roi, dtype=bool))

        max_candidates = min(len(candidates_local), max(320, num_points * 120))
        if len(candidates_local) > max_candidates:
            intensities = roi[candidates_local[:, 0], candidates_local[:, 1]]
            idx = np.argsort(intensities)[:max_candidates]
            candidates_local = candidates_local[idx]

        candidate_scores = []
        roi_h, roi_w = roi.shape
        max_border_dist = max(1.0, min(roi_h, roi_w) / 2.0)
        for yy, xx in candidates_local:
            intensity = float(roi[yy, xx])
            neigh = float(neighborhood_mean[yy, xx])
            darkness_score = (255.0 - intensity) / 255.0
            neighborhood_score = (255.0 - neigh) / 255.0
            line_score = self._line_continuity_score(roi, int(yy), int(xx), line_dark_threshold)
            border_dist = float(min(yy, xx, roi_h - 1 - yy, roi_w - 1 - xx))
            center_score = np.clip(border_dist / max_border_dist, 0.0, 1.0)
            final_score = (
                0.40 * darkness_score
                + 0.20 * neighborhood_score
                + 0.30 * line_score
                + 0.10 * center_score
            )
            candidate_scores.append(final_score)

        candidate_scores = np.asarray(candidate_scores, dtype=np.float32)
        order = np.argsort(-candidate_scores)
        candidates_local = candidates_local[order]

        top_n = min(len(candidates_local), max(200, num_points * 60))
        candidates_local = candidates_local[:top_n]

        # Convert to absolute xy
        candidates_xy = np.stack(
            [candidates_local[:, 1] + x1, candidates_local[:, 0] + y1],
            axis=1,
        ).astype(np.float32)

        sampled_xy = self._farthest_point_sample(candidates_xy, num_points)
        point_labels = np.ones((len(sampled_xy),), dtype=np.int32)
        return sampled_xy, point_labels

    def sample_negative_points_in_and_around_box(
        self,
        image,
        box_xyxy,
        num_points=4,
        positive_point_coords=None,
        ring_ratio=0.15,
        white_percentile=85.0,
        min_distance_from_positive=6,
    ):
        """
        Sample negative point prompts only outside the bounding box using a ring.
        Sampling regions are split by distance-to-box-edge:
        - 40% near boundary (<=5 px)
        - 40% mid region (5-15 px)
        - 20% far region (>15 px)

        Args:
            image: RGB image as numpy array
            box_xyxy: Bounding box [x1, y1, x2, y2]
            num_points: Number of negative prompts
            positive_point_coords: Positive point coordinates (Nx2 xy)
            ring_ratio: Relative ring thickness around the box
            white_percentile: Bright-pixel percentile threshold in ring candidates
            min_distance_from_positive: Minimum distance (pixels) from positive points

        Returns:
            point_coords: Nx2 float array in absolute image coordinates (x, y)
            point_labels: N int array (all zeros = negative prompts)
        """
        if num_points is None or int(num_points) <= 0:
            return None, None
        num_points = int(num_points)

        x1, y1, x2, y2 = self._clip_box_to_image(box_xyxy, image.shape)
        if x2 <= x1 or y2 <= y1:
            return None, None

        box_w = x2 - x1 + 1
        box_h = y2 - y1 + 1
        pad_x = max(3, int(round(box_w * ring_ratio)))
        pad_y = max(3, int(round(box_h * ring_ratio)))

        h, w = image.shape[:2]
        outer_x1 = max(0, x1 - pad_x)
        outer_y1 = max(0, y1 - pad_y)
        outer_x2 = min(w - 1, x2 + pad_x)
        outer_y2 = min(h - 1, y2 + pad_y)

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if outer_x2 <= outer_x1 or outer_y2 <= outer_y1:
            return None, None

        outer_roi = gray[outer_y1:outer_y2 + 1, outer_x1:outer_x2 + 1]
        yy_xx = np.argwhere(np.ones_like(outer_roi, dtype=bool))
        if len(yy_xx) == 0:
            return None, None

        yy = yy_xx[:, 0] + outer_y1
        xx = yy_xx[:, 1] + outer_x1

        outside_mask = (xx < x1) | (xx > x2) | (yy < y1) | (yy > y2)
        if not np.any(outside_mask):
            return None, None

        yy = yy[outside_mask]
        xx = xx[outside_mask]
        intensities = gray[yy, xx].astype(np.float32)

        # Keep candidates away from positive points
        if positive_point_coords is not None and len(positive_point_coords) > 0:
            pos = np.asarray(positive_point_coords, dtype=np.float32)
            candidates_xy_all = np.stack([xx, yy], axis=1).astype(np.float32)
            d2 = np.sum((candidates_xy_all[:, None, :] - pos[None, :, :]) ** 2, axis=2)
            min_d2 = np.min(d2, axis=1)
            keep_dist = min_d2 >= float(min_distance_from_positive) ** 2
            if not np.any(keep_dist):
                return None, None
            xx = xx[keep_dist]
            yy = yy[keep_dist]
            intensities = intensities[keep_dist]

        # Bright-pixel filtering with safe fallback
        white_threshold = np.percentile(intensities, float(white_percentile))
        bright_mask = intensities >= white_threshold
        if np.any(bright_mask):
            xx = xx[bright_mask]
            yy = yy[bright_mask]
            intensities = intensities[bright_mask]

        if len(xx) == 0:
            return None, None

        # Distance from each outside candidate to nearest box boundary (outside distance)
        dx = np.where(xx < x1, x1 - xx, np.where(xx > x2, xx - x2, 0))
        dy = np.where(yy < y1, y1 - yy, np.where(yy > y2, yy - y2, 0))
        edge_dist = np.sqrt(dx.astype(np.float32) ** 2 + dy.astype(np.float32) ** 2)

        candidates_xy = np.stack([xx, yy], axis=1).astype(np.float32)

        near_mask = edge_dist <= 5.0
        mid_mask = (edge_dist > 5.0) & (edge_dist <= 15.0)
        far_mask = edge_dist > 15.0

        weights = np.array([0.40, 0.40, 0.20], dtype=np.float32)
        raw = weights * float(num_points)
        counts = np.floor(raw).astype(np.int32)
        remainder = int(num_points - np.sum(counts))
        if remainder > 0:
            frac = raw - counts
            for idx in np.argsort(-frac)[:remainder]:
                counts[idx] += 1
        n_near, n_mid, n_far = int(counts[0]), int(counts[1]), int(counts[2])

        def _sample_region(region_xy, region_dist, k):
            if k <= 0 or len(region_xy) == 0:
                return np.empty((0, 2), dtype=np.float32)
            order = np.argsort(region_dist)  # prioritize closer-to-boundary points
            region_xy = region_xy[order]
            max_candidates = min(len(region_xy), max(300, k * 120))
            region_xy = region_xy[:max_candidates]
            return self._farthest_point_sample(region_xy.astype(np.float32), int(k)).astype(np.float32)

        near_points = _sample_region(candidates_xy[near_mask], edge_dist[near_mask], n_near)
        mid_points = _sample_region(candidates_xy[mid_mask], edge_dist[mid_mask], n_mid)
        far_points = _sample_region(candidates_xy[far_mask], edge_dist[far_mask], n_far)

        sampled_parts = [arr for arr in (near_points, mid_points, far_points) if len(arr) > 0]
        sampled_xy = (
            np.concatenate(sampled_parts, axis=0).astype(np.float32)
            if len(sampled_parts) > 0
            else np.empty((0, 2), dtype=np.float32)
        )

        if len(sampled_xy) < num_points:
            needed = int(num_points - len(sampled_xy))
            selected_set = set(map(tuple, sampled_xy.astype(np.int32))) if len(sampled_xy) > 0 else set()
            all_order = np.argsort(edge_dist)  # fill with boundary-priority outside points
            pool = candidates_xy[all_order].astype(np.float32)
            if len(selected_set) > 0:
                pool_mask = np.array(
                    [tuple(pt.astype(np.int32)) not in selected_set for pt in pool],
                    dtype=bool,
                )
                pool = pool[pool_mask]
            if len(pool) > 0 and needed > 0:
                fill = self._farthest_point_sample(pool, needed).astype(np.float32)
                sampled_xy = np.concatenate([sampled_xy, fill], axis=0).astype(np.float32)

        if len(sampled_xy) == 0:
            return None, None

        if len(sampled_xy) > num_points:
            sampled_xy = sampled_xy[:num_points].astype(np.float32)

        point_labels = np.zeros((len(sampled_xy),), dtype=np.int32)
        return sampled_xy, point_labels

    def _set_sam2_image(self, image):
        """Set SAM2 predictor image robustly across NumPy/PyTorch environments."""
        if isinstance(image, np.ndarray):
            image_np = np.ascontiguousarray(image)
            if image_np.dtype != np.uint8:
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)

            try:
                self.sam2_predictor.set_image(image_np)
                return
            except TypeError as exc:
                msg = str(exc)
                if ('np.ndarray' not in msg) and ('numpy.ndarray' not in msg):
                    raise
                image_pil = Image.fromarray(image_np, mode='RGB')
                self.sam2_predictor.set_image(image_pil)
                return

        if isinstance(image, Image.Image):
            self.sam2_predictor.set_image(image)
            return

        raise TypeError(f"Unsupported image type for SAM2: {type(image)}")

    def segment_crack(
        self,
        image,
        box_xyxy,
        num_dark_points=6,
        num_negative_points=12,
        prompt_mode='box+points',
        return_prompts=False,
    ):
        """
        Use SAM2 to generate segmentation mask from box + darkest-pixel point prompts
        
        Args:
            image: Input image (numpy array)
            box_xyxy: Bounding box in [x1, y1, x2, y2] format
            num_dark_points: Number of positive prompts sampled on the crack
            num_negative_points: Number of negative prompts sampled around the crack
            prompt_mode: Prompt strategy: 'box', 'points', or 'box+points'
            return_prompts: If True, also return sampled point coordinates
            
        Returns:
            mask: Binary segmentation mask
            score: Prediction score
        """
        self._set_sam2_image(image)

        prompt_mode = str(prompt_mode).strip().lower()
        if prompt_mode not in {'box', 'points', 'box+points'}:
            raise ValueError(
                f"Invalid prompt_mode='{prompt_mode}'. Use one of: 'box', 'points', 'box+points'."
            )

        pos_point_coords, pos_point_labels = self.sample_darkest_points(
            image=image,
            box_xyxy=box_xyxy,
            num_points=num_dark_points,
            darkest_percentile=10.0,
        )

        if num_negative_points is None or int(num_negative_points) <= 0:
            num_negative_points = int(num_dark_points * 2)
        else:
            num_negative_points = int(num_negative_points)

        neg_point_coords, neg_point_labels = self.sample_negative_points_in_and_around_box(
            image=image,
            box_xyxy=box_xyxy,
            num_points=num_negative_points,
            positive_point_coords=pos_point_coords,
            ring_ratio=0.15,
            white_percentile=85.0,
        )

        point_coords = None
        point_labels = None
        if pos_point_coords is not None and len(pos_point_coords) > 0:
            point_coords = pos_point_coords
            point_labels = pos_point_labels

        if neg_point_coords is not None and len(neg_point_coords) > 0:
            if point_coords is None:
                point_coords = neg_point_coords
                point_labels = neg_point_labels
            else:
                point_coords = np.concatenate([point_coords, neg_point_coords], axis=0)
                point_labels = np.concatenate([point_labels, neg_point_labels], axis=0)

        if prompt_mode == 'points' and (point_coords is None or len(point_coords) == 0):
            x1, y1, x2, y2 = self._clip_box_to_image(box_xyxy, image.shape)
            center_x = float((x1 + x2) / 2.0)
            center_y = float((y1 + y2) / 2.0)
            point_coords = np.array([[center_x, center_y]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)

        predict_kwargs = {
            'multimask_output': False,
        }

        if prompt_mode in {'box', 'box+points'}:
            predict_kwargs['box'] = box_xyxy

        if prompt_mode in {'points', 'box+points'} and point_coords is not None and len(point_coords) > 0:
            predict_kwargs['point_coords'] = point_coords
            predict_kwargs['point_labels'] = point_labels

        masks, scores, _ = self.sam2_predictor.predict(**predict_kwargs)

        if return_prompts:
            return {
                'mask': masks[0],
                'score': scores[0],
                'positive_coords': pos_point_coords,
                'negative_coords': neg_point_coords,
                'point_coords': point_coords,
                'point_labels': point_labels,
            }
        return masks[0], scores[0]
    
    def run_pipeline(
        self,
        image_path,
        confidence_threshold=0.25,
        nms_iou_threshold=0.5,
        output_path=None,
        num_dark_points=6,
        num_negative_points=0,
        max_cracks=0,
        prompt_mode='box+points',
    ):
        """
        Run complete pipeline: detection + segmentation
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum detection confidence
            nms_iou_threshold: IoU threshold for RF-DETR NMS (<=0 disables NMS)
            output_path: Path to save output visualization
            num_dark_points: Number of positive crack prompts for SAM2
            num_negative_points: Number of negative prompts around crack for SAM2
            max_cracks: Max non-overlapping detections to segment (0 = use all detections).
            prompt_mode: SAM2 prompt strategy: 'box', 'points', or 'box+points'
            
        Returns:
            result: Dictionary containing detections, per-box masks, and image
        """
        print(f"\nProcessing: {image_path}")
        
        # Step 1: Detect cracks with RF-DETR
        print("  [1/2] Running RF-DETR detection...")
        boxes, scores, image = self.detect_cracks(
            image_path,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )
        
        if len(boxes) == 0:
            print("  ⚠️  No cracks detected!")
            return None
        
        print(f"  ✓ Detected {len(boxes)} crack(s)")

        if int(max_cracks) > 0:
            selected_boxes, selected_scores, selected_indices = self.select_top_non_overlapping_detections(
                boxes,
                scores,
                max_cracks=max_cracks,
                return_indices=True,
            )
            print(
                f"  ✓ Selected {len(selected_boxes)} crack(s) after non-overlap "
            )
        else:
            selected_boxes = np.asarray(boxes, dtype=np.float32)
            selected_scores = np.asarray(scores, dtype=np.float32)
            selected_indices = np.arange(len(selected_boxes), dtype=np.int32)
            print(f"  ✓ Using all {len(selected_boxes)} RF-DETR detection(s)")

        if len(selected_boxes) == 0:
            print("  ⚠️  No boxes left after selection")
            return None

        # Step 2: Segment selected box with SAM2
        print(
            "  [2/2] Running SAM2 segmentation "
            f"(mode={prompt_mode}, {num_dark_points} positive / {num_negative_points} negative points)..."
        )
        masks = []
        mask_scores = []
        positive_point_coords_list = []
        negative_point_coords_list = []

        for i, (box, score) in enumerate(
            zip(selected_boxes, selected_scores),
            start=1,
        ):
            print(f"    - Crack {i}: det_score={float(score):.3f}, box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

            seg_result = self.segment_crack(
                image,
                box,
                num_dark_points=num_dark_points,
                num_negative_points=num_negative_points,
                prompt_mode=prompt_mode,
                return_prompts=True,
            )
            masks.append(seg_result['mask'])
            mask_scores.append(float(seg_result['score']))
            positive_point_coords_list.append(seg_result['positive_coords'])
            negative_point_coords_list.append(seg_result['negative_coords'])

        print(f"  ✓ Segmentation complete for {len(masks)} crack(s)")

        # Preserve best_* fields for backward compatibility
        best_rel_idx = int(np.argmax(selected_scores))
        best_box = selected_boxes[best_rel_idx]
        best_score = float(selected_scores[best_rel_idx])
        best_mask = masks[best_rel_idx]
        best_mask_score = mask_scores[best_rel_idx]
        best_point_coords = positive_point_coords_list[best_rel_idx]
        best_negative_point_coords = negative_point_coords_list[best_rel_idx]
        
        # Visualize results
        if output_path:
            self.visualize_results(
                image,
                selected_boxes,
                masks,
                selected_scores,
                output_path,
                positive_point_coords_list=positive_point_coords_list,
                negative_point_coords_list=negative_point_coords_list,
            )
            print(f"  💾 Saved visualization: {output_path}")
        
        return {
            'boxes': boxes,
            'scores': scores,
            'selected_boxes': selected_boxes,
            'selected_scores': selected_scores,
            'masks': masks,
            'mask_scores': mask_scores,
            'point_coords_list': positive_point_coords_list,
            'positive_point_coords_list': positive_point_coords_list,
            'negative_point_coords_list': negative_point_coords_list,
            'best_box': best_box,
            'best_score': best_score,
            'mask': best_mask,
            'mask_score': best_mask_score,
            'point_coords': best_point_coords,
            'negative_point_coords': best_negative_point_coords,
            'image': image
        }
    
    @staticmethod
    def visualize_results(
        image,
        boxes,
        masks,
        scores,
        output_path,
        positive_point_coords_list=None,
        negative_point_coords_list=None,
    ):
        """Create visualization with selected detection boxes and segmentation masks"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Detection result
        axes[1].imshow(image)
        for i, (box, score) in enumerate(zip(boxes, scores), start=1):
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            axes[1].add_patch(rect)
            axes[1].text(
                box[0], max(0, box[1] - 10),
                f'Crack {i}: {float(score):.2f}',
                color='red',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7)
            )
        axes[1].set_title('RF-DETR Detections')
        axes[1].axis('off')

        if positive_point_coords_list is not None or negative_point_coords_list is not None:
            plotted_pos = False
            plotted_neg = False

            if positive_point_coords_list is not None:
                for point_coords in positive_point_coords_list:
                    if point_coords is not None and len(point_coords) > 0:
                        axes[1].scatter(
                            point_coords[:, 0],
                            point_coords[:, 1],
                            c='yellow',
                            s=22,
                            edgecolors='black',
                            linewidths=0.5,
                        )
                        plotted_pos = True

            if negative_point_coords_list is not None:
                for point_coords in negative_point_coords_list:
                    if point_coords is not None and len(point_coords) > 0:
                        axes[1].scatter(
                            point_coords[:, 0],
                            point_coords[:, 1],
                            c='cyan',
                            s=20,
                            edgecolors='black',
                            linewidths=0.5,
                        )
                        plotted_neg = True

            if plotted_pos or plotted_neg:
                legend_parts = []
                if plotted_pos:
                    legend_parts.append('Yellow: positive prompts')
                if plotted_neg:
                    legend_parts.append('Cyan: negative prompts')
                axes[1].text(
                    10,
                    image.shape[0] - 10,
                    ' | '.join(legend_parts),
                    color='white',
                    fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5),
                )
        
        # Segmentation result
        axes[2].imshow(image)
        # Overlay combined mask with transparency
        combined_mask = np.zeros(image.shape[:2], dtype=bool)
        for mask in masks:
            combined_mask |= mask.astype(bool)
        mask_overlay = np.zeros((*combined_mask.shape, 4))
        mask_overlay[combined_mask] = [1, 0, 0, 0.5]  # Red with 50% transparency
        axes[2].imshow(mask_overlay)
        axes[2].set_title('SAM2 Segmentation (all selected cracks)')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Crack Detection + SAM2 Segmentation Pipeline')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--rfdetr-checkpoint', type=str, required=True, help='Path to RF-DETR checkpoint')
    parser.add_argument('--sam2-checkpoint', type=str, required=True, help='Path to SAM2 checkpoint')
    parser.add_argument('--sam2-config', type=str, default='sam2_hiera_t.yaml', help='SAM2 config')
    parser.add_argument('--output', type=str, default='output_crack_segmentation.png', help='Output path')
    parser.add_argument('--confidence', type=float, default=0.25, help='Detection confidence threshold')
    parser.add_argument('--nms-iou', type=float, default=0.1, help='RF-DETR NMS IoU threshold (<=0 disables NMS)')
    parser.add_argument('--max-cracks', type=int, default=3, help='Max non-overlapping detections to segment (0 = use all RF-DETR detections)')
    parser.add_argument(
        '--num-dark-points',
        '--num-positive-points',
        dest='num_dark_points',
        type=int,
        default=0,
        help='Number of positive crack point prompts for SAM2',
    )
    parser.add_argument(
        '--num-negative-points',
        type=int,
        default=0,
        help='Number of negative point prompts sampled around each crack',
    )
    parser.add_argument(
        '--prompt-mode',
        type=str,
        default='box+points',
        choices=['box', 'points', 'box+points'],
        help="SAM2 prompt strategy: 'box', 'points', or 'box+points'",
    )
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    parser.add_argument(
        '--temperature',
        type=float,
        default=3.0,
        help='Post-hoc RF-DETR logits temperature (T>0). T>1 softens confidence.',
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=None,
        help='Optional RF-DETR num_classes override (only used when provided)',
    )
    
    args = parser.parse_args()
    
    # Auto-detect CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Initialize pipeline
    pipeline = CrackDetectionPipeline(
        rfdetr_checkpoint=args.rfdetr_checkpoint,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        device=args.device,
        num_classes=args.num_classes,
        detection_temperature=args.temperature,
    )
    
    # Run pipeline
    result = pipeline.run_pipeline(
        image_path=args.image,
        confidence_threshold=args.confidence,
        nms_iou_threshold=args.nms_iou,
        output_path=args.output,
        num_dark_points=args.num_dark_points,
        num_negative_points=args.num_negative_points,
        max_cracks=args.max_cracks,
        prompt_mode=args.prompt_mode,
    )
    
    if result:
        print("\n✅ Pipeline completed successfully!")
    else:
        print("\n❌ No cracks detected in image")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Crack Detection + SAM2 Segmentation Pipeline

This script:
1. Loads trained RF-DETR model
2. Detects crack bounding boxes
3. Uses RF-DETR detections directly (no forced top-1 selection)
4. Samples point prompts from darkest pixels inside each selected crack box
5. Passes each box + point prompts to SAM2
6. Generates final segmentation masks and visualization
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Prefer local source tree when available.
PROJECT_DIR = Path(__file__).parent
LOCAL_RFDETR = PROJECT_DIR / "rf-detr"
LOCAL_SAM2 = PROJECT_DIR / "sam2"
if LOCAL_RFDETR.exists():
    sys.path.insert(0, str(LOCAL_RFDETR))
if LOCAL_SAM2.exists():
    sys.path.insert(0, str(LOCAL_SAM2))

# RF-DETR imports
from rfdetr.util.misc import nested_tensor_from_tensor_list
from rfdetr.datasets.coco import make_coco_transforms, make_coco_transforms_square_div_64

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class CrackDetectionPipeline:
    """End-to-end pipeline for crack detection and segmentation."""

    def __init__(self, rfdetr_checkpoint, sam2_checkpoint, sam2_config, device="cpu", num_classes=None):
        self.device = torch.device(device)
        self.num_classes_override = int(num_classes) if num_classes is not None else None
        print(f"Using device: {self.device}")

        print("Loading RF-DETR model...")
        self.rfdetr_model = self._load_rfdetr(rfdetr_checkpoint)
        self.rfdetr_model.eval()

        self.transform = self._build_rfdetr_transform()

        print("Loading SAM2 model...")
        self.sam2_model = build_sam2(
            sam2_config,
            sam2_checkpoint,
            device=self.device,
            apply_postprocessing=False,
        )
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        print("Pipeline initialized successfully!")

    def _get_checkpoint_arg(self, name, default=None):
        if isinstance(self.rfdetr_args, dict):
            return self.rfdetr_args.get(name, default)
        return getattr(self.rfdetr_args, name, default)

    def _build_rfdetr_transform(self):
        resolution = int(self._get_checkpoint_arg("resolution", 512))
        square_resize_div_64 = bool(self._get_checkpoint_arg("square_resize_div_64", True))
        self.rfdetr_resolution = resolution
        self.rfdetr_square_resize_div_64 = square_resize_div_64

        if square_resize_div_64:
            transform = make_coco_transforms_square_div_64("val", resolution=resolution)
            mode = "square_resize_div_64"
        else:
            transform = make_coco_transforms("val", resolution=resolution)
            mode = "standard_val_resize"

        print(f"RF-DETR preprocess: {mode} @ resolution={resolution}")
        return transform


    def _load_rfdetr(self, checkpoint_path):
        """Load checkpoint with shape-aware filtering to avoid dimension mismatches."""
        from rfdetr import RFDETRBase
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract state_dict and args from checkpoint
        state_dict = checkpoint.get("model", None) or checkpoint
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint does not contain a state_dict")
        
        ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
        if not isinstance(ckpt_args, dict):
            try:
                ckpt_args = vars(ckpt_args)
            except Exception:
                ckpt_args = {}
        
        # Infer num_queries from checkpoint embeddings
        # RF-DETR commonly stores total rows = num_queries * group_detr
        rows = None
        if "refpoint_embed.weight" in state_dict:
            rows = int(state_dict["refpoint_embed.weight"].shape[0])
        elif "query_feat.weight" in state_dict:
            rows = int(state_dict["query_feat.weight"].shape[0])
        
        group_detr = int(ckpt_args.get("group_detr", 13))
        num_queries = int(ckpt_args.get("num_queries", 300))
        if rows is not None and group_detr > 0 and rows % group_detr == 0:
            num_queries = rows // group_detr
        
        # Infer class count directly from classifier head rows.
        # RF-DETR stores exactly num_classes rows (no +1 background row).
        num_classes = int(ckpt_args.get("num_classes", 1))
        if "class_embed.weight" in state_dict:
            num_classes = int(state_dict["class_embed.weight"].shape[0])
        
        # Only honour an explicit override if caller specifically set one.
        if self.num_classes_override is not None:
            num_classes = int(self.num_classes_override)
        
        two_stage = bool(ckpt_args.get("two_stage", True))
        
        print(f"Init config -> num_queries={num_queries}, num_classes={num_classes}, two_stage={two_stage}, group_detr={group_detr}")
        
        # Initialize model with inferred checkpoint-compatible config
        model = RFDETRBase(
            num_queries=num_queries,
            num_classes=num_classes,
            two_stage=two_stage,
            group_detr=group_detr,
            class_names=["crack"],
        )
        
        # Resolve wrapped torch module across RF-DETR versions
        target = None
        container = None
        if hasattr(model, "load_state_dict"):
            target = model
        elif hasattr(model, "model"):
            container = model.model
            for candidate_name in ("model", "inference_model"):
                candidate = getattr(container, candidate_name, None)
                if hasattr(candidate, "load_state_dict"):
                    target = candidate
                    break
        
        if target is None:
            raise TypeError(
                f"Could not find a torch module to load weights into. Wrapper type={type(model)}"
            )
        
        # Ensure detection head matches checkpoint class count when API supports it
        if container is not None and hasattr(container, "reinitialize_detection_head"):
            try:
                container.reinitialize_detection_head(num_classes)
                print(f"Reinitialized detection head to {num_classes} class(es)")
            except Exception as e:
                print(f"Skipping detection head reinit: {e}")
        
        # Shape-safe load: keep only checkpoint tensors that match target parameter shapes
        target_state = target.state_dict()
        filtered_state = {}
        mismatched = []
        for k, v in state_dict.items():
            if k in target_state and target_state[k].shape == v.shape:
                filtered_state[k] = v
            elif k in target_state:
                mismatched.append((k, tuple(v.shape), tuple(target_state[k].shape)))
        
        missing, unexpected = target.load_state_dict(filtered_state, strict=False)
        print(f"Loaded checkpoint tensors: {len(filtered_state)}")
        print(f"Missing keys after load: {len(missing)} | Unexpected keys: {len(unexpected)}")
        print(f"Skipped mismatched keys: {len(mismatched)}")
        if len(mismatched) > 0:
            print("First mismatches:")
            for item in mismatched[:5]:
                print("  ", item)
        
        # Put both wrapper and inner module in eval mode when available
        if hasattr(target, "eval"):
            target.eval()
        if hasattr(model, "eval"):
            model.eval()
        
        # Store args for transform configuration
        self.rfdetr_args = ckpt_args
        self.backbone_block_size = int(ckpt_args.get("patch_size", 14)) * int(
            ckpt_args.get("num_windows", 4)
        )
        
        # Always return the loadable target (either model itself or internal module)
        return target

    @staticmethod
    def _pil_to_tensor_no_numpy(pil_img):
        w, h = pil_img.size
        pixels = pil_img.getdata()
        tensor = torch.tensor(list(pixels), dtype=torch.float32).view(h, w, 3)
        tensor = tensor.permute(2, 0, 1) / 255.0
        return tensor

    def _apply_transform_safe(self, image_pil):
        try:
            transformed, _ = self.transform(image_pil, None)
            return transformed
        except TypeError as exc:
            msg = str(exc)
            if "expected np.ndarray" not in msg and "numpy.ndarray" not in msg:
                raise

            # Fallback path for torchvision/numpy bridge incompatibility.
            res = int(getattr(self, "rfdetr_resolution", 512))
            if bool(getattr(self, "rfdetr_square_resize_div_64", False)):
                side = int(np.ceil(res / 64.0) * 64)
                resized = image_pil.resize((side, side), resample=Image.BILINEAR)
            else:
                resized = image_pil.resize((res, res), resample=Image.BILINEAR)
            return self._pil_to_tensor_no_numpy(resized)

    def _ensure_backbone_divisible_size(self, image_tensor):
        """Resize CHW tensor so H/W are divisible by backbone block size."""
        if not torch.is_tensor(image_tensor) or image_tensor.ndim != 3:
            return image_tensor

        block = int(getattr(self, "backbone_block_size", 56))
        h, w = int(image_tensor.shape[-2]), int(image_tensor.shape[-1])
        if h % block == 0 and w % block == 0:
            return image_tensor

        # Prefer shrinking to nearest valid size to avoid unnecessary upscaling.
        new_h = (h // block) * block
        new_w = (w // block) * block
        if new_h <= 0:
            new_h = block
        if new_w <= 0:
            new_w = block

        resized = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        print(f"Adjusted RF-DETR input from {h}x{w} to {new_h}x{new_w} (block_size={block})")
        return resized

    def detect_cracks(self, image_path, confidence_threshold=0.5, nms_iou_threshold=0.5, return_details=False):
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        transformed = self._apply_transform_safe(image)
        transformed = self._ensure_backbone_divisible_size(transformed)
        samples = nested_tensor_from_tensor_list([transformed]).to(self.device)

        with torch.no_grad():
            outputs = self.rfdetr_model(samples)

        pred_logits = outputs["pred_logits"][0]
        pred_boxes = outputs["pred_boxes"][0]

        probs = pred_logits.softmax(-1)[:, :-1]
        if probs.shape[-1] == 0:
            probs = torch.sigmoid(pred_logits)
        scores, labels = probs.max(-1)

        keep = scores > confidence_threshold
        query_indices = torch.nonzero(keep, as_tuple=False).squeeze(1)
        boxes = pred_boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        h, w = image_np.shape[:2]
        boxes_xyxy = self._box_cxcywh_to_xyxy(boxes, w, h)

        if nms_iou_threshold is not None and float(nms_iou_threshold) > 0.0:
            keep_indices = self._nms_torch(boxes_xyxy, scores, float(nms_iou_threshold))
            boxes_xyxy = boxes_xyxy[keep_indices]
            scores = scores[keep_indices]
            labels = labels[keep_indices]
            query_indices = query_indices[keep_indices]

        if return_details:
            return (
                boxes_xyxy.cpu().numpy(),
                scores.cpu().numpy(),
                image_np,
                {"query_indices": query_indices.cpu().numpy(), "labels": labels.cpu().numpy()},
            )

        return boxes_xyxy.cpu().numpy(), scores.cpu().numpy(), image_np

    @staticmethod
    def _box_cxcywh_to_xyxy(boxes, img_w, img_h):
        cx, cy, w, h = boxes.unbind(-1)
        cx = cx * img_w
        cy = cy * img_h
        w = w * img_w
        h = h * img_h
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _iou_one_to_many(box, boxes):
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
        h, w = image_shape[:2]
        x1, y1, x2, y2 = box_xyxy
        x1 = int(np.clip(np.floor(x1), 0, w - 1))
        y1 = int(np.clip(np.floor(y1), 0, h - 1))
        x2 = int(np.clip(np.ceil(x2), 0, w - 1))
        y2 = int(np.clip(np.ceil(y2), 0, h - 1))
        return [x1, y1, x2, y2]

    @staticmethod
    def _farthest_point_sample(points_xy, k):
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
        h, w = roi_gray.shape
        directions = [
            (1, 0),
            (2, 1),
            (1, 1),
            (1, 2),
            (0, 1),
            (-1, 2),
            (-1, 1),
            (-2, 1),
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
        inter_x1 = max(float(box_a[0]), float(box_b[0]))
        inter_y1 = max(float(box_a[1]), float(box_b[1]))
        inter_x2 = min(float(box_a[2]), float(box_b[2]))
        inter_y2 = min(float(box_a[3]), float(box_b[3]))
        return (inter_x2 - inter_x1) > 0.0 and (inter_y2 - inter_y1) > 0.0

    def select_top_non_overlapping_detections(self, boxes, scores, max_cracks=2, return_indices=False):
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
            empty_boxes = np.empty((0, 4), dtype=np.float32)
            empty_scores = np.empty((0,), dtype=np.float32)
            if return_indices:
                return empty_boxes, empty_scores, np.empty((0,), dtype=np.int32)
            return empty_boxes, empty_scores

        selected_indices = np.asarray(selected_indices, dtype=np.int32)
        if return_indices:
            return boxes[selected_indices], scores[selected_indices], selected_indices
        return boxes[selected_indices], scores[selected_indices]

    def sample_darkest_points(self, image, box_xyxy, num_points=1, darkest_percentile=10.0):
        if num_points <= 0:
            return None, None

        x1, y1, x2, y2 = self._clip_box_to_image(box_xyxy, image.shape)
        if x2 <= x1 or y2 <= y1:
            return None, None

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        roi = gray[y1 : y2 + 1, x1 : x2 + 1]
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
            final_score = 0.40 * darkness_score + 0.20 * neighborhood_score + 0.30 * line_score + 0.10 * center_score
            candidate_scores.append(final_score)

        candidate_scores = np.asarray(candidate_scores, dtype=np.float32)
        order = np.argsort(-candidate_scores)
        candidates_local = candidates_local[order]

        top_n = min(len(candidates_local), max(200, num_points * 60))
        candidates_local = candidates_local[:top_n]

        candidates_xy = np.stack([candidates_local[:, 1] + x1, candidates_local[:, 0] + y1], axis=1).astype(np.float32)
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

        outer_roi = gray[outer_y1 : outer_y2 + 1, outer_x1 : outer_x2 + 1]
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

        white_threshold = np.percentile(intensities, float(white_percentile))
        bright_mask = intensities >= white_threshold
        if np.any(bright_mask):
            xx = xx[bright_mask]
            yy = yy[bright_mask]
            intensities = intensities[bright_mask]

        if len(xx) == 0:
            return None, None

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
            order = np.argsort(region_dist)
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
            all_order = np.argsort(edge_dist)
            pool = candidates_xy[all_order].astype(np.float32)
            if len(selected_set) > 0:
                pool_mask = np.array([tuple(pt.astype(np.int32)) not in selected_set for pt in pool], dtype=bool)
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
        if isinstance(image, np.ndarray):
            image_np = np.ascontiguousarray(image)
            if image_np.dtype != np.uint8:
                image_np = np.clip(image_np, 0, 255).astype(np.uint8)
            try:
                self.sam2_predictor.set_image(image_np)
                return
            except TypeError as exc:
                msg = str(exc)
                if ("np.ndarray" not in msg) and ("numpy.ndarray" not in msg):
                    raise
                self.sam2_predictor.set_image(Image.fromarray(image_np, mode="RGB"))
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
        prompt_mode="box+points",
        return_prompts=False,
    ):
        self._set_sam2_image(image)

        prompt_mode = str(prompt_mode).strip().lower()
        if prompt_mode not in {"box", "points", "box+points"}:
            raise ValueError(f"Invalid prompt_mode={prompt_mode}")

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

        if prompt_mode == "points" and (point_coords is None or len(point_coords) == 0):
            x1, y1, x2, y2 = self._clip_box_to_image(box_xyxy, image.shape)
            point_coords = np.array([[(x1 + x2) / 2.0, (y1 + y2) / 2.0]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)

        predict_kwargs = {"multimask_output": False}
        if prompt_mode in {"box", "box+points"}:
            predict_kwargs["box"] = box_xyxy
        if prompt_mode in {"points", "box+points"} and point_coords is not None and len(point_coords) > 0:
            predict_kwargs["point_coords"] = point_coords
            predict_kwargs["point_labels"] = point_labels

        masks, scores, _ = self.sam2_predictor.predict(**predict_kwargs)

        if return_prompts:
            return {
                "mask": masks[0],
                "score": scores[0],
                "positive_coords": pos_point_coords,
                "negative_coords": neg_point_coords,
                "point_coords": point_coords,
                "point_labels": point_labels,
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
        prompt_mode="box+points",
    ):
        print(f"\nProcessing: {image_path}")

        print("  [1/2] Running RF-DETR detection...")
        boxes, scores, image = self.detect_cracks(
            image_path,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )

        if len(boxes) == 0:
            print("  No cracks detected")
            return None

        print(f"  Detected {len(boxes)} crack(s)")

        if int(max_cracks) > 0:
            selected_boxes, selected_scores, _ = self.select_top_non_overlapping_detections(
                boxes,
                scores,
                max_cracks=max_cracks,
                return_indices=True,
            )
            print(f"  Selected {len(selected_boxes)} crack(s) after non-overlap filtering")
        else:
            selected_boxes = np.asarray(boxes, dtype=np.float32)
            selected_scores = np.asarray(scores, dtype=np.float32)
            print(f"  Using all {len(selected_boxes)} detections")

        if len(selected_boxes) == 0:
            print("  No boxes left after selection")
            return None

        print(
            "  [2/2] Running SAM2 segmentation "
            f"(mode={prompt_mode}, +{num_dark_points}/-{num_negative_points})..."
        )

        masks = []
        mask_scores = []
        positive_point_coords_list = []
        negative_point_coords_list = []

        for i, (box, score) in enumerate(zip(selected_boxes, selected_scores), start=1):
            print(
                f"    - Crack {i}: det_score={float(score):.3f}, "
                f"box=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
            )
            seg_result = self.segment_crack(
                image,
                box,
                num_dark_points=num_dark_points,
                num_negative_points=num_negative_points,
                prompt_mode=prompt_mode,
                return_prompts=True,
            )
            masks.append(seg_result["mask"])
            mask_scores.append(float(seg_result["score"]))
            positive_point_coords_list.append(seg_result["positive_coords"])
            negative_point_coords_list.append(seg_result["negative_coords"])

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
            print(f"  Saved visualization: {output_path}")

        best_rel_idx = int(np.argmax(selected_scores))
        return {
            "boxes": boxes,
            "scores": scores,
            "selected_boxes": selected_boxes,
            "selected_scores": selected_scores,
            "masks": masks,
            "mask_scores": mask_scores,
            "positive_point_coords_list": positive_point_coords_list,
            "negative_point_coords_list": negative_point_coords_list,
            "best_box": selected_boxes[best_rel_idx],
            "best_score": float(selected_scores[best_rel_idx]),
            "mask": masks[best_rel_idx],
            "mask_score": mask_scores[best_rel_idx],
            "point_coords": positive_point_coords_list[best_rel_idx],
            "negative_point_coords": negative_point_coords_list[best_rel_idx],
            "image": image,
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
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(image)
        for i, (box, score) in enumerate(zip(boxes, scores), start=1):
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            axes[1].add_patch(rect)
            axes[1].text(
                box[0],
                max(0, box[1] - 10),
                f"Crack {i}: {float(score):.2f}",
                color="red",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.7),
            )

        if positive_point_coords_list is not None:
            for point_coords in positive_point_coords_list:
                if point_coords is not None and len(point_coords) > 0:
                    axes[1].scatter(point_coords[:, 0], point_coords[:, 1], c="yellow", s=22, edgecolors="black", linewidths=0.5)

        if negative_point_coords_list is not None:
            for point_coords in negative_point_coords_list:
                if point_coords is not None and len(point_coords) > 0:
                    axes[1].scatter(point_coords[:, 0], point_coords[:, 1], c="cyan", s=20, edgecolors="black", linewidths=0.5)

        axes[1].set_title("RF-DETR Detections + Prompts")
        axes[1].axis("off")

        axes[2].imshow(image)
        combined_mask = np.zeros(image.shape[:2], dtype=bool)
        for mask in masks:
            combined_mask |= mask.astype(bool)
        mask_overlay = np.zeros((*combined_mask.shape, 4))
        mask_overlay[combined_mask] = [1, 0, 0, 0.5]
        axes[2].imshow(mask_overlay)
        axes[2].set_title("SAM2 Segmentation")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Crack Detection + SAM2 Segmentation Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--rfdetr-checkpoint", type=str, required=True, help="Path to RF-DETR checkpoint")
    parser.add_argument("--sam2-checkpoint", type=str, required=True, help="Path to SAM2 checkpoint")
    parser.add_argument("--sam2-config", type=str, default="sam2_hiera_t.yaml", help="SAM2 config")
    parser.add_argument("--output", type=str, default="output_crack_segmentation.png", help="Output path")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--nms-iou", type=float, default=0.1, help="RF-DETR NMS IoU threshold (<=0 disables NMS)")
    parser.add_argument("--max-cracks", type=int, default=3, help="Max non-overlapping detections to segment")
    parser.add_argument("--num-dark-points", "--num-positive-points", dest="num_dark_points", type=int, default=6)
    parser.add_argument("--num-negative-points", type=int, default=3)
    parser.add_argument("--prompt-mode", type=str, default="box+points", choices=["box", "points", "box+points"])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-classes", type=int, default=None, help="Optional RF-DETR num_classes override")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    pipeline = CrackDetectionPipeline(
        rfdetr_checkpoint=args.rfdetr_checkpoint,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        device=args.device,
        num_classes=args.num_classes,
    )

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
        print("\nPipeline completed successfully")
    else:
        print("\nNo cracks detected in image")


if __name__ == "__main__":
    main()

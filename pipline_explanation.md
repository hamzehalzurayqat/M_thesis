# Pipeline Code — Step by Step Explanation

---

## Step 1 — RF-DETR Detection (`detect_cracks`)

The image is loaded, preprocessed, and passed through RF-DETR. Each object query outputs one bounding box and a class logit vector. The no-object class (last column) is excluded before scoring.

```python
# Load and preprocess image
image = Image.open(image_path).convert('RGB')
image_np = np.array(image)

transformed, _ = self.transform(image, None)
samples = nested_tensor_from_tensor_list([transformed])
samples = samples.to(self.device)

# Forward pass
with torch.no_grad():
    outputs = self.rfdetr_model(samples)

# Extract predictions
pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
pred_boxes  = outputs['pred_boxes'][0]   # [num_queries, 4] — normalised (cx, cy, w, h)

# Apply temperature scaling then softmax — exclude no-object class (last column)
with torch.no_grad():
    scaled_logits = self.temperature_scaler(pred_logits)
    probs = scaled_logits.softmax(-1)[:, :-1]
    scores, labels = probs.max(-1)

# Filter by confidence threshold
keep   = scores > confidence_threshold
boxes  = pred_boxes[keep]
scores = scores[keep]
```

**Box format conversion** — RF-DETR outputs normalised `(cx, cy, w, h)`, converted to pixel `(x1, y1, x2, y2)`:

```python
def _box_cxcywh_to_xyxy(boxes, img_w, img_h):
    cx, cy, w, h = boxes.unbind(-1)

    # Denormalise to pixel coordinates
    cx = cx * img_w
    cy = cy * img_h
    w  = w  * img_w
    h  = h  * img_h

    # Convert center format to corner format
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=-1)
```

**NMS** — removes duplicate overlapping boxes:

```python
def _nms_torch(cls, boxes_xyxy, scores, iou_threshold):
    order = torch.argsort(scores, descending=True)
    keep  = []

    while order.numel() > 0:
        current = order[0]
        keep.append(current)

        if order.numel() == 1:
            break

        # Compute IoU between current box and all remaining boxes
        remaining = order[1:]
        ious = cls._iou_one_to_many(boxes_xyxy[current], boxes_xyxy[remaining])

        # Keep only boxes with IoU below threshold
        order = remaining[ious <= iou_threshold]

    return torch.stack(keep)
```

---

## Step 2 — Positive Prompt Sampling (`sample_darkest_points`)

Selects 6 positive point prompts inside each detected box to guide SAM2 toward crack pixels. Only dark, locally-dark, line-like pixels are candidates. Farthest Point Sampling spreads them along the crack.

```python
# Clip box to image bounds
x1, y1, x2, y2 = self._clip_box_to_image(box_xyxy, image.shape)

# Convert to grayscale and crop region of interest
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
roi  = gray[y1:y2 + 1, x1:x2 + 1]

# Candidate filtering — dark AND surrounded by dark neighbors
dark_threshold         = np.percentile(roi, darkest_percentile)        # bottom 10%
neighborhood_mean      = cv2.blur(roi.astype(np.float32), (3, 3))      # 3x3 local mean
neighborhood_threshold = np.percentile(neighborhood_mean, min(30.0, darkest_percentile + 8.0))

dark_mask              = roi <= dark_threshold
neighborhood_dark_mask = neighborhood_mean <= neighborhood_threshold

candidate_mask   = dark_mask & neighborhood_dark_mask   # both conditions must hold
candidates_local = np.argwhere(candidate_mask)           # (row, col) of each candidate
```

**Composite scoring** — each candidate pixel gets a score from four factors:

```python
roi_h, roi_w    = roi.shape
max_border_dist = max(1.0, min(roi_h, roi_w) / 2.0)

for yy, xx in candidates_local:
    intensity = float(roi[yy, xx])
    neigh     = float(neighborhood_mean[yy, xx])

    # Factor 1 — how dark the pixel is (weight 0.40)
    darkness_score = (255.0 - intensity) / 255.0

    # Factor 2 — how dark the surrounding area is (weight 0.20)
    neighborhood_score = (255.0 - neigh) / 255.0

    # Factor 3 — lies on a linear dark structure (weight 0.30)
    line_score = self._line_continuity_score(roi, int(yy), int(xx), line_dark_threshold)

    # Factor 4 — not too close to bounding box boundary (weight 0.10)
    border_dist  = float(min(yy, xx, roi_h - 1 - yy, roi_w - 1 - xx))
    center_score = np.clip(border_dist / max_border_dist, 0.0, 1.0)

    final_score = (
        0.40 * darkness_score
      + 0.20 * neighborhood_score
      + 0.30 * line_score
      + 0.10 * center_score
    )
    candidate_scores.append(final_score)
```

**Sort and spread with FPS:**

```python
# Sort candidates by score — best first
candidate_scores = np.asarray(candidate_scores, dtype=np.float32)
order            = np.argsort(-candidate_scores)
candidates_local = candidates_local[order]

# Keep top candidates only
top_n            = min(len(candidates_local), max(200, num_points * 60))
candidates_local = candidates_local[:top_n]

# Convert local (row, col) to absolute image (x, y)
candidates_xy = np.stack(
    [candidates_local[:, 1] + x1,   # col -> absolute x
     candidates_local[:, 0] + y1],  # row -> absolute y
    axis=1,
).astype(np.float32)

# Spread 6 prompts along the crack using Farthest Point Sampling
sampled_xy   = self._farthest_point_sample(candidates_xy, num_points)
point_labels = np.ones((len(sampled_xy),), dtype=np.int32)  # 1 = positive
```

---

## Step 3 — Line Continuity Score (`_line_continuity_score`)

Checks whether a pixel lies on a crack by probing dark pixel continuity in 8 directions. Both sides of a direction must be dark for a strong score — this distinguishes a crack (continuous dark line) from an isolated dark blob.

```python
def _line_continuity_score(roi_gray, y, x, line_dark_threshold):
    h, w = roi_gray.shape

    # 8 directions covering 0 to 153 degrees in roughly equal steps
    directions = [
        (1, 0),   # 0 degrees
        (2, 1),   # ~26 degrees
        (1, 1),   # 45 degrees
        (1, 2),   # ~63 degrees
        (0, 1),   # 90 degrees
        (-1, 2),  # ~117 degrees
        (-1, 1),  # 135 degrees
        (-2, 1),  # ~153 degrees
    ]

    best = 0.0
    for dx, dy in directions:
        side_scores = []

        # Check both sides of the current pixel along this direction
        for sign in (-1, 1):
            dark_hits = 0
            valid     = 0

            # Take 2 steps out in each direction
            for step in (1, 2):
                xx = x + sign * dx * step
                yy = y + sign * dy * step

                if 0 <= xx < w and 0 <= yy < h:
                    valid += 1
                    if roi_gray[yy, xx] <= line_dark_threshold:
                        dark_hits += 1

            side_scores.append((dark_hits / valid) if valid > 0 else 0.0)

        # min() enforces BOTH sides must be dark — not just one
        direction_score = min(side_scores)
        if direction_score > best:
            best = direction_score

    return float(best)
```

---

## Step 4 — Negative Prompt Sampling (`sample_negative_points_in_and_around_box`)

Samples 3 negative points outside the bounding box to tell SAM2 where the crack is not. Only bright pixels qualify. Three distance zones concentrate coverage near the critical boundary edge.

```python
# Expand box outward by 15% on each side to define ring region
box_w = x2 - x1 + 1
box_h = y2 - y1 + 1
pad_x = max(3, int(round(box_w * ring_ratio)))   # ring_ratio = 0.15
pad_y = max(3, int(round(box_h * ring_ratio)))

h, w     = image.shape[:2]
outer_x1 = max(0, x1 - pad_x)
outer_y1 = max(0, y1 - pad_y)
outer_x2 = min(w - 1, x2 + pad_x)
outer_y2 = min(h - 1, y2 + pad_y)

# Build candidate grid then keep only pixels OUTSIDE the original box
yy_xx        = np.argwhere(np.ones_like(outer_roi, dtype=bool))
yy           = yy_xx[:, 0] + outer_y1
xx           = yy_xx[:, 1] + outer_x1
outside_mask = (xx < x1) | (xx > x2) | (yy < y1) | (yy > y2)
yy           = yy[outside_mask]
xx           = xx[outside_mask]

# Keep only bright pixels — background is typically brighter than crack
intensities     = gray[yy, xx].astype(np.float32)
white_threshold = np.percentile(intensities, white_percentile)   # 85th percentile
bright_mask     = intensities >= white_threshold
xx              = xx[bright_mask]
yy              = yy[bright_mask]
```

**Distance-from-edge zones with fixed allocation:**

```python
# Distance of each outside pixel to the nearest box boundary
dx = np.where(xx < x1, x1 - xx, np.where(xx > x2, xx - x2, 0))
dy = np.where(yy < y1, y1 - yy, np.where(yy > y2, yy - y2, 0))
edge_dist = np.sqrt(dx.astype(np.float32)**2 + dy.astype(np.float32)**2)

# Three zones — near boundary matters most
near_mask = edge_dist <= 5.0                              # <=5 px   -> 40%
mid_mask  = (edge_dist > 5.0) & (edge_dist <= 15.0)      # 5-15 px  -> 40%
far_mask  = edge_dist > 15.0                              # >15 px   -> 20%

candidates_xy = np.stack([xx, yy], axis=1).astype(np.float32)

# FPS within each zone for spatial diversity
near_pts = _farthest_point_sample(candidates_xy[near_mask], n_near)
mid_pts  = _farthest_point_sample(candidates_xy[mid_mask],  n_mid)
far_pts  = _farthest_point_sample(candidates_xy[far_mask],  n_far)

sampled_xy   = np.concatenate([near_pts, mid_pts, far_pts], axis=0)
point_labels = np.zeros((len(sampled_xy),), dtype=np.int32)   # 0 = negative
```

---

## Step 5 — Farthest Point Sampling (`_farthest_point_sample`)

Greedy algorithm that picks k points maximally spread in space. Each iteration picks the point farthest from all already-selected points. Guarantees spatial coverage regardless of candidate distribution.

```python
def _farthest_point_sample(points_xy, k):
    n = len(points_xy)
    if n <= k:
        return points_xy          # fewer candidates than requested — return all

    selected  = [0]               # always start from index 0
    min_dists = np.full(n, np.inf, dtype=np.float32)

    for _ in range(1, k):
        last = points_xy[selected[-1]]

        # Squared distance from every point to the last selected point
        d2 = np.sum((points_xy - last) ** 2, axis=1)

        # Update minimum distance to ANY selected point so far
        min_dists = np.minimum(min_dists, d2)

        # Pick the point farthest from all selected points
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)

    return points_xy[np.array(selected, dtype=np.int32)]
```

---

## Step 6 — SAM2 Segmentation (`segment_crack`)

Combines the bounding box from RF-DETR with positive and negative point prompts into a single SAM2 call. Returns a binary pixel mask.

```python
# Sample positive prompts — dark crack pixels inside the box
pos_point_coords, pos_point_labels = self.sample_darkest_points(
    image              = image,
    box_xyxy           = box_xyxy,
    num_points         = num_dark_points,      # 6
    darkest_percentile = 10.0,
)

# Sample negative prompts — bright background pixels outside the box
neg_point_coords, neg_point_labels = self.sample_negative_points_in_and_around_box(
    image                 = image,
    box_xyxy              = box_xyxy,
    num_points            = num_negative_points,   # 3
    positive_point_coords = pos_point_coords,
    ring_ratio            = 0.15,
    white_percentile      = 85.0,
)

# Merge positive and negative into one array
point_coords = np.concatenate([pos_point_coords, neg_point_coords], axis=0)
point_labels = np.concatenate([
    np.ones (len(pos_point_coords), dtype=np.int32),   # 1 = positive (crack)
    np.zeros(len(neg_point_coords), dtype=np.int32),   # 0 = negative (background)
])

# Set SAM2 image encoder — runs once per image
self.sam2_predictor.set_image(image)

# Run SAM2 with box + point prompts
masks, scores, _ = self.sam2_predictor.predict(
    box              = box_xyxy,       # bounding box from RF-DETR
    point_coords     = point_coords,   # positive + negative points combined
    point_labels     = point_labels,   # 1=crack  0=background
    multimask_output = False,          # return single best mask
)

mask  = masks[0]    # binary mask — True where crack is predicted
score = scores[0]   # SAM2 confidence for this mask
```

---

## Full Pipeline Flow

```python
def run_pipeline(self, image_path, confidence_threshold=0.85, ...):

    # Stage 1 — Detection
    boxes, scores, image = self.detect_cracks(
        image_path,
        confidence_threshold = confidence_threshold,
        nms_iou_threshold    = 0.5,
    )
    # boxes  -> [N, 4]  pixel (x1, y1, x2, y2)
    # scores -> [N]     confidence per detection

    # Stage 2 — Segmentation (one SAM2 call per detected box)
    masks = []
    for box, score in zip(boxes, scores):
        result = self.segment_crack(
            image               = image,
            box_xyxy            = box,
            num_dark_points     = 6,
            num_negative_points = 3,
            prompt_mode         = 'box+points',
        )
        masks.append(result['mask'])

    return {'boxes': boxes, 'scores': scores, 'masks': masks, 'image': image}
```

```
Image
  |
  v
detect_cracks()
  |-- RF-DETR forward pass
  |-- softmax + confidence threshold
  +-- NMS -> boxes, scores
        |
        v  for each box:
  |-- sample_darkest_points()
  |     |-- dark + neighborhood candidate filter
  |     |-- composite score (darkness / neighborhood / line / center)
  |     +-- FPS -> 6 positive prompts
  |
  |-- sample_negative_points_in_and_around_box()
  |     |-- ring region outside box
  |     |-- bright pixel filter
  |     |-- near / mid / far zone split
  |     +-- FPS per zone -> 3 negative prompts
  |
  +-- segment_crack()
        |-- merge positive + negative prompts
        |-- SAM2 predict(box, point_coords, point_labels)
        +-- binary mask output
```

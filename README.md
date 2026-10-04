# Automated Crack Detection and Segmentation Using RF-DETR and SAM2

---

## Overview

A two-stage pipeline that combines **RF-DETR** (detection) with **SAM2** (segmentation) to produce pixel-level crack masks using only bounding-box annotations for training — no pixel-level masks required.

**Key idea:** RF-DETR detects crack regions as bounding boxes → bounding boxes + sampled point prompts are passed to SAM2 → SAM2 outputs pixel-level crack masks without any crack-specific fine-tuning.


---

## How It Works

```
Input Image
    |
    v
RF-DETR Detector (modified)
    |-- DINOv2 backbone — multi-scale outputs 
    |-- Feature Pyramid P2-P6
    |-- Deformable Attention Encoder
    +-- Detection Head + NMS  ->  bounding boxes
                                        |
                                        v
                            Prompt Construction
                                |-- box prompt
                                |-- 6 positive points (dark crack pixels)
                                +-- 3 negative points (background ring)
                                        |
                                        v
                            SAM2 (Hiera-B+) zero-shot
                                        |
                                        v
                            Pixel-level crack masks
```

For a detailed explanation of each stage with full code, see [`pipeline_explanation.md`](pipeline_explanation.md).

---

## Modifications to RF-DETR

Three targeted changes were made to the standard RF-DETR base. Full before/after code diffs are in [`code_edit.md`](code_edit.md).

**1. P2 feature level** — extends the feature pyramid from P3–P6 to P2–P6, adding stride-4 resolution to detect thin and small cracks.

**2. Multi-scale backbone features** — extracts features from DINOv2 stages `[3, 6, 9, 12]` instead of only the final stage, fusing fine detail with deep semantic context.

**3. Deformable attention encoder** — replaces the absent encoder with a multi-scale deformable attention encoder for adaptive cross-scale reasoning.

---


## Training for full mode

```
import sys
import os
from pathlib import Path
import subprocess
import sys

# Set up paths
HOME = Path.home()
PROJECT_DIR = Path(".....")
# Add RF-DETR and SAM2 to path
sys.path.insert(0, str(PROJECT_DIR / "rf-detr"))
sys.path.insert(0, str(PROJECT_DIR / "sam2"))

train_cmd = [
    sys.executable,
    "-m", "rfdetr.main",
    "--encoder", "dinov2_base",
    "--dataset_file", "coco",
    "--dataset_dir", "C:/Users/20228014/Desktop/crack-augmented",
    "--output_dir", "../outputfull34enc0"
    "/crack_detection",
    "--batch_size", "2",
    "--grad_accum_steps","8",
    "--epochs", "70",
    "--lr", "5e-5",
    "--num_classes", "5",
    "--projector_scale","P2","P3", "P4", "P5", "P6",
    "--enc_layers", "3",
    "--num_feature_levels", "5",
    "--dec_layers", "4",
    "--num_queries", "50",
    "--resolution", "512",               
    "--print_freq", "1",
    "--two_stage",
    "--device", "cuda",
    "--num_workers", "0",
    "--square_resize_div_64",
    "--eval_max_dets", "100"
    ]

print("⏳ Starting training...")
subprocess.run(train_cmd, cwd=str(PROJECT_DIR / "rf-detr"), check=True)
print("✓ Training completed!")

```
[`dataset`](https://drive.google.com/drive/folders/1Y9z8WXlv8zS2GhndwcqsF1OZ6Perlcbb?usp=drive_link). 

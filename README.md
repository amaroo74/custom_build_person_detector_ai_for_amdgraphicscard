# Person detector from scratch

A small person detector written and trained from scratch in PyTorch, with no pretrained weights and no detection libraries. It is the first part of a planned navigation aid for blind users; speech output and obstacle detection come later.

The model has about 1.4 million parameters, runs at about 30 frames per second on a laptop CPU, and was trained on an AMD GPU through DirectML.

> **Prototype, not a safety device.** The detector misses most distant people and has only been tested indoors on one webcam. Nobody should rely on it to move around safely.

## Results

Measured with `evaluate_v5.py` on 270 COCO val2017 images containing 1,022 persons. A detection counts as correct if its IoU with an unmatched person is at least 0.5 (NMS 0.45).

| Confidence | Precision | Recall | F1 |
|---|---|---|---|
| 0.50 | 0.418 | 0.456 | 0.436 |
| 0.60 | 0.514 | 0.422 | 0.463 |
| **0.70** | 0.637 | 0.395 | **0.488** |
| 0.80 | 0.768 | 0.341 | 0.472 |
| 0.90 | 0.921 | 0.263 | 0.409 |

Persons found by size at confidence 0.70 (size is the longer side of the box in the 320×320 input):

| Size | Persons | Found |
|---|---|---|
| Tiny (under 32 px) | 282 | 1.1% |
| Small (32–96 px) | 371 | 29.4% |
| Medium (96–192 px) | 197 | 71.6% |
| Large (192 px and over) | 172 | 87.8% |

For reference, a pretrained SSDLite320 scored an F1 of about 0.53 on the same images. That figure came from a different script, so the comparison is approximate. With only 270 images, F1 differences of a few hundredths are within noise.

In live use on a laptop webcam, a standing person is detected up to roughly 5 to 6 metres away.

## How it works

- **Input**: the whole frame is resized to 320×320 (aspect ratio not kept), RGB, values 0 to 1.
- **Network** (`model_v4.py`, `model_v5.py`): a stride-2 stem, then convolution and residual stages down to a 20×20 grid, plus a coarser 10×10 branch that is upsampled and fused in.
- **Output**: a `[15, 20, 20]` tensor. Each 16 px grid cell has 3 slots, and each slot predicts x, y, width, height and an objectness score.
- **Labels** (`targets_v5.py`): every cell in the central part of a person's box (up to 5×5 cells) is a positive and predicts the same full box. Tiny and small persons get one cell. In V5, x and y can range from −2.5 to 3.5 cells, so a cell next to a person's centre can still point at it.
- **Loss** (`loss_v3_localization.py`): Smooth-L1 plus GIoU for boxes, weighted binary cross-entropy for objectness.
- **Post-processing**: NMS at 0.45, then a merge rule that drops a box when 60% or more of it (or of the other box) lies inside a more confident box. This stops one moving person from being counted two or three times.

### What the merge rule costs

On the 270 still images at confidence 0.70:

| Merge threshold | Real persons lost | Wrong boxes removed | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Off | | | 0.637 | 0.395 | 0.488 |
| 0.60 (demo default) | 30 | 63 | 0.691 | 0.366 | 0.479 |
| 0.70 | 18 | 53 | 0.686 | 0.378 | 0.487 |

The rule trades some recall for precision. The persons it loses are people standing in front of or right beside someone else.

## Files

| File | Purpose |
|---|---|
| `model_v4.py`, `model_v5.py` | Network definitions (V5 reuses the V4 layers with a wider x, y range) |
| `targets.py`, `targets_v5.py` | Label encoders (V4 single cell, V5 central region) |
| `loss_v3_localization.py` | Loss function |
| `train_v4_big.py` | Training script |
| `test_v5.py` | Checks to run before training, including CPU versus DirectML gradients |
| `evaluate_v5.py` | Evaluator for V4 and V5 checkpoints |
| `live_demo.py` | Webcam, image or video demo |
| `person_detector_v4_big_neg_v5_full.pth` | Trained weights (epoch 10, EMA) |
| `train_v4_big_log_neg_v5_full.csv` | Epoch-by-epoch log of that training run |

## Setup

The project was developed with Python 3.12 on Windows.

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision opencv-python numpy pillow pycocotools torch-directml
```

`torch-directml` is only needed for training on an AMD or Intel GPU, for `test_v5.py`, and for the demo's `--dml` option. The demo and the evaluator run on the CPU without it.

Training, testing and evaluating need COCO 2017 in a `dataset` folder next to this repository's folder:

```
parent folder
├── dataset
│   ├── train2017
│   ├── val2017
│   └── annotations
└── this repository
```

## Usage

Run every command from the repository folder.

### Live demo

```powershell
python live_demo.py
```

| Option | Meaning |
|---|---|
| `--conf=0.70` | Confidence threshold at start |
| `--merge=0.60` | Merge rule threshold; `--merge=0` starts with it off |
| `--camera=1` | Use another camera |
| `--image=photo.jpg` | Run on one picture |
| `--video=clip.mp4` | Run on a video file |
| `--mirror` | Flip the picture left to right |
| `--dml` | Run the model on DirectML |

Keys while the window is open: `q` or `Esc` quits, `+` and `-` change the threshold, `m` switches the merge rule on and off, `s` saves a picture.

### Evaluate

```powershell
python evaluate_v5.py person_detector_v4_big_neg_v5_full.pth
```

This prints the confidence sweep, the per-size results and the merge rule comparison shown above. It picks the model class from `_v5` in the file name; a V5 checkpoint loaded as V4 gives no error, only wrong boxes.

### Check before training

```powershell
python test_v5.py
```

It must end with `ALL CHECKS PASSED` before a training run is started.

### Train

```powershell
python train_v4_big.py --negatives --v5 --images=64115 --epochs=10 --tag=full
```

This is the command that produced the included weights. It uses 64,115 person images and 21,371 person-free images from train2017 at batch size 8, and took about 5 hours on a laptop AMD GPU. Other options: `--resume`, `--smoke`.

## Lessons learned

- **DirectML and in-place assignment**: in-place slice assignment in `forward` produced wrong box gradients on DirectML without any error, and quietly broke a whole series of earlier experiments. Outputs are now built with `torch.cat`, and `test_v5.py` compares CPU and DirectML gradients for every parameter.
- **One positive cell is not enough for large persons**: with a single labelled cell, large persons got good boxes but low confidence, because many neighbouring cells looked the same and were labelled negative. Labelling the central region raised the share of large persons found from 30% to 81%.
- **More data helped**: going from 20,000 to all 64,115 person images raised F1 from 0.41 to 0.49 with no other change.

## Known limits and next steps

- **Small and distant persons** make up 64% of the persons in the validation images and are mostly missed. This is a limit of the 20×20 grid; a finer 40×40 output is the planned fix.
- **A raised hand or arm** can be detected as a second person. The merge rule reduces this but does not remove it entirely.
- **Speech output** is next: announcing a detected person as left, centre or right.
- **Distance** will come from a dedicated sensor, not from the camera.
- **Obstacle classes** (poles, steps, vehicles, furniture) need a multi-class model and more data.

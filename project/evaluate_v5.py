"""
evaluate_v5.py  (READ-ONLY - changes none of your files)

Run from : C:\\Users\\Nandhu\\Desktop\\amar\\project
Commands :
  ..\\venv\\Scripts\\python.exe .\\evaluate_v5.py person_detector_v4_big_neg_v5_short.pth
  ..\\venv\\Scripts\\python.exe .\\evaluate_v5.py person_detector_v4_big_neg_short.pth

One evaluator for BOTH kinds of checkpoint, so the numbers are directly comparable:
  * file name contains "_v5"  -> loaded with PersonDetectorV5 (x/y range -2.5 .. 3.5)
  * otherwise                 -> loaded with PersonDetectorV4 (x/y range 0 .. 1)
  * add --v4 or --v5 at the end to force one of them
A V5 checkpoint loaded as V4 (or the other way round) gives NO error, only wrong
boxes - that is why this script prints which one it used.

Same 270 validation images, same ground truth and same matching rule as
evaluate_v4.py (a detection is correct if IoU >= 0.5 with a not-yet-matched person,
detections taken from the most confident down). It runs on the CPU (about a minute).

It prints
  1. precision / recall / F1 for several confidence thresholds (NMS 0.45)
  2. recall per person size at the best-F1 threshold
  3. the merge rule (also in live_demo.py): the same numbers with it off and at
     0.50 / 0.60 / 0.70, so its cost in lost real persons is measured
The 0.10 row can differ slightly from evaluate_v4.py, which uses OpenCV's NMS.
Sections 1 and 2 are unchanged and never use the merge rule.
"""
import os
import sys

import numpy as np
import torch
import torchvision
from PIL import Image
from pycocotools.coco import COCO

IMAGE_SIZE = 320
GRID = 20
CELL = IMAGE_SIZE / GRID
IMAGE_DIR = r"..\dataset\val2017"
ANN = r"..\dataset\annotations\instances_val2017.json"
MAX_IMAGES = 270
NMS_THRESHOLD = 0.45
IOU_THRESHOLD = 0.50
THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
MIN_SCORE = 0.05
MERGE_VALUES = [0.50, 0.60, 0.70]
MERGE_CONFS = [0.60, 0.70, 0.80]

BUCKETS = [
    ("tiny   (< 32 px)", 0, 32),
    ("small  (32-96)", 32, 96),
    ("medium (96-192)", 96, 192),
    ("large  (>= 192)", 192, 100000),
]

# ------------------------------------------------------------
# arguments
# ------------------------------------------------------------
args = [a for a in sys.argv[1:] if not a.startswith("--")]
if not args:
    print("Give the checkpoint file, for example:")
    print("  ..\\venv\\Scripts\\python.exe .\\evaluate_v5.py person_detector_v4_big_neg_v5_short.pth")
    sys.exit(1)
MODEL_PATH = args[0]
if not os.path.exists(MODEL_PATH):
    print("Checkpoint not found:", MODEL_PATH)
    print("Checkpoints in this folder:")
    for name in sorted(os.listdir(".")):
        if name.endswith(".pth"):
            print("  ", name)
    sys.exit(1)

if "--v4" in sys.argv:
    use_v5 = False
elif "--v5" in sys.argv:
    use_v5 = True
else:
    use_v5 = "_v5" in os.path.basename(MODEL_PATH).lower()

if use_v5:
    from model_v5 import PersonDetectorV5 as ModelClass
else:
    from model_v4 import PersonDetectorV4 as ModelClass

# ------------------------------------------------------------
# data: the same 270 validation images as every earlier experiment
# ------------------------------------------------------------
coco = COCO(ANN)
all_ids = coco.getImgIds(catIds=[1])
train_size = int(len(all_ids) * 0.9)
perm = torch.randperm(len(all_ids), generator=torch.Generator().manual_seed(42)).tolist()
val_ids = [all_ids[i] for i in perm[train_size:]][:MAX_IMAGES]


def load_gt(image_id):
    info = coco.loadImgs(image_id)[0]
    W, H = info["width"], info["height"]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], catIds=[1], iscrowd=False))
    boxes = []
    for a in anns:
        x, y, w, h = a["bbox"]
        x1 = max(0.0, min(x, W)); y1 = max(0.0, min(y, H))
        x2 = max(0.0, min(x + w, W)); y2 = max(0.0, min(y + h, H))
        if x2 > x1 and y2 > y1:
            sx, sy = IMAGE_SIZE / W, IMAGE_SIZE / H
            boxes.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
    return info["file_name"], boxes


# ------------------------------------------------------------
# model
# ------------------------------------------------------------
model = ModelClass(num_boxes=3)
ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
model.eval()

print()
print("Checkpoint :", MODEL_PATH, "| epoch", ck.get("epoch", "?") if isinstance(ck, dict) else "?")
print("Loaded as  :", "PersonDetectorV5 (x/y range -2.5 .. 3.5)" if use_v5
      else "PersonDetectorV4 (x/y range 0 .. 1)")
print("Images     :", len(val_ids))

gy, gx = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij")
gx = gx.float().unsqueeze(0)
gy = gy.float().unsqueeze(0)

# ------------------------------------------------------------
# inference: keep every candidate above MIN_SCORE, once per image
# ------------------------------------------------------------
per_image = []
total_gt = 0
for n, iid in enumerate(val_ids, start=1):
    fname, gts = load_gt(iid)
    total_gt += len(gts)

    img = Image.open(os.path.join(IMAGE_DIR, fname)).convert("RGB").resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        out = model(x)[0].reshape(3, 5, GRID, GRID)

    prob = torch.sigmoid(out[:, 4])
    cx = (gx + out[:, 0]) * CELL
    cy = (gy + out[:, 1]) * CELL
    w = out[:, 2] * IMAGE_SIZE
    h = out[:, 3] * IMAGE_SIZE
    boxes = torch.stack([
        (cx - w / 2).clamp(0, IMAGE_SIZE), (cy - h / 2).clamp(0, IMAGE_SIZE),
        (cx + w / 2).clamp(0, IMAGE_SIZE), (cy + h / 2).clamp(0, IMAGE_SIZE),
    ], dim=-1).reshape(-1, 4)
    prob = prob.reshape(-1)

    keep = (prob >= MIN_SCORE) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    per_image.append((boxes[keep], prob[keep], torch.tensor(gts, dtype=torch.float32).reshape(-1, 4)))

    if n % 90 == 0 or n == len(val_ids):
        print(f"  processed {n}/{len(val_ids)} images")


def merge_contained(boxes, threshold):
    """Second clean-up step after NMS.
    boxes: list of [x1, y1, x2, y2], most confident first (the order NMS returns).
    A box is dropped when the part it shares with an already kept (more confident)
    box is at least `threshold` of the SMALLER of the two boxes. NMS misses these
    because a small box inside a big one has a low IoU.
    Returns the positions of the boxes to keep. threshold <= 0 keeps everything."""
    if threshold <= 0:
        return list(range(len(boxes)))
    keep = []
    for i, a in enumerate(boxes):
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        duplicate = False
        for j in keep:
            b = boxes[j]
            iw = min(a[2], b[2]) - max(a[0], b[0])
            ih = min(a[3], b[3]) - max(a[1], b[1])
            if iw <= 0 or ih <= 0:
                continue
            smaller = min(area_a, (b[2] - b[0]) * (b[3] - b[1]))
            if smaller > 0 and iw * ih / smaller >= threshold:
                duplicate = True
                break
        if not duplicate:
            keep.append(i)
    return keep


def evaluate(conf, merge=0.0):
    """Returns TP, FP, FN, matched IoUs and, for every person, whether it was found."""
    tp = fp = 0
    ious = []
    found_sizes = []
    for boxes, scores, gts in per_image:
        m = scores >= conf
        b, s = boxes[m], scores[m]
        if len(b):
            order = torchvision.ops.nms(b, s, NMS_THRESHOLD)   # highest score first
            b = b[order]
            if merge > 0:
                b = b[merge_contained(b.tolist(), merge)]
        matched = torch.zeros(len(gts), dtype=torch.bool)
        if len(b) and len(gts):
            iou = torchvision.ops.box_iou(b, gts)
            for i in range(len(b)):
                row = iou[i].clone()
                row[matched] = -1.0
                best = int(row.argmax())
                if float(row[best]) >= IOU_THRESHOLD:
                    matched[best] = True
                    tp += 1
                    ious.append(float(row[best]))
                else:
                    fp += 1
        else:
            fp += len(b)
        for g, hit in zip(gts.tolist(), matched.tolist()):
            found_sizes.append((max(g[2] - g[0], g[3] - g[1]), hit))
    fn = total_gt - tp
    return tp, fp, fn, ious, found_sizes


# ------------------------------------------------------------
# 1. threshold sweep
# ------------------------------------------------------------
print()
print("=" * 78)
print(f"1. Confidence sweep   (NMS {NMS_THRESHOLD}, match IoU >= {IOU_THRESHOLD}, persons: {total_gt})")
print("=" * 78)
print(f"{'conf':>5s} | {'TP':>5s} {'FP':>6s} {'FN':>5s} | {'precision':>9s} {'recall':>7s} {'F1':>7s} | {'mean IoU':>8s}")
print("-" * 78)
best = None
for conf in THRESHOLDS:
    tp, fp, fn, ious, found = evaluate(conf)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    print(f"{conf:5.2f} | {tp:5d} {fp:6d} {fn:5d} | {precision:9.4f} {recall:7.4f} {f1:7.4f} | {mean_iou:8.4f}")
    if best is None or f1 > best[0]:
        best = (f1, conf, found)

# ------------------------------------------------------------
# 2. recall per person size at the best threshold
# ------------------------------------------------------------
f1, conf, found = best
print()
print("=" * 78)
print(f"2. Persons found per size, at the best-F1 threshold (conf {conf:.2f}, F1 {f1:.4f})")
print("   size = longer side of the person box in the 320x320 image")
print("=" * 78)
sizes = np.array([s for s, _ in found], dtype=float)
hits = np.array([h for _, h in found], dtype=float)
for name, lo, hi in BUCKETS:
    sel = (sizes >= lo) & (sizes < hi)
    if sel.sum() == 0:
        print(f"{name:18s} {0:5d}")
    else:
        print(f"{name:18s} {int(sel.sum()):5d} persons | found {hits[sel].mean() * 100:5.1f}%")
print("-" * 78)
print(f"{'ALL persons':18s} {len(sizes):5d} persons | found {hits.mean() * 100:5.1f}%")

# ------------------------------------------------------------
# 3. merge rule: what it costs and what it removes
# ------------------------------------------------------------
def scores(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


print()
print("=" * 78)
print("3. Merge rule (drop a box mostly inside a more confident box), after NMS")
print("   TP lost = real persons removed by the rule, FP removed = wrong boxes removed")
print("=" * 78)
print(f"{'conf':>5s} {'merge':>6s} | {'TP':>5s} {'FP':>6s} {'FN':>5s} | {'precision':>9s} {'recall':>7s} {'F1':>7s} | "
      f"{'TP lost':>7s} {'FP removed':>10s}")
print("-" * 78)
for conf in MERGE_CONFS:
    tp0, fp0, fn0, _, _ = evaluate(conf)
    p, r, f = scores(tp0, fp0, fn0)
    print(f"{conf:5.2f} {'off':>6s} | {tp0:5d} {fp0:6d} {fn0:5d} | {p:9.4f} {r:7.4f} {f:7.4f} |")
    for mv in MERGE_VALUES:
        tp, fp, fn, _, _ = evaluate(conf, mv)
        p, r, f = scores(tp, fp, fn)
        print(f"{conf:5.2f} {mv:6.2f} | {tp:5d} {fp:6d} {fn:5d} | {p:9.4f} {r:7.4f} {f:7.4f} | "
              f"{tp0 - tp:7d} {fp0 - fp:10d}")
    print("-" * 78)

print()
print("Reference on the same images: pretrained SSDLite320 reached F1 0.53.")
print("Done. Please paste this whole output.")

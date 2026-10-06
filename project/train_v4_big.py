"""
train_v4_big.py - V4 trained on MORE DATA (COCO train2017 person images)

Same model (model_v4.py), same loss, same stability features as train_v4.py,
plus:
  * training images come from train2017 (default 30,000 persons images);
    the SAME 270 val2017 images as before are used for validation, and they
    never appear in training -> results stay directly comparable
  * background worker processes load/augment images (NUM_WORKERS)
  * a light dataset (plain Python lists), so workers do not copy the huge
    annotation file
  * GRAD_CLIP raised to 10 (your gradient norm is ~6, so clipping at 5 was
    active on almost every step)

Run from the project folder:

  ..\\venv\\Scripts\\python.exe .\\train_v4_big.py --check-aug   # aug_check_big.jpg
  ..\\venv\\Scripts\\python.exe .\\train_v4_big.py --smoke       # ~100 steps: speed + estimate
  ..\\venv\\Scripts\\python.exe .\\train_v4_big.py               # full training
  ..\\venv\\Scripts\\python.exe .\\train_v4_big.py --resume      # continue after interruption

  --negatives  also trains on train2017 images that contain NO person at all
               (cars, animals, poles, flags, rooms ...), as pure background.
               Without it the model never sees a person-free image. Output files
               get a "_neg" suffix, so nothing is overwritten:
                   person_detector_v4_big_neg.pth, train_v4_big_neg_log.csv, ...
               Example:  python train_v4_big.py --negatives
                         python train_v4_big.py --negatives --resume

  --soft       soft labels for the cells around medium/large persons (targets_v4.py,
               loss_v4.py). Validation LOSS values are not comparable with runs
               without --soft; compare AUC / IoU and the evaluation scripts.
               Example:  python train_v4_big.py --negatives --soft --images=20000 --epochs=8 --tag=short

  --v5         multi-cell labels: every cell in the central region of a person is a
               positive and predicts the full box (targets_v5.py + model_v5.py, which
               allows x/y offsets beyond the own cell). A different fix for the same
               large-person problem as --soft; use ONE of them per run.
               Example:  python train_v4_big.py --negatives --v5 --images=20000 --epochs=8 --tag=short
               NOTE: AUC / IoU / loss printed per epoch are NOT comparable with
               earlier runs (there are many more positive cells). Compare runs with
               the evaluation scripts (F1 on the 270 images), not with the epoch lines.

  --images=N   number of person images to use (default MAX_TRAIN_IMAGES)
  --epochs=N   number of epochs (default EPOCHS)
  --tag=NAME   adds _NAME to every output file name, so different runs never
               share checkpoint / resume / log files
               Short example (~80 min):
                   python train_v4_big.py --negatives --images=20000 --epochs=8 --tag=short
               IMPORTANT: always pass the SAME options again with --resume.

If background workers cause trouble on Windows, set NUM_WORKERS = 0.
"""
import copy
import csv
import functools
import gc
import math
import os
import random
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

from model_v4 import PersonDetectorV4
from targets import encode_targets
try:
    from targets_v4 import encode_targets_soft
except ImportError:
    encode_targets_soft = None
from loss_v3_localization import DetectionLossV3Localization
try:
    from loss_v4 import DetectionLossV4
except ImportError:
    DetectionLossV4 = None


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 320
GRID = 20
CELL = IMAGE_SIZE / GRID

TRAIN_IMAGE_DIR = r"..\dataset\train2017"
TRAIN_ANN = r"..\dataset\annotations\instances_train2017.json"
VAL_IMAGE_DIR = r"..\dataset\val2017"
VAL_ANN = r"..\dataset\annotations\instances_val2017.json"

MAX_TRAIN_IMAGES = 30000
NEGATIVE_FRACTION = 0.25     # with --negatives: share of training images that contain NO person
EPOCHS = 12
BATCH_SIZE = 8
NUM_WORKERS = 3
SEED = 42

PEAK_LR = 1e-3
MIN_LR = 1e-5
WARMUP_STEPS = 300
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 10.0
EMA_DECAY = 0.999

AUG_FLIP = True
AUG_CROP = True
AUG_COLOR = True

BEST_PATH = "person_detector_v4_big.pth"
LAST_PATH = "person_detector_v4_big_last.pth"
RESUME_PATH = "person_detector_v4_big_resume.pth"
LOG_PATH = "train_v4_big_log.csv"

SMOKE_STEPS = 100


# ============================================================
# Data (module-level helpers only; nothing heavy runs on import,
# because Windows worker processes import this file)
# ============================================================

def build_records(annotation_file, image_dir, skip_empty):
    """List of (path, width, height, [(x1,y1,x2,y2), ...]) for person images."""
    coco = COCO(annotation_file)
    records = []
    for image_id in coco.getImgIds(catIds=[1]):
        info = coco.loadImgs(image_id)[0]
        W, H = info["width"], info["height"]
        anns = coco.loadAnns(
            coco.getAnnIds(imgIds=[image_id], catIds=[1], iscrowd=False)
        )
        boxes = []
        for a in anns:
            x, y, w, h = a["bbox"]
            x1 = max(0.0, min(x, W)); y1 = max(0.0, min(y, H))
            x2 = max(0.0, min(x + w, W)); y2 = max(0.0, min(y + h, H))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        if skip_empty and not boxes:
            continue
        records.append((os.path.join(image_dir, info["file_name"]), W, H, boxes))
    del coco
    gc.collect()
    return records


def build_negative_records(annotation_file, image_dir):
    """Images with NO person annotation at all (crowd persons included)."""
    coco = COCO(annotation_file)
    person_ids = set(coco.getImgIds(catIds=[1]))
    records = []
    for image_id in coco.getImgIds():
        if image_id in person_ids:
            continue
        info = coco.loadImgs(image_id)[0]
        records.append(
            (os.path.join(image_dir, info["file_name"]), info["width"], info["height"], [])
        )
    del coco
    gc.collect()
    return records


def augment(image, boxes):
    """image [3,320,320] in 0..1, boxes [N,4] in 320-space (CPU tensors)."""

    boxes = boxes.reshape(-1, 4).clone()

    if AUG_CROP and random.random() < 0.5:
        keep_fraction = random.uniform(0.6, 1.0)
        crop = int(round(IMAGE_SIZE * keep_fraction))
        x0 = random.randint(0, IMAGE_SIZE - crop)
        y0 = random.randint(0, IMAGE_SIZE - crop)

        image = image[:, y0:y0 + crop, x0:x0 + crop]
        image = F.interpolate(
            image.unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        if boxes.shape[0] > 0:
            x1, y1, x2, y2 = boxes.unbind(dim=1)
            area_before = (x2 - x1) * (y2 - y1)

            x1 = (x1 - x0).clamp(0, crop)
            x2 = (x2 - x0).clamp(0, crop)
            y1 = (y1 - y0).clamp(0, crop)
            y2 = (y2 - y0).clamp(0, crop)

            area_after = (x2 - x1) * (y2 - y1)
            keep = (
                (area_after >= 0.4 * area_before)
                & ((x2 - x1) > 2)
                & ((y2 - y1) > 2)
            )
            scale = IMAGE_SIZE / crop
            boxes = torch.stack([x1, y1, x2, y2], dim=1)[keep] * scale

    if AUG_FLIP and random.random() < 0.5:
        image = image.flip(-1)
        if boxes.shape[0] > 0:
            new_x1 = IMAGE_SIZE - boxes[:, 2]
            new_x2 = IMAGE_SIZE - boxes[:, 0]
            boxes = torch.stack(
                [new_x1, boxes[:, 1], new_x2, boxes[:, 3]], dim=1
            )

    if AUG_COLOR:
        brightness = random.uniform(0.75, 1.25)
        contrast = random.uniform(0.75, 1.25)
        mean = image.mean()
        image = ((image - mean) * contrast + mean) * brightness
        image = image.clamp(0.0, 1.0)

    return image.contiguous(), boxes


class PersonRecords(Dataset):

    def __init__(self, records, augment_data):
        self.records = records
        self.augment_data = augment_data

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, W, H, boxes = self.records[index]

        image = Image.open(path).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        )
        image = torch.from_numpy(
            np.array(image, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).contiguous()

        sx, sy = IMAGE_SIZE / W, IMAGE_SIZE / H
        boxes = torch.tensor(
            [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in boxes],
            dtype=torch.float32
        ).reshape(-1, 4)

        if self.augment_data:
            image, boxes = augment(image, boxes)

        return image, boxes


def collate_fn(batch, soft=False, v5=False):
    if v5:
        from targets_v5 import encode_targets_v5   # imported lazily: only needed with --v5
        encoder = encode_targets_v5
    else:
        encoder = encode_targets_soft if soft else encode_targets
    images, targets = [], []
    for image, boxes in batch:
        images.append(image)
        targets.append(
            encoder(
                boxes,
                image_size=IMAGE_SIZE,
                grid_size=GRID,
                num_predictions=3
            )
        )
    return torch.stack(images), torch.stack(targets)


def save_aug_check(dataset, path, count=6):
    tiles = []
    for i in range(count):
        image, boxes = dataset[i]
        array = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(array)
        draw = ImageDraw.Draw(pil)
        for x1, y1, x2, y2 in boxes.tolist():
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        tiles.append(pil)

    sheet = Image.new("RGB", (IMAGE_SIZE * 3, IMAGE_SIZE * 2))
    for k, tile in enumerate(tiles):
        sheet.paste(tile, ((k % 3) * IMAGE_SIZE, (k // 3) * IMAGE_SIZE))
    sheet.save(path)
    print("Saved", path, "- red boxes must sit on the people.")


# ============================================================
# Validation metrics (CPU)
# ============================================================

_gy, _gx = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij")
GRID_X = _gx.reshape(1, 1, GRID, GRID).float()
GRID_Y = _gy.reshape(1, 1, GRID, GRID).float()


def to_xyxy(v):
    cx = (GRID_X + v[:, :, 0]) * CELL
    cy = (GRID_Y + v[:, :, 1]) * CELL
    w = v[:, :, 2] * IMAGE_SIZE
    h = v[:, :, 3] * IMAGE_SIZE
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def batch_metrics(pred, target):
    b = pred.shape[0]
    pred = pred.reshape(b, 3, 5, GRID, GRID)
    target = target.reshape(b, 3, 5, GRID, GRID)

    mask = target[:, :, 4] >= 0.999          # real positives only
    negative_mask = target[:, :, 4] <= 0.0   # soft neighbour cells are ignored here
    prob = torch.sigmoid(pred[:, :, 4])
    pos = prob[mask].numpy()
    neg = prob[negative_mask].numpy()

    px1, py1, px2, py2 = [a[mask] for a in to_xyxy(pred)]
    tx1, ty1, tx2, ty2 = [a[mask] for a in to_xyxy(target)]

    iw = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(min=0)
    ih = (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(min=0)
    inter = iw * ih
    union = (px2 - px1) * (py2 - py1) + (tx2 - tx1) * (ty2 - ty1) - inter
    iou = (inter / union.clamp(min=1e-6)).numpy()

    return pos, neg, iou


def auc_score(pos, neg):
    scores = np.concatenate([pos, neg])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


# ============================================================
# Main
# ============================================================

def main():

    import torch_directml

    global BEST_PATH, LAST_PATH, RESUME_PATH, LOG_PATH, EPOCHS, MAX_TRAIN_IMAGES

    check_aug = "--check-aug" in sys.argv
    smoke = "--smoke" in sys.argv
    resume = "--resume" in sys.argv
    use_negatives = "--negatives" in sys.argv
    use_soft = "--soft" in sys.argv
    use_v5 = "--v5" in sys.argv

    if use_v5 and use_soft:
        print("Use either --soft OR --v5, not both (they are two different fixes).")
        return

    tag = ""
    for arg in sys.argv[1:]:
        if arg.startswith("--epochs="):
            EPOCHS = int(arg.split("=", 1)[1])
        elif arg.startswith("--images="):
            MAX_TRAIN_IMAGES = int(arg.split("=", 1)[1])
        elif arg.startswith("--tag="):
            tag = "_" + arg.split("=", 1)[1]

    suffix = ("_neg" if use_negatives else "") + ("_soft" if use_soft else "") \
        + ("_v5" if use_v5 else "") + tag
    if suffix:
        BEST_PATH = BEST_PATH.replace(".pth", suffix + ".pth")
        LAST_PATH = LAST_PATH.replace(".pth", suffix + ".pth")
        RESUME_PATH = RESUME_PATH.replace(".pth", suffix + ".pth")
        LOG_PATH = LOG_PATH.replace(".csv", suffix + ".csv")
    print(f"Persons images: {MAX_TRAIN_IMAGES} | epochs: {EPOCHS} | "
          f"negatives: {use_negatives} | soft labels: {use_soft} | V5 multi-cell labels: {use_v5} | "
          f"best checkpoint -> {BEST_PATH}")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch_directml.device(0)
    print("Using device:", device)

    # ---------------- data ----------------
    print("Reading train2017 annotations (about a minute, a few GB of RAM)...")
    train_records = build_records(TRAIN_ANN, TRAIN_IMAGE_DIR, skip_empty=True)
    print("train2017 images with persons:", len(train_records))

    if MAX_TRAIN_IMAGES and len(train_records) > MAX_TRAIN_IMAGES:
        train_records = random.Random(SEED).sample(train_records, MAX_TRAIN_IMAGES)
    if smoke:
        train_records = train_records[:1200]

    if use_negatives:
        negative_pool = build_negative_records(TRAIN_ANN, TRAIN_IMAGE_DIR)
        wanted = int(len(train_records) * NEGATIVE_FRACTION / (1.0 - NEGATIVE_FRACTION))
        negative_records = random.Random(SEED + 1).sample(
            negative_pool, min(wanted, len(negative_pool))
        )
        print(f"Person-free train2017 images available: {len(negative_pool)} | "
              f"added to training: {len(negative_records)}")
        train_records = train_records + negative_records

    # the same 270 validation images as every earlier experiment
    val_all = build_records(VAL_ANN, VAL_IMAGE_DIR, skip_empty=False)
    perm = torch.randperm(
        len(val_all), generator=torch.Generator().manual_seed(42)
    ).tolist()
    val_start = int(len(val_all) * 0.9)
    val_records = [val_all[i] for i in perm[val_start:]]

    train_dataset = PersonRecords(train_records, augment_data=True)
    val_dataset = PersonRecords(val_records, augment_data=False)

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    if check_aug:
        save_aug_check(train_dataset, "aug_check_big.jpg")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=partial(collate_fn, soft=use_soft, v5=use_v5),
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
        prefetch_factor=4 if NUM_WORKERS > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_fn, soft=use_soft, v5=use_v5)
    )

    # ---------------- model ----------------
    if use_v5:
        from model_v5 import PersonDetectorV5      # imported lazily: only needed with --v5
        model = PersonDetectorV5(num_boxes=3).to(device)
    else:
        model = PersonDetectorV4(num_boxes=3).to(device)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)

    loss_class = DetectionLossV4 if use_soft else DetectionLossV3Localization
    criterion = loss_class(
        objectness_weight=1.0,
        bbox_weight=5.0,
        positive_objectness_weight=5.0
    )

    decay_params, no_decay_params = [], []
    for _, p in model.named_parameters():
        (no_decay_params if p.ndim <= 1 else decay_params).append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=PEAK_LR
    )

    all_params = list(model.parameters())
    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch

    def lr_at(step):
        if step < WARMUP_STEPS:
            return PEAK_LR * (step + 1) / WARMUP_STEPS
        progress = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * progress))

    @torch.no_grad()
    def update_ema(step):
        decay = min(EMA_DECAY, (1 + step) / (10 + step))
        for e, m in zip(ema.parameters(), model.parameters()):
            e.mul_(decay).add_(m.detach(), alpha=1.0 - decay)
        for (name, eb), (_, mb) in zip(ema.named_buffers(), model.named_buffers()):
            if name.endswith("num_batches_tracked"):
                continue
            eb.copy_(mb)

    def clip_gradients(max_norm):
        grads = [p.grad for p in all_params if p.grad is not None]
        total_sq = sum((g.detach() ** 2).sum() for g in grads)
        total_norm = float(total_sq.sqrt().item())
        if not math.isfinite(total_norm):
            return total_norm, False
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for g in grads:
                g.mul_(scale)
        return total_norm, True

    @torch.no_grad()
    def validate(net):
        net.eval()
        total = obj = bbox = 0.0
        pos_all, neg_all, iou_all = [], [], []

        for images, targets in val_loader:
            images = images.to(device)
            preds = net(images)
            t, o, bb = criterion(preds, targets.to(device))

            total += t.item()
            obj += o.item()
            bbox += bb.item()

            pos, neg, iou = batch_metrics(preds.detach().cpu(), targets)
            pos_all.append(pos)
            neg_all.append(neg)
            iou_all.append(iou)

        n = len(val_loader)
        pos_all = np.concatenate(pos_all)
        neg_all = np.concatenate(neg_all)
        iou_all = np.concatenate(iou_all)

        return {
            "loss": total / n,
            "obj": obj / n,
            "bbox": bbox / n,
            "auc": auc_score(pos_all, neg_all),
            "iou": float(iou_all.mean()),
            "iou50": float((iou_all >= 0.5).mean()),
        }

    # ---------------- resume ----------------
    start_epoch = 0
    step = 0
    best_val = float("inf")

    if resume and os.path.exists(RESUME_PATH):
        state = torch.load(RESUME_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        ema.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"]
        step = state["step"]
        best_val = state["best_val"]
        print(f"Resumed from epoch {start_epoch}, step {step}, best EMA val loss {best_val:.4f}")

    log_fields = [
        "epoch", "lr", "train_loss", "train_obj", "train_bbox", "grad_norm",
        "skipped", "val_loss_raw", "val_loss_ema", "val_obj_ema", "val_bbox_ema",
        "val_auc_ema", "val_iou_ema", "val_iou50_ema", "seconds"
    ]
    if not smoke and (start_epoch == 0 or not os.path.exists(LOG_PATH)):
        with open(LOG_PATH, "w", newline="") as f:
            csv.writer(f).writerow(log_fields)

    print(f"\nSteps per epoch: {steps_per_epoch} | total steps: {total_steps}")
    print(f"Workers: {NUM_WORKERS} | grad clip: {GRAD_CLIP}")

    # ---------------- training ----------------
    for epoch in range(start_epoch, EPOCHS):

        epoch_start = time.time()
        model.train()

        sum_total = sum_obj = sum_bbox = sum_gnorm = 0.0
        counted = 0
        skipped = 0
        lr = PEAK_LR
        timer_start = None

        for images, targets in train_loader:

            lr = lr_at(step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            preds = model(images)
            total, obj, bbox = criterion(preds, targets)

            loss_value = total.item()
            if not math.isfinite(loss_value):
                skipped += 1
                step += 1
                continue

            total.backward()

            gnorm, finite = clip_gradients(GRAD_CLIP)
            if not finite:
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                step += 1
                continue

            optimizer.step()
            update_ema(step)

            sum_total += loss_value
            sum_obj += obj.item()
            sum_bbox += bbox.item()
            sum_gnorm += gnorm
            counted += 1
            step += 1

            if smoke:
                if counted == 10:
                    timer_start = time.time()
                if counted >= SMOKE_STEPS:
                    break

        if smoke:
            per_step = (time.time() - timer_start) / (SMOKE_STEPS - 10)
            full_images = min(MAX_TRAIN_IMAGES, 64000)
            if use_negatives:
                full_images = int(full_images / (1.0 - NEGATIVE_FRACTION))
            full_steps = full_images // BATCH_SIZE
            print(f"\nSMOKE TEST: {SMOKE_STEPS} steps ran, skipped={skipped}")
            print(f"  seconds per step (with data loading): {per_step:.3f}")
            print(f"  estimated minutes per epoch  : {per_step * full_steps / 60:.1f}")
            print(f"  estimated hours for {EPOCHS} epochs: "
                  f"{per_step * full_steps * EPOCHS / 3600:.1f}")
            val = validate(ema)
            print(f"  validation path works: loss {val['loss']:.3f}, AUC {val['auc']:.3f}, IoU {val['iou']:.3f}")
            print("Smoke test finished OK. Nothing was saved.")
            return

        counted = max(counted, 1)
        train_total = sum_total / counted
        train_obj = sum_obj / counted
        train_bbox = sum_bbox / counted
        train_gnorm = sum_gnorm / counted

        val_raw = validate(model)
        val_ema = validate(ema)

        seconds = time.time() - epoch_start
        remaining = seconds * (EPOCHS - epoch - 1) / 60

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]  lr={lr:.2e}  "
              f"time={seconds / 60:.1f} min  (~{remaining:.0f} min left)")
        print(f"Train Loss: {train_total:.4f} | Objectness: {train_obj:.4f} | "
              f"BBox: {train_bbox:.4f} | grad norm: {train_gnorm:.2f} | skipped: {skipped}")
        print(f"Validation (raw weights): loss {val_raw['loss']:.4f}")
        print(f"Validation (EMA weights): loss {val_ema['loss']:.4f} | "
              f"objectness {val_ema['obj']:.4f} | bbox {val_ema['bbox']:.4f}")
        print(f"Validation (EMA) AUC {val_ema['auc']:.3f} | "
              f"box IoU {val_ema['iou']:.3f} | IoU>=0.5 {val_ema['iou50'] * 100:.1f}%")

        with open(LOG_PATH, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{lr:.3e}", f"{train_total:.4f}", f"{train_obj:.4f}",
                f"{train_bbox:.4f}", f"{train_gnorm:.3f}", skipped,
                f"{val_raw['loss']:.4f}", f"{val_ema['loss']:.4f}",
                f"{val_ema['obj']:.4f}", f"{val_ema['bbox']:.4f}",
                f"{val_ema['auc']:.4f}", f"{val_ema['iou']:.4f}",
                f"{val_ema['iou50']:.4f}", f"{seconds:.0f}"
            ])

        checkpoint = {
            "model_state_dict": to_cpu(ema.state_dict()),
            "epoch": epoch + 1,
            "validation_loss": val_ema["loss"],
            "val_auc": val_ema["auc"],
            "val_iou": val_ema["iou"],
        }
        torch.save(checkpoint, LAST_PATH)

        if val_ema["loss"] < best_val:
            best_val = val_ema["loss"]
            torch.save(checkpoint, BEST_PATH)
            print(f"Saved best model -> {BEST_PATH}")

        torch.save(
            {
                "model": to_cpu(model.state_dict()),
                "ema": to_cpu(ema.state_dict()),
                "optimizer": to_cpu(optimizer.state_dict()),
                "epoch": epoch + 1,
                "step": step,
                "best_val": best_val,
            },
            RESUME_PATH
        )

    print("\nTraining complete.")
    print("Best EMA validation loss:", best_val)
    print("Best model:", BEST_PATH, "| last epoch:", LAST_PATH, "| log:", LOG_PATH)


if __name__ == "__main__":
    main()


"""
test_v5.py  (READ-ONLY)

Run from : C:\\Users\\Nandhu\\Desktop\\amar\\project
Command  : ..\\venv\\Scripts\\python.exe .\\test_v5.py

Checks BEFORE training V5:
  1. label encoder: counts, decode-back to the original box, offsets, slot limits
  2. model_v5 output ranges (x,y in [-2.5,3.5], w,h in (0,1), objectness ~1%)
  3. CPU vs DirectML: outputs and gradients of EVERY parameter, using the real loss
  4. targets_v5_check.jpg : validation images with the positive cells drawn on them
"""
import copy

import numpy as np
import torch
import torch_directml
from PIL import Image, ImageDraw
from pycocotools.coco import COCO

from model_v5 import PersonDetectorV5
from targets_v5 import encode_targets_v5
from loss_v3_localization import DetectionLossV3Localization

CELL = 16.0
problems = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        problems.append(name)


def positives(t):
    out = []
    for slot in range(3):
        ys, xs = torch.nonzero(t[slot * 5 + 4] > 0, as_tuple=True)
        for y, x in zip(ys.tolist(), xs.tolist()):
            out.append((slot, y, x, t[slot * 5:slot * 5 + 5, y, x]))
    return out


def decode(p, gx, gy):
    cx = (gx + float(p[0])) * CELL
    cy = (gy + float(p[1])) * CELL
    w, h = float(p[2]) * 320, float(p[3]) * 320
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# ------------------------------------------------------------
# 1. encoder
# ------------------------------------------------------------
print("--- 1. label encoder ---")
t = encode_targets_v5(torch.zeros((0, 4)))
check("empty boxes -> zeros", tuple(t.shape) == (15, 20, 20) and float(t.sum()) == 0.0)

tiny = encode_targets_v5(torch.tensor([[100.0, 100.0, 112.0, 118.0]]))
check("tiny person -> 1 positive", len(positives(tiny)) == 1)

box = torch.tensor([[40.0, 60.0, 280.0, 300.0]])
large = encode_targets_v5(box)
check("large person -> 25 positives", len(positives(large)) == 25, f"({len(positives(large))})")

errs = [max(abs(a - b) for a, b in zip(decode(p, gx, gy), box[0].tolist()))
        for _, gy, gx, p in positives(large)]
check("every positive decodes back to the same box", max(errs) < 1e-3, f"(max error {max(errs):.5f} px)")

offs = torch.stack([p[:2] for _, _, _, p in positives(large)])
check("x,y offsets inside [-2, 3)", float(offs.min()) >= -2.0 and float(offs.max()) < 3.0,
      f"(min {float(offs.min()):.2f}, max {float(offs.max()):.2f})")

g = torch.Generator().manual_seed(0)
crowd = []
for _ in range(40):
    c = torch.rand(2, generator=g) * 240 + 40
    s = torch.rand(2, generator=g) * 180 + 20
    crowd.append([float(c[0] - s[0] / 2), float(c[1] - s[1] / 2), float(c[0] + s[0] / 2), float(c[1] + s[1] / 2)])
crowd = torch.tensor(crowd).clamp(0, 320)
tc = encode_targets_v5(crowd)
per_cell = (tc[4::5] > 0).sum(dim=0)
check("crowd of 40: no cell uses more than 3 slots", int(per_cell.max()) <= 3)
missing = 0
for b in crowd.tolist():
    gx = min(max(int((b[0] + b[2]) / 2 / CELL), 0), 19)
    gy = min(max(int((b[1] + b[3]) / 2 / CELL), 0), 19)
    if int(per_cell[gy, gx]) == 0:
        missing += 1
check("crowd of 40: every person's centre cell is positive", missing == 0)

# ------------------------------------------------------------
# 2. model ranges
# ------------------------------------------------------------
print("\n--- 2. model_v5 ---")
torch.manual_seed(0)
model = PersonDetectorV5(num_boxes=3)
model.eval()
probe = torch.rand(2, 3, 320, 320)
with torch.no_grad():
    out = model(probe)
check("output shape [2,15,20,20]", tuple(out.shape) == (2, 15, 20, 20))
xy = torch.cat([out[:, i * 5:i * 5 + 2] for i in range(3)], dim=1)
wh = torch.cat([out[:, i * 5 + 2:i * 5 + 4] for i in range(3)], dim=1)
obj = torch.cat([out[:, i * 5 + 4:i * 5 + 5] for i in range(3)], dim=1)
check("x,y inside [-2.5, 3.5]", float(xy.min()) >= -2.5 and float(xy.max()) <= 3.5,
      f"(range {float(xy.min()):.2f} .. {float(xy.max()):.2f})")
# Measured in TRAINING mode on a copy: in eval mode an untrained model uses the
# untouched batch-norm statistics, which shifts x,y and gave a false alarm.
train_copy = copy.deepcopy(model)
train_copy.train()
with torch.no_grad():
    out_t = train_copy(probe)
xy_t = torch.cat([out_t[:, i * 5:i * 5 + 2] for i in range(3)], dim=1)
check("x,y start near the cell middle (0.5)", abs(float(xy_t.mean()) - 0.5) < 0.15,
      f"(training mode, mean {float(xy_t.mean()):.3f}; eval mode {float(xy.mean()):.3f})")
check("w,h inside (0,1)", float(wh.min()) >= 0.0 and float(wh.max()) <= 1.0)
p_obj = float(torch.sigmoid(obj).mean())
check("initial objectness about 1%", 0.002 < p_obj < 0.05, f"({p_obj:.4f})")

# ------------------------------------------------------------
# 3. CPU vs DirectML
# ------------------------------------------------------------
print("\n--- 3. CPU vs DirectML (real loss, V5 targets) ---")
device = torch_directml.device(0)
torch.manual_seed(1)
reference = PersonDetectorV5(num_boxes=3)
images = torch.rand(4, 3, 320, 320)
boxes_per_image = [
    [[40, 60, 120, 260], [200, 100, 260, 230]],
    [[10, 10, 300, 310]],
    [[150, 40, 310, 300], [20, 200, 60, 300], [100, 100, 140, 180]],
    [[60, 90, 100, 170]],
]
targets = torch.stack([encode_targets_v5(torch.tensor(b, dtype=torch.float32)) for b in boxes_per_image])
criterion = DetectionLossV3Localization()


def run(dev):
    net = copy.deepcopy(reference).to(dev)
    net.train()
    o = net(images.to(dev))
    loss = criterion(o, targets.to(dev))[0]
    loss.backward()
    grads = {n: (p.grad.detach().cpu() if p.grad is not None else None) for n, p in net.named_parameters()}
    return o.detach().cpu(), float(loss.detach().cpu()), grads


out_c, loss_c, g_c = run("cpu")
out_d, loss_d, g_d = run(device)
diff = float((out_c - out_d).abs().max())
check("forward CPU vs DML", diff < 1e-2, f"(max diff {diff:.2e})")
check("loss CPU vs DML", abs(loss_c - loss_d) < 1e-2 * max(1.0, abs(loss_c)), f"({loss_c:.4f} vs {loss_d:.4f})")
check("every parameter has a gradient on DML", all(v is not None for v in g_d.values()))
worst = []
for n, gc in g_c.items():
    gd = g_d.get(n)
    if gc is None or gd is None:
        continue
    a, b = gc.flatten(), gd.flatten()
    if float(a.norm()) < 1e-12 and float(b.norm()) < 1e-12:
        continue
    worst.append((float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)), n))
worst.sort()
check("gradient direction CPU vs DML, every parameter", worst[0][0] > 0.98,
      f"(worst cosine {worst[0][0]:.4f} on '{worst[0][1]}')")

# ------------------------------------------------------------
# 4. picture of the positive cells
# ------------------------------------------------------------
print("\n--- 4. picture ---")
IMAGE_DIR = r"..\dataset\val2017"
ANN = r"..\dataset\annotations\instances_val2017.json"
coco = COCO(ANN)
all_ids = coco.getImgIds(catIds=[1])
perm = torch.randperm(len(all_ids), generator=torch.Generator().manual_seed(42)).tolist()
val_ids = [all_ids[i] for i in perm[int(len(all_ids) * 0.9):]]

tiles = []
for iid in val_ids:
    if len(tiles) == 6:
        break
    info = coco.loadImgs(iid)[0]
    W, H = info["width"], info["height"]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[iid], catIds=[1], iscrowd=False))
    boxes = []
    for a in anns:
        x, y, w, h = a["bbox"]
        boxes.append([x * 320 / W, y * 320 / H, (x + w) * 320 / W, (y + h) * 320 / H])
    if not boxes or max(max(b[2] - b[0], b[3] - b[1]) for b in boxes) < 120:
        continue
    img = Image.open(f"{IMAGE_DIR}\\{info['file_name']}").convert("RGB").resize((320, 320), Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(img)
    tgt = encode_targets_v5(torch.tensor(boxes, dtype=torch.float32))
    pos_cells = (tgt[4::5] > 0).any(dim=0)
    for gy_, gx_ in torch.nonzero(pos_cells).tolist():
        draw.rectangle([gx_ * 16, gy_ * 16, gx_ * 16 + 15, gy_ * 16 + 15], outline=(255, 60, 60), width=1)
    for b in boxes:
        draw.rectangle(b, outline=(0, 255, 0), width=2)
    tiles.append(img)

sheet = Image.new("RGB", (320 * 3, 320 * 2))
for k, tile in enumerate(tiles):
    sheet.paste(tile, ((k % 3) * 320, (k // 3) * 320))
sheet.save("targets_v5_check.jpg", quality=90)
print("Saved targets_v5_check.jpg  (green = person, red squares = positive cells)")

print()
if problems:
    print("SOME CHECKS FAILED:", problems)
    print("Do NOT train. Paste this whole output.")
else:
    print("ALL CHECKS PASSED - safe to train with:  train_v4_big.py --v5 ...")

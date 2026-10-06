"""
live_demo.py  (READ-ONLY for your project: it only reads the checkpoint)

Run from : C:\\Users\\Nandhu\\Desktop\\amar\\project
Commands :
  ..\\venv\\Scripts\\python.exe .\\live_demo.py                      webcam
  ..\\venv\\Scripts\\python.exe .\\live_demo.py --camera=1           another camera
  ..\\venv\\Scripts\\python.exe .\\live_demo.py --image=photo.jpg    one picture
  ..\\venv\\Scripts\\python.exe .\\live_demo.py --video=clip.mp4     a video file
  ..\\venv\\Scripts\\python.exe .\\live_demo.py --dml                run the model on DirectML

Other options:
  --model=FILE   checkpoint (default person_detector_v4_big_neg_v5_full.pth)
  --conf=0.70    confidence threshold at start
  --mirror       flip the picture left/right (natural for a camera facing you)
  --merge=0.60   merge rule: drop a box when this share of it (or of the other box)
                 lies inside a more confident box. --merge=0 starts with it off

Keys while the window is open:
  q or Esc : quit
  + / -    : raise / lower the confidence threshold by 0.05
  s        : save the current picture as live_demo_shot_N.jpg
  m        : merge rule on / off (to compare the person count while moving)

The picture is prepared exactly like in training (whole frame squeezed to
320x320, bilinear, RGB, 0..1) and decoded exactly like in evaluate_v5.py.
The model runs on the CPU unless you add --dml.
"""
import os
import sys
import time

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image

from model_v5 import PersonDetectorV5

IMAGE_SIZE = 320
GRID = 20
CELL = IMAGE_SIZE / GRID
NMS_THRESHOLD = 0.45


def option(name, default=None):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return a.split("=", 1)[1]
    return default


MODEL_PATH = option("model", "person_detector_v4_big_neg_v5_full.pth")
conf = float(option("conf", "0.70"))
image_path = option("image")
video_path = option("video")
camera_index = int(option("camera", "0"))
max_frames = int(option("max-frames", "0"))          # 0 = no limit (used for testing)
use_dml = "--dml" in sys.argv
mirror = "--mirror" in sys.argv
show_window = "--no-window" not in sys.argv
merge_value = float(option("merge", "0.60"))
merge_on = merge_value > 0
if merge_value <= 0:
    merge_value = 0.60          # value used if it is switched on later with m

if not os.path.exists(MODEL_PATH):
    print("Checkpoint not found:", MODEL_PATH)
    print("Checkpoints in this folder:")
    for name in sorted(os.listdir(".")):
        if name.endswith(".pth"):
            print("  ", name)
    sys.exit(1)

# ------------------------------------------------------------
# model
# ------------------------------------------------------------
if use_dml:
    import torch_directml
    device = torch_directml.device(0)
else:
    device = torch.device("cpu")

model = PersonDetectorV5(num_boxes=3)
ck = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
model.eval().to(device)

print("Checkpoint :", MODEL_PATH)
print("Device     :", "DirectML" if use_dml else "CPU")
print("Confidence :", conf)
print("Merge rule :", f"{merge_value:.2f}" if merge_on else "off")

gy, gx = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij")
gx = gx.float().unsqueeze(0)
gy = gy.float().unsqueeze(0)


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


def detect(frame_bgr, threshold, merge=0.0):
    """Returns a list of (x1, y1, x2, y2, score) in the frame's own pixels."""
    H, W = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    small = Image.fromarray(rgb).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        out = model(x.to(device))[0].cpu().reshape(3, 5, GRID, GRID)

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

    keep = (prob >= threshold) & (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, prob = boxes[keep], prob[keep]
    if len(boxes) == 0:
        return []
    order = torchvision.ops.nms(boxes, prob, NMS_THRESHOLD)
    sx, sy = W / IMAGE_SIZE, H / IMAGE_SIZE
    idx = order.tolist()
    if merge > 0:
        idx = [idx[k] for k in merge_contained([boxes[i].tolist() for i in idx], merge)]
    result = []
    for i in idx:
        x1, y1, x2, y2 = boxes[i].tolist()
        result.append((int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy), float(prob[i])))
    return result


def draw(frame, detections, threshold, fps=None, merge=0.0):
    for x1, y1, x2, y2, score in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"person {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        # above the box, or just inside it when the box touches the status bar at the top
        ty = y1 - 6 if y1 - th - 10 > 28 else max(y1, 28) + th + 8
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 4), (0, 220, 0), -1)
        cv2.putText(frame, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    status = f"persons: {len(detections)}   conf >= {threshold:.2f}"
    status += f"   merge {merge:.2f}" if merge > 0 else "   merge off"
    if fps is not None:
        status += f"   {fps:.1f} FPS"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(frame, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame


# ------------------------------------------------------------
# one picture
# ------------------------------------------------------------
if image_path:
    frame = cv2.imread(image_path)
    if frame is None:
        print("Could not read the image:", image_path)
        sys.exit(1)
    merge_now = merge_value if merge_on else 0.0
    dets = detect(frame, conf, merge_now)
    draw(frame, dets, conf, merge=merge_now)
    cv2.imwrite("live_demo_output.jpg", frame)
    print(f"Found {len(dets)} person(s). Saved live_demo_output.jpg")
    for d in dets:
        print(f"  box {d[:4]}  confidence {d[4]:.2f}")
    if show_window:
        cv2.imshow("Person detector (press any key to close)", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    sys.exit(0)

# ------------------------------------------------------------
# webcam or video file
# ------------------------------------------------------------
if video_path:
    cap = cv2.VideoCapture(video_path)
    source = video_path
elif os.name == "nt":
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)   # opens faster on Windows
    source = f"camera {camera_index}"
else:
    cap = cv2.VideoCapture(camera_index)
    source = f"camera {camera_index}"

if not cap.isOpened():
    print("Could not open", source)
    if not video_path:
        print("Close other apps that use the camera (Teams, Zoom, Camera app), check")
        print("Windows Settings > Privacy > Camera, or try another one: --camera=1")
    sys.exit(1)

print("Source     :", source)
print("Keys       : q or Esc = quit | + / - = threshold | s = save picture | m = merge on/off")

window = "Person detector (q = quit)"
fps = None
frames = 0
shots = 0
model_ms = []

while True:
    ok, frame = cap.read()
    if not ok:
        print("No more frames." if video_path else "The camera stopped sending frames.")
        break
    if mirror:
        frame = cv2.flip(frame, 1)

    t0 = time.perf_counter()
    merge_now = merge_value if merge_on else 0.0
    dets = detect(frame, conf, merge_now)
    dt = time.perf_counter() - t0
    model_ms.append(dt * 1000)
    fps = 1.0 / dt if fps is None else 0.9 * fps + 0.1 * (1.0 / dt)

    draw(frame, dets, conf, fps, merge_now)
    frames += 1

    if show_window:
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("+"), ord("=")):
            conf = min(0.95, conf + 0.05)
        if key in (ord("-"), ord("_")):
            conf = max(0.05, conf - 0.05)
        if key in (ord("m"), ord("M")):
            merge_on = not merge_on
            print("Merge rule:", f"{merge_value:.2f}" if merge_on else "off")
        if key == ord("s"):
            shots += 1
            cv2.imwrite(f"live_demo_shot_{shots}.jpg", frame)
            print(f"Saved live_demo_shot_{shots}.jpg")

    if max_frames and frames >= max_frames:
        break

cap.release()
if show_window:
    cv2.destroyAllWindows()

if model_ms:
    steady = model_ms[5:] if len(model_ms) > 10 else model_ms
    print(f"Frames: {frames} | detection time per frame: {np.mean(steady):.0f} ms "
          f"(about {1000 / np.mean(steady):.0f} FPS)")

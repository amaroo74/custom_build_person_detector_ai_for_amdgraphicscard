"""
targets_v5.py - label encoder with a CENTRAL REGION of positive cells

Problem found by size_diagnostic.py: for large persons the box at the centre
cell is good (IoU 0.77-0.81), but the objectness there is low (median 0.13-0.15),
because a large person covers dozens of nearly identical cells and only ONE of
them was labelled positive. The network cannot tell which one, so the best it
can do is to be unsure everywhere.

Fix: every cell whose centre lies in the CENTRAL part of the person's box is
positive, and each of them predicts the SAME full box (relative to its own cell).

  * region      : half-width of the positive region = region * box width (and height)
  * max_radius  : never more than this many cells on each side of the centre cell
                  (2 -> at most 5 x 5 = 25 positive cells for one person)
  * tiny / small persons still get exactly one positive cell
  * a person's own centre cell is always assigned first (pass 1); neighbours only
    use the slots that are still free (pass 2)

Target layout is unchanged:  [15, grid, grid] = 3 slots x (x, y, w, h, objectness)
For neighbour cells x and y are NOT inside 0..1 any more: they lie in
[-max_radius, 1 + max_radius). model_v5.py produces exactly that range.
"""
import numpy as np
import torch

IMAGE_SIZE = 320
GRID_SIZE = 20
NUM_PREDICTIONS = 3
REGION = 0.25
MAX_RADIUS = 2


def encode_targets_v5(
    boxes,
    image_size=IMAGE_SIZE,
    grid_size=GRID_SIZE,
    num_predictions=NUM_PREDICTIONS,
    region=REGION,
    max_radius=MAX_RADIUS,
):
    target = np.zeros((num_predictions * 5, grid_size, grid_size), dtype=np.float32)
    counts = np.zeros((grid_size, grid_size), dtype=np.int64)
    cell = image_size / grid_size

    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)

    infos = []
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = x2 - x1, y2 - y1
        gx0 = min(max(int(cx / cell), 0), grid_size - 1)
        gy0 = min(max(int(cy / cell), 0), grid_size - 1)
        infos.append((cx, cy, w, h, gx0, gy0))

    def write(info, gx, gy):
        cx, cy, w, h, _, _ = info
        slot = int(counts[gy, gx])
        if slot >= num_predictions:
            return False
        counts[gy, gx] += 1
        ch = slot * 5
        target[ch + 0, gy, gx] = cx / cell - gx
        target[ch + 1, gy, gx] = cy / cell - gy
        target[ch + 2, gy, gx] = w / image_size
        target[ch + 3, gy, gx] = h / image_size
        target[ch + 4, gy, gx] = 1.0
        return True

    # pass 1: every person's own centre cell
    for info in infos:
        write(info, info[4], info[5])

    # pass 2: neighbours inside the central region (only into free slots)
    for info in infos:
        cx, cy, w, h, gx0, gy0 = info
        rx = min(max_radius, int(region * w / cell))
        ry = min(max_radius, int(region * h / cell))
        for dy in range(-ry, ry + 1):
            for dx in range(-rx, rx + 1):
                if dx == 0 and dy == 0:
                    continue
                gx, gy = gx0 + dx, gy0 + dy
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    write(info, gx, gy)

    return torch.from_numpy(target)


if __name__ == "__main__":

    large = torch.tensor([[40.0, 40.0, 280.0, 300.0]])
    t = encode_targets_v5(large)
    print("target shape:", tuple(t.shape))
    print("positive slots for one large person:", int((t[4::5] > 0).sum()), "(expected 25)")
    tiny = torch.tensor([[100.0, 100.0, 112.0, 118.0]])
    print("positive slots for one tiny person :", int((encode_targets_v5(tiny)[4::5] > 0).sum()), "(expected 1)")

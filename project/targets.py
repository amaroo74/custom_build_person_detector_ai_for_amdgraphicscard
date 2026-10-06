import torch


IMAGE_SIZE = 320
GRID_SIZE = 20
NUM_PREDICTIONS = 3


def encode_targets(
    boxes,
    image_size=IMAGE_SIZE,
    grid_size=GRID_SIZE,
    num_predictions=NUM_PREDICTIONS
):
    """
    Convert bounding boxes into a V3 grid target.

    Target shape:
        [15, grid_size, grid_size]

    There are 3 predictions per grid cell.

    Each prediction contains:
        x
        y
        width
        height
        objectness

    Channel layout:
        prediction 0 -> channels 0-4
        prediction 1 -> channels 5-9
        prediction 2 -> channels 10-14
    """

    target = torch.zeros(
        num_predictions * 5,
        grid_size,
        grid_size,
        dtype=torch.float32
    )

    cell_size = image_size / grid_size

    # Number of objects already assigned to each cell.
    cell_counts = torch.zeros(
        grid_size,
        grid_size,
        dtype=torch.long
    )

    for box in boxes:

        x1, y1, x2, y2 = box

        # Bounding-box center
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        width = x2 - x1
        height = y2 - y1

        # Determine grid cell
        grid_x = int(center_x / cell_size)
        grid_y = int(center_y / cell_size)

        grid_x = min(
            max(grid_x, 0),
            grid_size - 1
        )

        grid_y = min(
            max(grid_y, 0),
            grid_size - 1
        )

        # Determine which prediction slot to use.
        slot = cell_counts[grid_y, grid_x].item()

        # Maximum of 3 objects per cell.
        if slot >= num_predictions:
            continue

        cell_counts[grid_y, grid_x] += 1

        # Position inside grid cell
        local_x = (
            center_x / cell_size
        ) - grid_x

        local_y = (
            center_y / cell_size
        ) - grid_y

        # Normalize width and height
        width_norm = width / image_size
        height_norm = height / image_size

        # Channel offset for this prediction
        channel = slot * 5

        # Store target
        target[channel + 0, grid_y, grid_x] = local_x
        target[channel + 1, grid_y, grid_x] = local_y
        target[channel + 2, grid_y, grid_x] = width_norm
        target[channel + 3, grid_y, grid_x] = height_norm
        target[channel + 4, grid_y, grid_x] = 1.0

    return target


# ============================================================
# V3 TARGET ENCODER TEST
# ============================================================

if __name__ == "__main__":

    # Three boxes whose centers all fall inside
    # the same 16x16 grid cell.
    boxes = torch.tensor(
        [
            [100.0, 100.0, 106.0, 106.0],
            [102.0, 102.0, 108.0, 108.0],
            [104.0, 104.0, 110.0, 110.0]
        ]
    )

    target = encode_targets(boxes)

    print(
        "Target shape:",
        target.shape
    )

    print(
        "Expected shape:",
        torch.Size([15, 20, 20])
    )

    print()

    for slot in range(NUM_PREDICTIONS):

        objectness_channel = slot * 5 + 4

        object_cells = torch.nonzero(
            target[objectness_channel] > 0
        )

        print(
            f"Prediction slot {slot}:"
        )

        for cell in object_cells:

            y = cell[0].item()
            x = cell[1].item()

            print(
                f"  Grid cell ({y}, {x}):",
                target[
                    slot * 5:(slot + 1) * 5,
                    y,
                    x
                ]
            )

        print()
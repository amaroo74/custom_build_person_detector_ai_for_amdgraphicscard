import torch
import torch.nn.functional as F


class DetectionLossV3Localization:
    """
    V3 loss focused on improving localization.

    Keeps the original V3 objectness behavior:
        - objectness weight = 1.0
        - positive objectness weight = 5.0

    Changes only the bounding-box loss:
        Old:
            MSE

        New:
            Smooth-L1 + GIoU

    Predictions:
        [B, 15, 20, 20]

    Interpreted as:
        [B, 3, 5, 20, 20]

    Each prediction:
        x
        y
        width
        height
        objectness
    """

    def __init__(
        self,
        objectness_weight=1.0,
        bbox_weight=5.0,
        positive_objectness_weight=5.0,
        smooth_l1_weight=1.0,
        giou_weight=1.0
    ):
        self.objectness_weight = objectness_weight
        self.bbox_weight = bbox_weight
        self.positive_objectness_weight = (
            positive_objectness_weight
        )
        self.smooth_l1_weight = smooth_l1_weight
        self.giou_weight = giou_weight

    def __call__(self, predictions, targets):
        return self.forward(
            predictions,
            targets
        )

    def forward(
        self,
        predictions,
        targets
    ):

        # ----------------------------------------------------
        # Shape checks
        # ----------------------------------------------------

        if predictions.ndim != 4:
            raise ValueError(
                f"Expected predictions [B,15,H,W], "
                f"got {predictions.shape}"
            )

        if targets.ndim != 4:
            raise ValueError(
                f"Expected targets [B,15,H,W], "
                f"got {targets.shape}"
            )

        batch_size = predictions.shape[0]
        grid_h = predictions.shape[2]
        grid_w = predictions.shape[3]

        if predictions.shape[1] != 15:
            raise ValueError(
                f"Expected 15 prediction channels, "
                f"got {predictions.shape[1]}"
            )

        if targets.shape[1] != 15:
            raise ValueError(
                f"Expected 15 target channels, "
                f"got {targets.shape[1]}"
            )

        # ----------------------------------------------------
        # [B,15,H,W] -> [B,3,5,H,W]
        # ----------------------------------------------------

        predictions = predictions.reshape(
            batch_size,
            3,
            5,
            grid_h,
            grid_w
        )

        targets = targets.reshape(
            batch_size,
            3,
            5,
            grid_h,
            grid_w
        )

        # ----------------------------------------------------
        # Model V3 already applies sigmoid to box values.
        # ----------------------------------------------------

        pred_x = predictions[:, :, 0]
        pred_y = predictions[:, :, 1]
        pred_w = predictions[:, :, 2]
        pred_h = predictions[:, :, 3]

        pred_objectness_logits = predictions[:, :, 4]

        # ----------------------------------------------------
        # Targets
        # ----------------------------------------------------

        target_x = targets[:, :, 0]
        target_y = targets[:, :, 1]
        target_w = targets[:, :, 2]
        target_h = targets[:, :, 3]

        target_objectness = targets[:, :, 4]

        positive_mask = (
            target_objectness > 0
        )

        # ----------------------------------------------------
        # Objectness loss
        #
        # Same behavior as original V3 loss.
        # ----------------------------------------------------

        positive_weight = torch.where(
            positive_mask,
            torch.tensor(
                self.positive_objectness_weight,
                device=predictions.device,
                dtype=predictions.dtype
            ),
            torch.tensor(
                1.0,
                device=predictions.device,
                dtype=predictions.dtype
            )
        )

        objectness_loss = (
            F.softplus(
                pred_objectness_logits
            )
            -
            (
                target_objectness
                * pred_objectness_logits
            )
        )

        objectness_loss = (
            objectness_loss
            * positive_weight
        ).mean()

        # ----------------------------------------------------
        # Localization loss
        # ----------------------------------------------------

        if positive_mask.any():

            # ------------------------------------------------
            # Smooth-L1 coordinate loss
            # ------------------------------------------------

            smooth_x = F.smooth_l1_loss(
                pred_x[positive_mask],
                target_x[positive_mask],
                reduction="mean"
            )

            smooth_y = F.smooth_l1_loss(
                pred_y[positive_mask],
                target_y[positive_mask],
                reduction="mean"
            )

            smooth_w = F.smooth_l1_loss(
                pred_w[positive_mask],
                target_w[positive_mask],
                reduction="mean"
            )

            smooth_h = F.smooth_l1_loss(
                pred_h[positive_mask],
                target_h[positive_mask],
                reduction="mean"
            )

            smooth_l1_loss = (
                smooth_x
                + smooth_y
                + smooth_w
                + smooth_h
            ) / 4.0

            # ------------------------------------------------
            # Grid coordinates
            # ------------------------------------------------

            device = predictions.device
            dtype = predictions.dtype

            grid_x = torch.arange(
                grid_w,
                device=device,
                dtype=dtype
            ).view(
                1,
                1,
                1,
                grid_w
            )

            grid_y = torch.arange(
                grid_h,
                device=device,
                dtype=dtype
            ).view(
                1,
                1,
                grid_h,
                1
            )

            # ------------------------------------------------
            # Convert local x/y to full-image normalized
            # center coordinates.
            # ------------------------------------------------

            pred_center_x = (
                grid_x + pred_x
            ) / float(grid_w)

            pred_center_y = (
                grid_y + pred_y
            ) / float(grid_h)

            target_center_x = (
                grid_x + target_x
            ) / float(grid_w)

            target_center_y = (
                grid_y + target_y
            ) / float(grid_h)

            # ------------------------------------------------
            # Predicted box
            # ------------------------------------------------

            pred_x1 = (
                pred_center_x
                - pred_w / 2.0
            )

            pred_y1 = (
                pred_center_y
                - pred_h / 2.0
            )

            pred_x2 = (
                pred_center_x
                + pred_w / 2.0
            )

            pred_y2 = (
                pred_center_y
                + pred_h / 2.0
            )

            # ------------------------------------------------
            # Target box
            # ------------------------------------------------

            target_x1 = (
                target_center_x
                - target_w / 2.0
            )

            target_y1 = (
                target_center_y
                - target_h / 2.0
            )

            target_x2 = (
                target_center_x
                + target_w / 2.0
            )

            target_y2 = (
                target_center_y
                + target_h / 2.0
            )

            # ------------------------------------------------
            # Intersection
            # ------------------------------------------------

            inter_x1 = torch.maximum(
                pred_x1,
                target_x1
            )

            inter_y1 = torch.maximum(
                pred_y1,
                target_y1
            )

            inter_x2 = torch.minimum(
                pred_x2,
                target_x2
            )

            inter_y2 = torch.minimum(
                pred_y2,
                target_y2
            )

            inter_w = torch.clamp(
                inter_x2 - inter_x1,
                min=0.0
            )

            inter_h = torch.clamp(
                inter_y2 - inter_y1,
                min=0.0
            )

            intersection = (
                inter_w * inter_h
            )

            # ------------------------------------------------
            # Areas
            # ------------------------------------------------

            pred_width = torch.clamp(
                pred_x2 - pred_x1,
                min=0.0
            )

            pred_height = torch.clamp(
                pred_y2 - pred_y1,
                min=0.0
            )

            target_width = torch.clamp(
                target_x2 - target_x1,
                min=0.0
            )

            target_height = torch.clamp(
                target_y2 - target_y1,
                min=0.0
            )

            pred_area = (
                pred_width
                * pred_height
            )

            target_area = (
                target_width
                * target_height
            )

            union = (
                pred_area
                + target_area
                - intersection
            )

            eps = torch.tensor(
                1e-7,
                device=device,
                dtype=dtype
            )

            iou = (
                intersection
                /
                torch.maximum(
                    union,
                    eps
                )
            )

            # ------------------------------------------------
            # Smallest enclosing box
            # ------------------------------------------------

            enclosing_x1 = torch.minimum(
                pred_x1,
                target_x1
            )

            enclosing_y1 = torch.minimum(
                pred_y1,
                target_y1
            )

            enclosing_x2 = torch.maximum(
                pred_x2,
                target_x2
            )

            enclosing_y2 = torch.maximum(
                pred_y2,
                target_y2
            )

            enclosing_width = torch.clamp(
                enclosing_x2 - enclosing_x1,
                min=0.0
            )

            enclosing_height = torch.clamp(
                enclosing_y2 - enclosing_y1,
                min=0.0
            )

            enclosing_area = (
                enclosing_width
                * enclosing_height
            )

            # ------------------------------------------------
            # GIoU
            # ------------------------------------------------

            giou = (
                iou
                -
                (
                    (
                        enclosing_area
                        - union
                    )
                    /
                    torch.maximum(
                        enclosing_area,
                        eps
                    )
                )
            )

            giou_loss = (
                1.0
                - giou
            )

            giou_loss = (
                giou_loss[
                    positive_mask
                ].mean()
            )

            # ------------------------------------------------
            # Combined localization loss
            # ------------------------------------------------

            bbox_loss = (
                self.smooth_l1_weight
                * smooth_l1_loss
                +
                self.giou_weight
                * giou_loss
            )

        else:

            bbox_loss = torch.zeros(
                (),
                device=predictions.device,
                dtype=predictions.dtype
            )

        # ----------------------------------------------------
        # Total
        # ----------------------------------------------------

        total_loss = (
            self.objectness_weight
            * objectness_loss
            +
            self.bbox_weight
            * bbox_loss
        )

        return (
            total_loss,
            objectness_loss,
            bbox_loss
        )


# ============================================================
# DirectML smoke test
# ============================================================

if __name__ == "__main__":

    import torch_directml

    device = torch_directml.device(0)

    print(
        "Using device:",
        device
    )

    # Fake V3 predictions
    predictions = torch.randn(
        2,
        15,
        20,
        20,
        device=device,
        requires_grad=True
    )

    # Fake V3 targets
    targets = torch.zeros(
        2,
        15,
        20,
        20,
        device=device
    )

    # --------------------------------------------------------
    # Three objects in the same cell
    # --------------------------------------------------------

    targets[
        0,
        4,
        6,
        6
    ] = 1.0

    targets[
        0,
        9,
        6,
        6
    ] = 1.0

    targets[
        0,
        14,
        6,
        6
    ] = 1.0

    # Box 1
    targets[
        0,
        0:4,
        6,
        6
    ] = torch.tensor(
        [0.5, 0.5, 0.2, 0.3],
        device=device
    )

    # Box 2
    targets[
        0,
        5:9,
        6,
        6
    ] = torch.tensor(
        [0.3, 0.4, 0.1, 0.2],
        device=device
    )

    # Box 3
    targets[
        0,
        10:14,
        6,
        6
    ] = torch.tensor(
        [0.7, 0.6, 0.15, 0.25],
        device=device
    )

    criterion = DetectionLossV3Localization()

    total_loss, objectness_loss, bbox_loss = criterion(
        predictions,
        targets
    )

    print(
        "Total loss:",
        total_loss.item()
    )

    print(
        "Objectness loss:",
        objectness_loss.item()
    )

    print(
        "Bounding-box loss:",
        bbox_loss.item()
    )

    print(
        "Loss finite:",
        torch.isfinite(
            total_loss
        ).item()
    )

    total_loss.backward()

    print(
        "Backward pass: successful"
    )

    print(
        "Gradient finite:",
        torch.isfinite(
            predictions.grad
        ).all().item()
    )
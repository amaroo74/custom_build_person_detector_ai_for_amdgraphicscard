"""
model_v5.py - V4 network + wider x/y output range

Identical layers and parameter names as model_v4.py (a V4 checkpoint would load),
but x and y are no longer squeezed into 0..1 of the OWN cell. A cell next to the
person's centre must be able to say "the centre is 1 or 2 cells to my left":

    x, y = sigmoid(raw) * (1 + 2R) - R      with R = 2.5   ->  range [-2.5, 3.5]
    (raw = 0 still means the middle of the own cell: 0.5)

targets_v5.py produces offsets in [-2, 3), so there is a half-cell margin on both
sides. The decoding used everywhere else is unchanged:  centre = (grid + x) * cell.

Same DirectML rules as V4: no in-place slice assignment, output built with
torch.cat, no tensor with more than 5 dimensions.
"""
import torch

from model_v4 import PersonDetectorV4, upsample2x

OFFSET_RADIUS = 2.5


class PersonDetectorV5(PersonDetectorV4):

    def forward(self, x):

        x = (x - 0.5) / 0.25

        x = self.stem(x)
        x = self.stage_a(x)
        x = self.stage_b(x)
        x = self.stage_c(x)

        fine = self.stage_d(x)

        coarse = self.coarse(fine)
        coarse = upsample2x(coarse)

        x = torch.cat([fine, coarse], dim=1)
        x = self.fuse(x)

        raw = self.detection_head(x)

        span = 1.0 + 2.0 * OFFSET_RADIUS

        parts = []
        for i in range(self.num_boxes):
            s = i * 5
            parts.append(torch.sigmoid(raw[:, s:s + 2]) * span - OFFSET_RADIUS)  # x, y
            parts.append(torch.sigmoid(raw[:, s + 2:s + 4]))                      # w, h
            parts.append(raw[:, s + 4:s + 5])                                     # objectness logit

        return torch.cat(parts, dim=1)


if __name__ == "__main__":

    model = PersonDetectorV5()
    out = model(torch.rand(2, 3, 320, 320))

    print("Output shape:", tuple(out.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("x,y range at start:", float(out[:, 0].min()), "to", float(out[:, 0].max()))

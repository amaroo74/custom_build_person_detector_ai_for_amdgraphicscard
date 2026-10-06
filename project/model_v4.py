"""
model_v4.py - V4 person detector (trained from scratch, ~1.4M parameters)

Same OUTPUT FORMAT as V3:  [B, 15, 20, 20]  =  3 slots x (x, y, w, h, objectness)
so targets.py, the loss and the evaluation scripts keep working unchanged.

What changed compared with V3 (0.4M parameters, 5 conv layers):
  * stride-2 stem, then 2 convs per stage with residual connections
  * a COARSE branch (10x10) that is upsampled and fused into the 20x20 map,
    so one cell can use context from a large part of the image
  * theoretical receptive field per output cell (computed by hand):
        fine path   ~165 px
        coarse path ~309 px   (V3: 78 px;  image: 320 px)
  * objectness bias starts at logit -4.6 (probability ~1%), so training does
    not begin by being flooded with ~300 background slots per person
  * fixed input normalisation inside the model: (x - 0.5) / 0.25

Stability rules learned the hard way on DirectML:
  * NO in-place slice assignment in forward (its backward was wrong on DML)
  * the output is assembled with torch.cat
  * the 2x upsampling uses expand + reshape (plain tensor ops) instead of
    interpolate, and never creates a tensor with more than 5 dimensions
    (DirectML limit); test_model_v4.py verifies that it equals
    nearest-neighbour and that CPU and DirectML gradients agree
"""
import torch
import torch.nn as nn


OBJECTNESS_PRIOR_LOGIT = -4.6   # sigmoid(-4.6) ~ 0.01


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """ReLU( x + BN(Conv(x)) )  - keeps the signal and the gradient flowing"""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x):
        return self.act(x + self.bn(self.conv(x)))


def upsample2x(x):
    """
    Nearest-neighbour 2x upsampling using only expand + reshape.

    DirectML supports at most 5 tensor dimensions, so this is done in two
    steps (width, then height); no tensor here has more than 5 dimensions.
    """
    b, c, h, w = x.shape

    # width: [b,c,h,w] -> [b,c,h,w,2] -> [b,c,h,2w]
    x = x.unsqueeze(4).expand(b, c, h, w, 2).reshape(b, c, h, w * 2)

    # height: [b,c,h,2w] -> [b,c,h,2,2w] -> [b,c,2h,2w]
    x = x.unsqueeze(3).expand(b, c, h, 2, w * 2).reshape(b, c, h * 2, w * 2)

    return x


class PersonDetectorV4(nn.Module):

    def __init__(self, num_boxes=3):
        super().__init__()

        self.num_boxes = num_boxes

        # 320 -> 160
        self.stem = ConvBlock(3, 16, stride=2)

        # 160 -> 80 -> 40 -> 20
        self.stage_a = nn.Sequential(
            ConvBlock(16, 32),
            nn.MaxPool2d(2)
        )
        self.stage_b = nn.Sequential(
            ConvBlock(32, 48),
            ResBlock(48),
            nn.MaxPool2d(2)
        )
        self.stage_c = nn.Sequential(
            ConvBlock(48, 96),
            ResBlock(96),
            nn.MaxPool2d(2)
        )

        # fine path at 20x20
        self.stage_d = nn.Sequential(
            ConvBlock(96, 160),
            ResBlock(160)
        )

        # coarse path at 10x10 (large receptive field)
        self.coarse = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(160, 192),
            ResBlock(192)
        )

        # fuse fine (160) + upsampled coarse (192) = 352 channels
        self.fuse = nn.Sequential(
            ConvBlock(352, 160, kernel_size=1),
            ResBlock(160)
        )

        self.detection_head = nn.Conv2d(
            160, num_boxes * 5, kernel_size=1
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.normal_(self.detection_head.weight, std=0.01)

        with torch.no_grad():
            self.detection_head.bias.zero_()
            for i in range(self.num_boxes):
                self.detection_head.bias[i * 5 + 4] = OBJECTNESS_PRIOR_LOGIT

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

        # Box values -> sigmoid; objectness stays a raw logit.
        # Built with torch.cat (NO in-place slice assignment).
        parts = []
        for i in range(self.num_boxes):
            s = i * 5
            parts.append(torch.sigmoid(raw[:, s:s + 4]))
            parts.append(raw[:, s + 4:s + 5])

        return torch.cat(parts, dim=1)


if __name__ == "__main__":

    model = PersonDetectorV4()
    out = model(torch.rand(2, 3, 320, 320))

    print("Output shape:", tuple(out.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))

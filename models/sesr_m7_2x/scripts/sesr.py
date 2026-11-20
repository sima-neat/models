"""SESR-M7 2x: the network, its interface, and the tile geometry every script shares.

The network is AIMET Model Zoo's SESRRelease (aimet_zoo_torch/common/super_resolution/models.py and blocks.py at
quic/aimet-model-zoo 59640d130992f984fe71339c27221aa6e3434aef), trimmed to inference. Upstream notice:

    Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
    Distributed under the BSD 3-Clause License (quic/aimet-model-zoo LICENSE.md).

SESR is by Bhardwaj et al., "Collapsible Linear Blocks for Super-Efficient Super Resolution" (MLSys 2022,
arXiv 2103.09404), first released in ARM-software/sesr under Apache-2.0.
"""

from __future__ import annotations

NUM_BLOCKS = 7
SCALE = 2
NUM_CHANNELS = 16
EXPAND_CHANNELS = 256
# The model takes one 224x224 RGB tile in [0, 1] and returns the 448x448 tile, also in [0, 1] before clipping.
TILE = 224
INPUT = "lr_tile"
OUTPUT = "sr_tile"
INPUT_SHAPE = [1, 3, TILE, TILE]
OUTPUT_SHAPE = [1, 3, TILE * SCALE, TILE * SCALE]


def build_network():
    """SESRRelease with NUM_BLOCKS residual collapsible linear blocks, uncollapsed as AIMET trains and scores it."""
    import torch
    from torch import nn

    class AnchorOp(nn.Module):
        """Repeats each input channel SCALE**2 times: nearest upsampling once the pixel shuffle runs."""

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Conv2d(3, 3 * SCALE**2, kernel_size=1)

        def forward(self, x):
            return self.net(x)

    class CollapsibleLinearBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, kernel_size: int, activation: str) -> None:
            super().__init__()
            self.conv_expand = nn.Conv2d(
                in_channels, EXPAND_CHANNELS, kernel_size, padding=(kernel_size - 1) // 2, bias=False
            )
            self.conv_squeeze = nn.Conv2d(EXPAND_CHANNELS, out_channels, 1)
            self.activation = nn.ReLU() if activation == "relu" else nn.Identity()

        def forward(self, x):
            return self.activation(self.conv_squeeze(self.conv_expand(x)))

    class ResidualCollapsibleLinearBlock(CollapsibleLinearBlock):
        def forward(self, x):
            return self.activation(x + self.conv_squeeze(self.conv_expand(x)))

    class SESR(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = AnchorOp()
            self.conv_first = CollapsibleLinearBlock(3, NUM_CHANNELS, 5, "relu")
            self.residual_block = nn.Sequential(
                *[ResidualCollapsibleLinearBlock(NUM_CHANNELS, NUM_CHANNELS, 3, "relu") for _ in range(NUM_BLOCKS)]
            )
            self.conv_last = CollapsibleLinearBlock(NUM_CHANNELS, 3 * SCALE**2, 5, "identity")
            self.depth_to_space = nn.PixelShuffle(SCALE)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            upsampled = self.anchor(x)
            features = self.conv_first(x)
            residual = self.residual_block(features) + features
            return self.depth_to_space(self.conv_last(residual) + upsampled)

    return SESR()

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch as t
from typing import Tuple, Generator


class MnistSegmentationUnet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 11):
        super().__init__()
        self.num_classes = num_classes

        # Mnist images can only be halved 4 times ((112 -> 56), (56 -> 28), (28 -> 14), (14 -> 7))
        down_seq = [in_channels, 64, 128, 256]
        up_seq = [*down_seq[::-1][:-1], 64]

        self.down = nn.ModuleList()
        for i, (in_c, out_c) in twos(down_seq):
            self.down.add_module(f"{i + 1}", DoubleConv(in_c, out_c))

        self.up = nn.ModuleList()
        for i, (in_c, out_c) in twos(up_seq):
            skip_c = in_c
            self.up.add_module(f"{i + 1}", UpSampleWithSkip(in_c, skip_c, out_c))

        self.final_conv = nn.Conv2d(up_seq[-1], self.num_classes, kernel_size=1)

    def forward(self, y: t.Tensor):
        skips: list[t.Tensor] = []

        for idx, down in enumerate(self.down):
            y = down(y)
            skips.append(y)
            y = F.max_pool2d(y, 2, 2)

        skips = skips[::-1]
        for idx, up in enumerate(self.up):
            y = up(y, skips[idx])

        y = self.final_conv(y)

        y = F.softmax(y, dim=1)

        return y


def twos(nums: list[int]) -> Generator[Tuple[int, Tuple[int, int]], None, None]:
    for i in range(len(nums) - 1):
        yield i, (nums[i], nums[i + 1])


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.seq = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                    ),
                    ("norm1", nn.BatchNorm2d(out_channels)),
                    ("relu1", nn.ReLU(inplace=True)),
                    (
                        "conv2",
                        nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                    ),
                    ("relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    def forward(self, x):
        return self.seq(x)


class UpSampleWithSkip(nn.Module):
    def __init__(
        self,
        up_in_channels: int,
        skip_in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.up_in_channels = up_in_channels
        self.skip_in_channels = skip_in_channels
        self.out_channels = out_channels

        self.up = nn.ConvTranspose2d(
            up_in_channels, up_in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(
            self.up.out_channels + skip_in_channels,
            out_channels,
        )

    def forward(self, down_img: t.Tensor, skip_img: t.Tensor):
        y = self.up(down_img)
        y = t.cat([y, skip_img], dim=1)
        y = self.conv(y)
        return y

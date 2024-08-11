import torchvision as tv  # type: ignore[import]
import torch as t
from torch.utils.data.dataset import Dataset
from typing import Tuple


class MnistGridDataset(Dataset):
    def __init__(self, grid_wh: int = 4, num_classes: int = 11):
        self.grid_wh = grid_wh
        self.num_classes = num_classes
        self.mnist_dataset = tv.datasets.MNIST(
            root="../data",
            download=True,
            transform=tv.transforms.ToTensor(),
            train=True,
        )

    def __len__(self):
        return len(self.mnist_dataset) // (self.grid_wh**2)

    def __getitem__(self, idx) -> Tuple[t.Tensor, t.Tensor]:
        idx_start = idx * (self.grid_wh**2)
        original_grid = t.tensor([])
        colored_grid = t.tensor([])
        for i in range(0, self.grid_wh):
            original_row = t.tensor([])
            colored_row = t.tensor([])
            for j in range(0, self.grid_wh):
                digit_grayscale, digit = self.mnist_dataset[
                    idx_start + (i * self.grid_wh + j)
                ]

                # (1C, W, H) grayscale img
                digit_grayscale = t.as_tensor(digit_grayscale)

                # (W, H) grayscale img
                digit_mask = digit_grayscale.squeeze()

                # (11C, W, H) one-hot channeled img
                one_hot_channeled_img = t.zeros_like(digit_grayscale).repeat(
                    self.num_classes, 1, 1
                )
                # add black background mask: all ones except where the digit is
                one_hot_channeled_img[10] = 1 - digit_mask

                # add digit mask to the correct one-hot channel
                one_hot_channeled_img[digit] += digit_mask

                original_row = t.cat([original_row, digit_grayscale], 2)
                colored_row = t.cat([colored_row, one_hot_channeled_img], 2)

            original_grid = t.cat([original_grid, original_row], 1)
            colored_grid = t.cat([colored_grid, colored_row], 1)

        return (original_grid, colored_grid)

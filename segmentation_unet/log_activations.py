from contextlib import contextmanager
from torch.utils.hooks import RemovableHandle
import torch as t
import torchvision as tv  # type: ignore
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Generator
import mlflow  # type: ignore
from matplotlib import colors
import PIL


class ActivationImgLogger:
    def __init__(self, model: nn.Module, step: int, true_seg_img: t.Tensor):
        self.model = model
        self.step = step
        self.true_seg_img = true_seg_img
        self.act_num = 0

    @contextmanager
    def wrap_forward_pass(self, do_log: bool) -> Generator[None, None, None]:
        unhook_fns: list[RemovableHandle] = []

        # Log activations halfway through each epoch
        if do_log:
            for layer_name in [
                "down.1",
                "down.2",
                # "down.3",
                # "bottleneck",
                "up.1",
                "up.2",
                # "up.3",
                "final_conv",
            ]:
                unhook_fns.append(
                    self.model.get_submodule(layer_name).register_forward_hook(
                        self.act_img_logger(layer_name)
                    )
                )

            unhook_fns.append(self.model.register_forward_hook(self.log_final_output))

        try:
            yield
        finally:
            for unhook_fn in unhook_fns:
                unhook_fn.remove()

    def act_img_logger(self, layer_name: str):
        def log_act_img(module, input, output: t.Tensor):
            # (B, C, W, H) -> (1, C, W, H)
            # Take only the first image of the batch
            output = output[:1]

            # (1, C, W, H) -> (C, 1, W, H)
            # One gray scale image per channel
            output = output.transpose(0, 1)

            grid = tv.utils.make_grid(output)
            img = tv.transforms.ToPILImage()(grid)

            self.act_num += 1
            mlflow.log_image(
                img, f"step_{self.step:05}_act_{self.act_num:02}_{layer_name}.png"
            )

        return log_act_img

    def log_final_output(self, model, input, prediction: t.Tensor):
        fig = plt.figure(figsize=(15, 7))
        fig.subplots(2, 2)
        ax1, ax2, ax3, ax4 = fig.axes

        input = input[0][:1]
        grid = tv.utils.make_grid(input)
        img = tv.transforms.ToPILImage()(grid)
        ax1.title.set_text("Input Image")
        ax1.imshow(img)

        # (B, C, W, H)
        true_seg_img = self.true_seg_img[:1]
        # (B, RGB, W, H)
        rgb_img = self.seg_mask_visualization_rgb(true_seg_img)
        img = tv.transforms.ToPILImage()(rgb_img[0])
        ax2.title.set_text("True Segmentation")
        ax2.imshow(img)

        # (B, C, W, H)
        prediction = prediction[:1]
        # (B, RGB, W, H)
        rgb_img = self.seg_mask_visualization_rgb(prediction)
        img = tv.transforms.ToPILImage()(rgb_img[0])
        ax3.title.set_text("Predicted Segmentation")
        ax3.imshow(img)

        grid = tv.utils.make_grid(prediction.transpose(0, 1))
        img = tv.transforms.ToPILImage()(grid)
        resize = int(500 / img.height)
        img = img.resize((resize * img.width, resize * img.height))
        ax4.title.set_text("Predicted one-hot class channels")
        ax4.imshow(img)

        mlflow.log_figure(fig, f"step_{self.step:05}_output.png")
        plt.close(fig)

    def seg_mask_visualization_rgb(self, one_hot: t.Tensor) -> t.Tensor:
        # (B, C, W, H) * (C, RGB) -> (B, RGB, W, H)
        # Convert one-hot-channeled image to a visually segmented image
        rgb_img = t.einsum("bcwh,cr->brwh", one_hot, to_rgb)

        return rgb_img

to_rgb = t.tensor(
    [
        colors.to_rgb("orange"),
        colors.to_rgb("red"),
        colors.to_rgb("green"),
        colors.to_rgb("blue"),
        colors.to_rgb("yellow"),
        colors.to_rgb("magenta"),
        colors.to_rgb("purple"),
        colors.to_rgb("gray"),
        colors.to_rgb("white"),
        colors.to_rgb("brown"),
        # Number 11 will mean unclassified
        colors.to_rgb("black"),
    ]
)

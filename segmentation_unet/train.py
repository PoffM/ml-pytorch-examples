import os
import torch.nn.functional as F
import torch as t
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
import mlflow  # type: ignore
from tqdm.auto import tqdm, trange  # type: ignore
import torchinfo
from collections import OrderedDict

device = t.device("cuda" if t.cuda.is_available() else "cpu")
t.set_default_device(device)

from segmentation_unet.mnist_grid_dataset import MnistGridDataset
from segmentation_unet.model import MnistSegmentationUnet
from segmentation_unet.log_activations import ActivationImgLogger


tracking_uri = os.path.join(os.path.dirname(__file__), "mlruns/")
mlflow.set_tracking_uri(f"file:{tracking_uri}")

experiment = mlflow.set_experiment("digit-grid-segmentation")
client = mlflow.tracking.MlflowClient()
run = client.create_run(experiment.experiment_id)

batch_size = 32

full_dataset = MnistGridDataset(grid_wh=4)
train_len = len(full_dataset)
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [0.7, 0.15, 0.15],
    generator=t.Generator(device="cuda"),
)

train_loader = DataLoader(
    generator=t.Generator(device="cuda"),
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    generator=t.Generator(device="cuda"),
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=0,
)

test_loader = DataLoader(
    generator=t.Generator(device="cuda"),
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=0,
)

checkpoint_path = os.path.join(os.path.dirname(__file__), ".model.py")

def train():
    with mlflow.start_run(run_id=run.info.run_id):
        # Setup

        model = MnistSegmentationUnet().train()
        torchinfo.summary(model, input_size=(1, 1, 28 * 4, 28 * 4), depth=1)
        mlflow.log_artifact(os.path.join(os.path.dirname(__file__), "model.py"))

        mlflow.log_param("batch_size", batch_size)

        epochs = 3
        mlflow.log_param("epochs", epochs)

        def loss_fn(
            prefix: str, step: int, prediction_img: t.Tensor, true_seg_img: t.Tensor
        ):
            pred_digit_mask = prediction_img[:,:10]
            true_digit_mask = true_seg_img[:,:10]

            loss = F.binary_cross_entropy(pred_digit_mask, true_digit_mask)

            return loss

        optimizer = t.optim.AdamW(model.parameters())
        mlflow.log_param("optimizer", optimizer.__class__.__name__)


        def loader_loop(loader: DataLoader, desc: str):
            pbar = tqdm(loader, desc=desc, leave=False)
            for step, (orig_img, true_seg_img) in enumerate(pbar):
                free_mem, total_mem = t.cuda.mem_get_info()
                # get gpu usage

                used_mem_mb = (total_mem - free_mem) / 1024**2
                total_mem_mb = total_mem / 1024**2
                gpu_util = t.cuda.utilization()

                live_stats = OrderedDict()
                live_stats["GPU util"] = f"{gpu_util:.2f}"
                live_stats["GPU Memory util"] = f"{used_mem_mb:.2f}MB/{total_mem_mb:.2f}MB"
                pbar.set_postfix(live_stats)

                yield step, (orig_img, true_seg_img)

        epoch_pbar = trange(epochs, desc="Epochs")
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch}")
            # Training loop
            for step, (orig_img, true_seg_img) in loader_loop(
                train_loader, "Training Steps"
            ):
                step = epoch * len(train_loader) + step

                activation_logger = ActivationImgLogger(
                    model=model, step=step, true_seg_img=true_seg_img
                )
                with activation_logger.wrap_forward_pass(do_log=step % 20 == 0):
                    predicted_seg_img: t.Tensor = model(orig_img)

                loss = loss_fn("training", step, predicted_seg_img, true_seg_img)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_pbar.set_postfix(training_loss=f"{loss:.4f}")
                mlflow.log_metric("training_loss", loss, step=step)

            # Validation loop
            for step, (orig_img, true_seg_img) in loader_loop(
                val_loader, "Validation Steps"
            ):
                step = epoch * (len(val_loader) + len(train_loader)) + step

                with t.no_grad():
                    predicted_seg_img = model(orig_img)
                    loss = loss_fn("validation", step, predicted_seg_img, true_seg_img)

                    epoch_pbar.set_postfix(validation_loss=f"{loss:.4f}")
                    mlflow.log_metric("validation_loss", loss, step=step)

        # save checkpoint
        t.save(model.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train()


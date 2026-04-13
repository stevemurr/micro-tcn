import glob
import os

import torch
import typer

from microtcn.lstm import LSTMModel
from microtcn.tcn import TCNModel


def _load_model(model_dir: str, model_id: str, gpu: bool = False):
    checkpoint_path = glob.glob(
        os.path.join(model_dir, "lightning_logs", "version_0", "checkpoints", "*")
    )[0]
    model_type = os.path.basename(model_id).split("-")[1]
    map_location = "cuda:0" if gpu else "cpu"
    if model_type == "LSTM":
        return LSTMModel.load_from_checkpoint(
            checkpoint_path=checkpoint_path, map_location=map_location
        )
    return TCNModel.load_from_checkpoint(
        checkpoint_path=checkpoint_path, map_location=map_location
    )


def export(
    model_dir: str = typer.Option("./lightning_logs/bulk"),
    save_dir: str = typer.Option("./models"),
):
    """TorchScript-export every trained model under ``model_dir``."""
    models = sorted(glob.glob(os.path.join(model_dir, "*")))
    os.makedirs(save_dir, exist_ok=True)

    for md in models:
        model_id = os.path.basename(md)
        print(model_id)
        model = _load_model(md, model_id)
        script = model.to_torchscript()
        torch.jit.save(script, os.path.join(save_dir, f"traced_{model_id}.pt"))

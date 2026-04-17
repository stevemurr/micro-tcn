import os
from typing import Optional

import pytorch_lightning as pl
import torch
import typer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from microtcn.data import SignalTrainLA2ADataset
from microtcn.lstm import LSTMModel
from microtcn.tcn import TCNModel

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


TRAIN_CONFIGS = [
    {"name": "uTCN-300", "model_type": "tcn", "nblocks": 4, "dilation_growth": 10,
     "kernel_size": 13, "causal": True, "train_fraction": 0.01, "batch_size": 32},
    {"name": "uTCN-100", "model_type": "tcn", "nblocks": 4, "dilation_growth": 10,
     "kernel_size": 5, "causal": True, "train_fraction": 1.00, "batch_size": 32},
    {"name": "uTCN-300", "model_type": "tcn", "nblocks": 4, "dilation_growth": 10,
     "kernel_size": 13, "causal": True, "train_fraction": 1.00, "batch_size": 32},
    {"name": "uTCN-1000", "model_type": "tcn", "nblocks": 5, "dilation_growth": 10,
     "kernel_size": 5, "causal": True, "train_fraction": 1.00, "batch_size": 32},
    {"name": "uTCN-100", "model_type": "tcn", "nblocks": 4, "dilation_growth": 10,
     "kernel_size": 5, "causal": False, "train_fraction": 1.00, "batch_size": 32},
    {"name": "uTCN-300", "model_type": "tcn", "nblocks": 4, "dilation_growth": 10,
     "kernel_size": 13, "causal": False, "train_fraction": 1.00, "batch_size": 32},
    {"name": "uTCN-1000", "model_type": "tcn", "nblocks": 5, "dilation_growth": 10,
     "kernel_size": 5, "causal": False, "train_fraction": 1.00, "batch_size": 32},
    {"name": "TCN-300", "model_type": "tcn", "nblocks": 10, "dilation_growth": 2,
     "kernel_size": 15, "causal": False, "train_fraction": 1.00, "batch_size": 32},
    {"name": "uTCN-300", "model_type": "tcn", "nblocks": 4, "dilation_growth": 10,
     "kernel_size": 13, "causal": True, "train_fraction": 0.10, "batch_size": 32},
    {"name": "LSTM-32", "model_type": "lstm", "num_layers": 1, "hidden_size": 32,
     "train_fraction": 1.00, "batch_size": 32},
    {"name": "uTCN-300", "model_type": "tcn", "nblocks": 3, "dilation_growth": 60,
     "kernel_size": 5, "causal": True, "train_fraction": 1.0, "batch_size": 32},
    {"name": "uTCN-300", "model_type": "tcn", "nblocks": 4, "dilation_growth": 10,
     "kernel_size": 13, "causal": True, "train_fraction": 1.0, "batch_size": 32,
     "max_epochs": 60, "train_loss": "l1"},
    {"name": "uTCN-300", "model_type": "tcn", "nblocks": 30, "dilation_growth": 2,
     "kernel_size": 15, "causal": False, "train_fraction": 1.0, "batch_size": 32,
     "max_epochs": 60},
    {"name": "uTCN-324-16", "model_type": "tcn", "nblocks": 10, "dilation_growth": 2,
     "kernel_size": 15, "causal": False, "train_fraction": 1.0, "batch_size": 32,
     "max_epochs": 60, "channel_width": 16},
]


def _run_training(
    config: dict,
    idx: int,
    total: int,
    root_dir: str,
    num_workers: int,
    train_length: int,
    eval_length: int,
    gpus: int,
    lr: float,
    precision: str,
    artifact_dir: str,
    compile_model: bool,
):
    print(f"* Training config {idx + 1}/{total}")
    print(config)

    pl.seed_everything(42)

    max_epochs = config.get("max_epochs", 60)
    if precision == "16-mixed":
        data_dtype = torch.float16
    elif precision == "bf16-mixed":
        data_dtype = torch.bfloat16
    else:
        data_dtype = torch.float32

    model_type = config["model_type"]
    if model_type == "tcn":
        specifier = f"{idx + 1}-{config['name']}"
        specifier += "__causal" if config["causal"] else "__noncausal"
        specifier += f"__{config['nblocks']}-{config['dilation_growth']}-{config['kernel_size']}"
        specifier += f"__fraction-{config['train_fraction']}-bs{config['batch_size']}"
    elif model_type == "lstm":
        specifier = f"{idx + 1}-{config['name']}"
        specifier += f"__{config['num_layers']}-{config['hidden_size']}"
        specifier += f"__fraction-{config['train_fraction']}-bs{config['batch_size']}"
    else:
        raise typer.BadParameter(f"unknown model_type {model_type!r}")

    if "train_loss" in config:
        specifier += f"__loss-{config['train_loss']}"

    default_root_dir = os.path.join(artifact_dir, specifier)
    print(default_root_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="epoch={epoch:02d}-val={val_loss:.4f}",
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        default_root_dir=default_root_dir,
        max_epochs=max_epochs,
        precision=precision,
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else 1,
        callbacks=[checkpoint_callback],
        logger=TensorBoardLogger(save_dir=default_root_dir, name="lightning_logs"),
    )

    train_dataset = SignalTrainLA2ADataset(
        root_dir,
        subset="train",
        fraction=config["train_fraction"],
        dtype=data_dtype,
        length=train_length,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config["batch_size"],
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    val_dataset = SignalTrainLA2ADataset(
        root_dir,
        dtype=data_dtype,
        subset="val",
        length=eval_length,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=8,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    model_kwargs = {"nparams": 2, "lr": lr, "sample_rate": 44100}
    if "train_loss" in config:
        model_kwargs["train_loss"] = config["train_loss"]

    if model_type == "tcn":
        model_kwargs.update(
            nblocks=config["nblocks"],
            dilation_growth=config["dilation_growth"],
            kernel_size=config["kernel_size"],
            causal=config["causal"],
        )
        if "channel_width" in config:
            model_kwargs["channel_width"] = config["channel_width"]
        model = TCNModel(**model_kwargs)
    else:
        model_kwargs.update(
            num_layers=config["num_layers"],
            hidden_size=config["hidden_size"],
        )
        model = LSTMModel(**model_kwargs)

    print(model)
    if compile_model and gpus > 0:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"(skipping torch.compile: {e})")
    trainer.fit(model, train_dataloader, val_dataloader)


def train(
    model_type: str = typer.Option("tcn", help="'tcn' or 'lstm'."),
    name: str = typer.Option("uTCN-300", help="Label used in the log directory name."),
    root_dir: str = typer.Option("./data", help="Dataset root directory."),
    nblocks: int = typer.Option(4, help="TCN: number of blocks."),
    dilation_growth: int = typer.Option(10, help="TCN: dilation growth factor."),
    kernel_size: int = typer.Option(13, help="TCN: kernel size."),
    channel_width: Optional[int] = typer.Option(None, help="TCN: channel width override."),
    causal: bool = typer.Option(True, help="TCN: causal convolutions."),
    num_layers: int = typer.Option(1, help="LSTM: number of layers."),
    hidden_size: int = typer.Option(32, help="LSTM: hidden size."),
    train_fraction: float = typer.Option(1.0, help="Fraction of training data to use."),
    batch_size: int = typer.Option(32),
    max_epochs: int = typer.Option(60),
    lr: float = typer.Option(1e-3, help="Adam learning rate."),
    train_loss: Optional[str] = typer.Option(None, help="Override training loss (e.g. 'l1')."),
    train_length: int = typer.Option(65536),
    eval_length: int = typer.Option(131072),
    num_workers: int = typer.Option(6),
    gpus: int = typer.Option(1, help="Number of GPUs (0 for CPU)."),
    precision: str = typer.Option("bf16-mixed", help="Trainer precision: 'bf16-mixed', '16-mixed', or '32-true'."),
    artifact_dir: str = typer.Option("./lightning_logs/bulk", help="Root directory for checkpoints and TensorBoard logs."),
    compile_model: bool = typer.Option(True, "--compile/--no-compile", help="Wrap the model with torch.compile(mode='reduce-overhead')."),
):
    """Train a single TCN or LSTM model."""
    config: dict = {
        "name": name,
        "model_type": model_type,
        "train_fraction": train_fraction,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
    }
    if model_type == "tcn":
        config.update(
            nblocks=nblocks,
            dilation_growth=dilation_growth,
            kernel_size=kernel_size,
            causal=causal,
        )
        if channel_width is not None:
            config["channel_width"] = channel_width
    elif model_type == "lstm":
        config.update(num_layers=num_layers, hidden_size=hidden_size)
    else:
        raise typer.BadParameter(f"model_type must be 'tcn' or 'lstm', got {model_type!r}")
    if train_loss is not None:
        config["train_loss"] = train_loss

    _run_training(
        config,
        idx=0,
        total=1,
        root_dir=root_dir,
        num_workers=num_workers,
        train_length=train_length,
        eval_length=eval_length,
        gpus=gpus,
        lr=lr,
        precision=precision,
        artifact_dir=artifact_dir,
        compile_model=compile_model,
    )


def train_all(
    root_dir: str = typer.Option("./data", help="Dataset root directory."),
    num_workers: int = typer.Option(6),
    train_length: int = typer.Option(65536),
    eval_length: int = typer.Option(131072),
    gpus: int = typer.Option(1, help="Number of GPUs (0 for CPU)."),
    lr: float = typer.Option(1e-3, help="Adam learning rate."),
    precision: str = typer.Option("bf16-mixed", help="Trainer precision: 'bf16-mixed', '16-mixed', or '32-true'."),
    artifact_dir: str = typer.Option("./lightning_logs/bulk", help="Root directory for checkpoints and TensorBoard logs."),
    compile_model: bool = typer.Option(True, "--compile/--no-compile", help="Wrap the model with torch.compile(mode='reduce-overhead')."),
):
    """Run the full sweep of training configurations from the paper."""
    total = len(TRAIN_CONFIGS)
    for idx, config in enumerate(TRAIN_CONFIGS):
        _run_training(
            config,
            idx=idx,
            total=total,
            root_dir=root_dir,
            num_workers=num_workers,
            train_length=train_length,
            eval_length=eval_length,
            gpus=gpus,
            lr=lr,
            precision=precision,
            artifact_dir=artifact_dir,
            compile_model=compile_model,
        )

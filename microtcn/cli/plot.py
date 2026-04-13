import itertools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

app = typer.Typer(no_args_is_help=True, help="Generate plots.")


def _compute_receptive_field(kernel_size: int, nblocks: int, dilation_growth: int, stack_size: int = 10):
    rf = kernel_size
    layer_rf = []
    for n in range(1, nblocks):
        layer_rf.append(rf)
        dilation = dilation_growth ** (n % stack_size)
        rf = rf + ((kernel_size - 1) * dilation)
    layer_rf.append(rf)
    return rf, layer_rf


@app.command("receptive-field")
def receptive_field(out_dir: str = typer.Option("plots", help="Directory to write plots into.")):
    """Plot receptive field growth across architectures."""
    os.makedirs(out_dir, exist_ok=True)

    models = [
        {"name": "TCN-100", "nblocks": 4, "dilation_growth": 10, "kernel_size": 5, "color": "#4053d3"},
        {"name": "TCN-300", "nblocks": 4, "dilation_growth": 10, "kernel_size": 13, "color": "#ddb310"},
        {"name": "TCN-324", "nblocks": 10, "dilation_growth": 2, "kernel_size": 15, "color": "#b51d14"},
        {"name": "TCN-1000", "nblocks": 5, "dilation_growth": 10, "kernel_size": 5, "color": "#00b25d"},
    ]

    fig, ax = plt.subplots()
    sample_rate = 44100

    for model in models:
        _, layer_rf = _compute_receptive_field(
            model["kernel_size"], model["nblocks"], model["dilation_growth"]
        )
        layers = np.arange(len(layer_rf)) + 1
        print(model["name"], layer_rf)
        plt.plot(
            layers,
            (np.array(layer_rf) / sample_rate) * 1e3,
            label=model["name"],
            marker="o",
            color=model["color"],
        )
        plt.hlines(
            (layer_rf[-1] / sample_rate) * 1e3, 1, 10, linestyle="--", colors=model["color"]
        )

    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    plt.xticks(np.arange(10) + 1)
    plt.xticks(np.arange(100, step=10) + 1)
    plt.legend()
    plt.grid(color="lightgray")

    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(out_dir, f"receptive_field_growth.{ext}"))


@app.command("runtime")
def runtime(
    gpu_csv: str = typer.Option("speed_gpu_rtx3090.csv"),
    cpu_csv: str = typer.Option("speed_cpu_macbook_v2.csv"),
    out_dir: str = typer.Option("plots"),
):
    """Plot joint CPU/GPU runtime from benchmark CSVs."""
    os.makedirs(out_dir, exist_ok=True)
    df_gpu = pd.read_csv(gpu_csv, index_col=0)
    df_cpu = pd.read_csv(cpu_csv, index_col=0)

    tcn300_gpu = df_gpu[df_gpu["model_id"] == "TCN-300-C"]
    tcn324_gpu = df_gpu[df_gpu["model_id"] == "TCN-324-N"]
    lstm32_gpu = df_gpu[df_gpu["model_id"] == "LSTM-32-C"]

    tcn300_cpu = df_cpu[df_cpu["model_id"] == "TCN-300-C"]
    tcn324_cpu = df_cpu[df_cpu["model_id"] == "TCN-324-N"]
    lstm32_cpu = df_cpu[df_cpu["model_id"] == "LSTM-32-C"]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    marker = itertools.cycle(("x", "+", ".", "^", "*"))
    color = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    mk = next(marker)
    c = next(color)
    plt.plot(tcn324_gpu["N"], tcn324_gpu["rtf"], c=c, linestyle="-", label="TCN-324-N", marker=mk)
    plt.plot(tcn324_cpu["N"], tcn324_cpu["rtf"], c=c, linestyle="--", marker=mk)

    mk = next(marker)
    c = next(color)
    plt.plot(tcn300_gpu["N"], tcn300_gpu["rtf"], c=c, linestyle="-", label="TCN-300-C", marker=mk)
    plt.plot(tcn300_cpu["N"], tcn300_cpu["rtf"], c=c, linestyle="--", marker=mk)

    mk = next(marker)
    c = next(color)
    plt.plot(lstm32_gpu["N"], lstm32_gpu["rtf"], c=c, linestyle="-", label="LSTM-32", marker=mk)
    plt.plot(lstm32_cpu["N"], lstm32_cpu["rtf"], c=c, linestyle="--", marker=mk)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_yscale("log", base=10)
    ax.set_xscale("log", base=2)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.xticks(tcn324_cpu["N"], rotation=-45)
    plt.xlim([32, 65540])
    plt.ylim([0.01, 1400])
    plt.yticks(
        [0.01, 0.1, 1.0, 10, 100, 1000],
        [f"{n}" for n in [0.01, 0.1, 1.0, 10, 100, 1000]],
    )
    plt.ylabel("Real-time factor")
    plt.xlabel("Frame size")
    plt.legend()
    plt.grid(c="lightgray")
    plt.hlines(1, 32, 65536, linestyles="solid", color="k", linewidth=1)
    plt.tight_layout()

    for ext in ("png", "pdf", "svg"):
        plt.savefig(os.path.join(out_dir, f"speed_cpu+gpu.{ext}"))
    plt.close("all")

import sys
import time
from itertools import product

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import typer

from microtcn.lstm import LSTMModel
from microtcn.tcn_bare import TCNModel


def _compute_receptive_field(nblocks: int, dilation_growth: int, kernel_size: int, stack_size: int = 10) -> int:
    rf = kernel_size
    for n in range(1, nblocks):
        dilation = dilation_growth ** (n % stack_size)
        rf = rf + ((kernel_size - 1) * dilation)
    return rf


def _run(nblocks, dilation_growth, kernel_size, channels, target_rf, model_type="TCN", causal=False, N=44100, gpu=False):
    pl.seed_everything(42)

    dict_args = {
        "nparams": 2,
        "nblocks": nblocks,
        "kernel_size": kernel_size,
        "channel_width": channels,
        "hidden_size": 32,
        "grouped": False,
        "causal": causal,
        "dilation_growth": dilation_growth,
    }

    sr = 44100
    duration = N / sr
    n_iters = 100
    timings = []

    if model_type == "TCN":
        rf = _compute_receptive_field(nblocks, dilation_growth, kernel_size)
        samples = N + rf
        if target_rf != -1:
            if (rf / sr) * 1e3 > target_rf * 2:
                return rf, 0
            if (rf / sr) * 1e3 < target_rf:
                return rf, 0
        model = TCNModel(**dict_args)
        input_ = (torch.rand(1, 1, samples) * 2) - 1
    else:
        rf = 0
        model = LSTMModel(**dict_args)
        input_ = (torch.rand(1, 1, N) * 2) - 1

    num_params = sum(p.numel() for p in model.parameters())
    print(
        f"{model_type} has {num_params} parameters with r.f. {(rf / sr) * 1e3:0.1f} ms requiring input size {N + rf}"
    )

    params = torch.rand(1, 1, 2) if dict_args["nparams"] > 0 else None

    if gpu:
        model.cuda()
        input_ = input_.to("cuda:0")
        if params is not None:
            params = params.to("cuda:0")

    model.eval()
    with torch.no_grad():
        for n in range(n_iters):
            tic = time.perf_counter()
            model(input_, params)
            toc = time.perf_counter()
            timings.append(toc - tic)
            sys.stdout.write(f"{n + 1:3d}/{n_iters:3d}\r")
            sys.stdout.flush()

    mean_time_s = np.mean(timings)
    mean_time_ms = mean_time_s * 1e3
    sec_sec = (1 / duration) * mean_time_s
    rtf = duration / mean_time_s
    rf_ms = (rf / sr) * 1e3
    print(f"Avg. time: {mean_time_ms:0.1f} ms  | sec/sec {sec_sec:0.3f} |  RTF: {rtf:0.2f}x")

    return rf_ms, rtf


def speed(
    full: bool = typer.Option(False, help="Sweep every architecture combination."),
    gpu: bool = typer.Option(False),
    rf: int = typer.Option(0, help="Target receptive field in ms (only for --full)."),
    output: str = typer.Option("", help="CSV output path (defaults to speed_gpu.csv or speed_cpu.csv)."),
):
    """Benchmark model runtime across frame sizes and architectures."""
    max_dilation = 128
    max_blocks = 5
    max_kernel = 33

    dilation_factors = np.arange(1, max_dilation + 1)
    nblocks = np.arange(1, max_blocks + 1)
    kernels = np.arange(3, max_kernel + 1, step=2)

    candidates: list = []

    if full:
        for b, d, k in product(nblocks, dilation_factors, kernels):
            print(b, d, k)
            rf_ms, rtf = _run(b, d, k, rf, N=512)
            if rf_ms > rf:
                candidates.append(
                    {"kernel": k, "dilation": d, "blocks": b, "rf": rf_ms, "rtf": rtf}
                )
    else:
        frame_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
        causal_opts = [True, False]
        for c, N in product(causal_opts, frame_sizes):
            model_ids = ["TCN-370", "TCN-100", "TCN-300", "TCN-1000", "TCN-324", "LSTM-32", "TCN-324-16", "TCN-324-8"]
            model_types = ["TCN", "TCN", "TCN", "TCN", "TCN", "LSTM", "TCN", "TCN"]
            nblocks_list = [3, 4, 4, 5, 10, 0, 10, 10]
            dilation_list = [64, 10, 10, 10, 2, 0, 2, 2]
            kernel_list = [5, 5, 13, 5, 15, 0, 15, 15]
            channel_list = [32, 32, 32, 32, 32, 0, 16, 8]
            for mid, m, b, d, k, ch in zip(
                model_ids, model_types, nblocks_list, dilation_list, kernel_list, channel_list
            ):
                print(b, d, k, ch)
                rf_ms, rtf = _run(b, d, k, ch, -1, causal=c, N=N, model_type=m, gpu=gpu)
                mid += "-C" if c else "-N"
                candidates.append(
                    {
                        "model_id": mid,
                        "causal": c,
                        "kernel": k,
                        "dilation": d,
                        "blocks": b,
                        "channels": ch,
                        "rf": rf_ms,
                        "rtf": rtf,
                        "N": N,
                    }
                )

    df = pd.DataFrame(candidates)
    print(df)
    out_path = output or ("speed_gpu.csv" if gpu else "speed_cpu.csv")
    df.to_csv(out_path)

    print("-" * 50)
    print("     ID      RTF       RF      Blocks  Dilation   Kernel")
    print("-" * 50)
    for n, c in enumerate(candidates[:11]):
        print(
            f"{n: 3d} {c.get('model_id', ''):>12s}  {c['rtf']: 2.2f}x  {c['rf']:0.1f} ms    {c['blocks']}        {c['dilation']}        {c['kernel']}"
        )
    print("-" * 50)

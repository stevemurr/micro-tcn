"""EGFxSet preprocessing: pair clean DI with wet effect, align per clip.

Emits the same ``{subset}_input.bin`` / ``{subset}_target.bin`` / ``{subset}_index.json``
layout that :mod:`microtcn.data` already mmap-loads, so the trainer can consume it
without touching the cache code path.

Per-clip alignment is necessary because the source recordings have random reamp
round-trip latency (up to ~4k samples per clip). Without it, sample-level losses
misbehave; even MR-STFT benefits from the phase coherence.
"""
import json
import os
import random
import sys
import wave

import numpy as np
import torch


CACHE_VERSION = 1


def _load_wav24(path: str):
    with wave.open(path, "rb") as w:
        nc = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    if nc != 1:
        raise RuntimeError(f"{path}: expected mono, got {nc} channels")
    if sw != 3:
        raise RuntimeError(f"{path}: expected 24-bit PCM, got {sw * 8}-bit")
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
    i32 = (
        a[:, 0].astype(np.int32)
        | a[:, 1].astype(np.int32) << 8
        | a[:, 2].astype(np.int32) << 16
    )
    i32 = np.where(i32 & 0x800000, i32 - (1 << 24), i32)
    return i32.astype(np.float32) / (1 << 23), sr


def _best_lag(clean: np.ndarray, wet: np.ndarray, max_lag: int):
    """FFT cross-correlation, restricted to ``|lag| < max_lag``. Positive lag
    means ``wet`` is delayed relative to ``clean``."""
    n = len(clean) + len(wet) - 1
    C = np.fft.rfft(clean, n)
    W = np.fft.rfft(wet, n)
    cc = np.fft.irfft(C * np.conj(W), n)
    norm = float(np.sqrt((clean ** 2).sum() * (wet ** 2).sum())) + 1e-12
    cc = cc / norm
    lags = np.concatenate([np.arange(0, max_lag), np.arange(-max_lag, 0)])
    i = int(np.argmax(np.abs(cc[lags])))
    return int(lags[i]), float(cc[lags[i]])


def _align_pair(clean: np.ndarray, wet: np.ndarray, lag: int):
    """Shift ``wet`` by ``lag`` samples relative to ``clean``, crop both to overlap."""
    if lag >= 0:
        c = clean[: len(clean) - lag] if lag > 0 else clean
        w = wet[lag:]
    else:
        c = clean[-lag:]
        w = wet[: len(wet) + lag]
    m = min(len(c), len(w))
    return c[:m], w[:m]


def _discover_pairs(clean_root: str, wet_root: str):
    """Return list of (pickup, fname, clean_path, wet_path) for every wet file
    that has a matching-name clean file."""
    pairs = []
    orphans = []
    pickups = sorted(
        p for p in os.listdir(wet_root) if os.path.isdir(os.path.join(wet_root, p))
    )
    for pickup in pickups:
        wet_dir = os.path.join(wet_root, pickup)
        clean_dir = os.path.join(clean_root, pickup)
        for f in sorted(os.listdir(wet_dir)):
            if not f.endswith(".wav"):
                continue
            cp = os.path.join(clean_dir, f)
            wp = os.path.join(wet_dir, f)
            if not os.path.exists(cp):
                orphans.append((pickup, f))
                continue
            pairs.append((pickup, f, cp, wp))
    return pairs, orphans


def _write_subset(
    subset_name: str,
    subset_pairs,
    cache_dir: str,
    max_lag: int,
    min_corr: float,
    verbose: bool,
):
    input_bin = os.path.join(cache_dir, f"{subset_name}_input.bin")
    target_bin = os.path.join(cache_dir, f"{subset_name}_target.bin")
    index_path = os.path.join(cache_dir, f"{subset_name}_index.json")
    tmp_i = input_bin + ".tmp"
    tmp_t = target_bin + ".tmp"
    tmp_idx = index_path + ".tmp"

    files_meta = []
    offset = 0
    sample_rate = None
    dropped = []

    try:
        with open(tmp_i, "wb") as fi, open(tmp_t, "wb") as ft:
            n = len(subset_pairs)
            for i, (pickup, fname, cp, wp) in enumerate(subset_pairs):
                if verbose:
                    sys.stdout.write(
                        f"* {subset_name} {i + 1:4d}/{n:4d}: {pickup}/{fname}          \r"
                    )
                    sys.stdout.flush()

                clean, sr_c = _load_wav24(cp)
                wet, sr_w = _load_wav24(wp)
                if sample_rate is None:
                    sample_rate = sr_c
                if sr_c != sample_rate or sr_w != sample_rate:
                    raise RuntimeError(
                        f"{pickup}/{fname}: sample rate mismatch "
                        f"({sr_c} / {sr_w} vs {sample_rate})"
                    )

                lag, corr = _best_lag(clean, wet, max_lag=max_lag)
                if corr < min_corr:
                    dropped.append({"id": f"{pickup}/{fname}", "corr": corr, "lag": lag})
                    continue

                c, w = _align_pair(clean, wet, lag)
                length = len(c)

                c_i16 = np.clip(c * 32768.0, -32768, 32767).astype(np.int16)
                w_i16 = np.clip(w * 32768.0, -32768, 32767).astype(np.int16)
                fi.write(c_i16.tobytes())
                ft.write(w_i16.tobytes())

                string_idx, fret_idx = (
                    int(x) for x in fname.replace(".wav", "").split("-")
                )
                files_meta.append({
                    "id": f"{pickup}/{fname}",
                    "offset": offset,
                    "length": length,
                    "pickup": pickup,
                    "string": string_idx,
                    "fret": fret_idx,
                    "lag": lag,
                    "align_corr": corr,
                    "clipped_input": bool(np.abs(clean).max() >= 0.999),
                    "params": [],
                })
                offset += length

            if verbose:
                sys.stdout.write("\n")

        meta = {
            "version": CACHE_VERSION,
            "dataset_type": "egfx",
            "subset": subset_name,
            "sample_rate": sample_rate,
            "nparams": 0,
            "param_names": [],
            "files": files_meta,
            "dropped": dropped,
        }
        with open(tmp_idx, "w") as f:
            json.dump(meta, f, indent=2)

        os.replace(tmp_i, input_bin)
        os.replace(tmp_t, target_bin)
        os.replace(tmp_idx, index_path)

        if verbose:
            total_sec = offset / sample_rate if sample_rate else 0
            n_clipped = sum(1 for f in files_meta if f["clipped_input"])
            print(
                f"  {subset_name}: {len(files_meta)} pairs  {total_sec / 60:.2f} min  "
                f"(dropped {len(dropped)}, clipped_input {n_clipped})"
            )
        return len(files_meta)
    except BaseException:
        for p in (tmp_i, tmp_t, tmp_idx):
            try:
                os.remove(p)
            except OSError:
                pass
        raise


def build_egfx_cache(
    clean_root: str,
    wet_root: str,
    cache_dir: str,
    val_frac: float = 0.1,
    seed: int = 42,
    max_lag: int = 8000,
    min_corr: float = 0.3,
    verbose: bool = True,
):
    """Discover clean/wet pairs, align each, write train + val int16 caches.

    Args:
        clean_root: path to the ``Clean`` EGFxSet folder (5 pickup subdirs).
        wet_root:   path to the effect folder, e.g. ``TubeScreamer``.
        cache_dir:  output directory for bin + index files.
        val_frac:   fraction of pairs reserved for validation.
        seed:       RNG seed for the train/val split.
        max_lag:    search window for per-clip cross-correlation alignment (samples).
        min_corr:   minimum absolute cross-correlation to accept a pair;
                    below this, alignment is unreliable so the pair is dropped.
    """
    os.makedirs(cache_dir, exist_ok=True)
    pairs, orphans = _discover_pairs(clean_root, wet_root)
    if verbose:
        print(f"discovered {len(pairs)} wet/clean pairs  (orphans: {orphans})")

    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(round(len(pairs) * val_frac)))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    _write_subset("train", train_pairs, cache_dir, max_lag, min_corr, verbose)
    _write_subset("val", val_pairs, cache_dir, max_lag, min_corr, verbose)

    build_meta = {
        "version": CACHE_VERSION,
        "clean_root": os.path.abspath(clean_root),
        "wet_root": os.path.abspath(wet_root),
        "val_frac": val_frac,
        "seed": seed,
        "max_lag": max_lag,
        "min_corr": min_corr,
        "orphans": [{"pickup": p, "file": f} for p, f in orphans],
    }
    with open(os.path.join(cache_dir, "build.json"), "w") as f:
        json.dump(build_meta, f, indent=2)
    return cache_dir


class EGFxDataset(torch.utils.data.Dataset):
    """Paired dataset for an EGFxSet effect, aligned per clip.

    Reads the int16 cache produced by :func:`build_egfx_cache` and serves fixed-length
    windows. Returns ``(input, target, params)`` where ``params`` is an empty tensor
    by default — the TubeScreamer folder is recorded at a single knob setting, so
    there's nothing to condition on.
    """

    def __init__(
        self,
        cache_dir: str,
        subset: str = "train",
        length: int = 16384,
        dtype: torch.dtype = torch.float32,
    ):
        input_bin = os.path.join(cache_dir, f"{subset}_input.bin")
        target_bin = os.path.join(cache_dir, f"{subset}_target.bin")
        index_path = os.path.join(cache_dir, f"{subset}_index.json")
        for p in (input_bin, target_bin, index_path):
            if not os.path.exists(p):
                raise RuntimeError(
                    f"Missing cache file {p}. Run build_egfx_cache() first."
                )

        with open(index_path) as f:
            self.meta = json.load(f)
        self.sample_rate = self.meta["sample_rate"]
        self.nparams = self.meta.get("nparams", 0)
        self.param_names = self.meta.get("param_names", [])
        self.dataset_type = self.meta.get("dataset_type", "egfx")
        self.length = length
        self.dtype = dtype
        self.input_store = np.memmap(input_bin, dtype=np.int16, mode="r")
        self.target_store = np.memmap(target_bin, dtype=np.int16, mode="r")

        self.examples = []
        for fm in self.meta["files"]:
            file_offset = fm["offset"]
            file_length = fm["length"]
            for n_patch in range(file_length // length):
                self.examples.append({
                    "global_offset": file_offset + n_patch * length,
                    "pickup": fm["pickup"],
                    "string": fm["string"],
                    "fret": fm["fret"],
                })

        self.minutes = (length * len(self.examples) / self.sample_rate) / 60
        print(
            f"EGFx {subset}: {len(self.examples)} examples "
            f"totaling {self.minutes:.2f} min  (window={length} samples)"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        start = ex["global_offset"]
        end = start + self.length

        input_np = np.array(self.input_store[start:end])
        target_np = np.array(self.target_store[start:end])

        input_t = (torch.from_numpy(input_np).to(self.dtype) / 32768.0).unsqueeze(0)
        target_t = (torch.from_numpy(target_np).to(self.dtype) / 32768.0).unsqueeze(0)

        if np.random.rand() > 0.5:
            input_t = -input_t
            target_t = -target_t

        params = torch.empty(1, 0, dtype=self.dtype)
        return input_t, target_t, params

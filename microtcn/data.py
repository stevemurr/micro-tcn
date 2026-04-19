import os
import sys
import glob
import json

import torch
import numpy as np
from torchcodec.decoders import AudioDecoder


CACHE_VERSION = 2


def _cache_paths(cache_dir: str, subset: str):
    return (
        os.path.join(cache_dir, f"{subset}_input.bin"),
        os.path.join(cache_dir, f"{subset}_target.bin"),
        os.path.join(cache_dir, f"{subset}_index.json"),
    )


def _cache_is_valid(cache_dir: str, subset: str) -> bool:
    input_bin, target_bin, index_path = _cache_paths(cache_dir, subset)
    if not (
        os.path.exists(input_bin)
        and os.path.exists(target_bin)
        and os.path.exists(index_path)
    ):
        return False
    try:
        with open(index_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if meta.get("version") != CACHE_VERSION:
        return False
    expected_bytes = sum(f["length"] for f in meta["files"]) * 2  # int16
    if (
        os.path.getsize(input_bin) != expected_bytes
        or os.path.getsize(target_bin) != expected_bytes
    ):
        return False
    return True


def _build_cache(input_files, target_files, cache_dir: str, subset: str):
    """Decode every WAV pair once and concatenate into two flat int16 stores.

    Writes atomically via ``.tmp`` + rename so a crashed build never leaves a
    partial cache that passes the size check on the next run.
    """
    os.makedirs(cache_dir, exist_ok=True)
    input_bin, target_bin, index_path = _cache_paths(cache_dir, subset)
    tmp_input = input_bin + ".tmp"
    tmp_target = target_bin + ".tmp"
    tmp_index = index_path + ".tmp"

    files_meta = []
    offset_samples = 0
    sample_rate = None
    n = len(input_files)

    try:
        with open(tmp_input, "wb") as fi, open(tmp_target, "wb") as ft:
            for idx, (ifile, tfile) in enumerate(zip(input_files, target_files)):
                sys.stdout.write(
                    f"* Building {subset} cache... {idx + 1:3d}/{n:3d} {os.path.basename(ifile)}   \r"
                )
                sys.stdout.flush()

                ifile_id = int(os.path.basename(ifile).split("_")[1])
                tfile_id = int(os.path.basename(tfile).split("_")[1])
                if ifile_id != tfile_id:
                    raise RuntimeError(
                        f"Non-matching file ids: {ifile_id} != {tfile_id}"
                    )

                i_dec = AudioDecoder(ifile)
                t_dec = AudioDecoder(tfile)
                sr = i_dec.metadata.sample_rate
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    raise RuntimeError(
                        f"Sample rate mismatch in {ifile}: {sr} vs {sample_rate}"
                    )

                i_samples = i_dec.get_all_samples().data
                t_samples = t_dec.get_all_samples().data
                if i_samples.shape[0] != 1 or t_samples.shape[0] != 1:
                    raise RuntimeError(
                        f"Expected mono audio, got shapes {tuple(i_samples.shape)} / "
                        f"{tuple(t_samples.shape)} for {ifile}"
                    )

                length = int(min(i_samples.shape[-1], t_samples.shape[-1]))
                i_int16 = (
                    (i_samples[..., :length] * 32768.0)
                    .clamp(-32768, 32767)
                    .to(torch.int16)
                    .contiguous()
                    .view(-1)
                )
                t_int16 = (
                    (t_samples[..., :length] * 32768.0)
                    .clamp(-32768, 32767)
                    .to(torch.int16)
                    .contiguous()
                    .view(-1)
                )

                fi.write(i_int16.numpy().tobytes())
                ft.write(t_int16.numpy().tobytes())

                params = (
                    float(os.path.basename(tfile).split("__")[1]),
                    float(os.path.basename(tfile).split("__")[2].replace(".wav", "")),
                )
                files_meta.append(
                    {
                        "id": ifile_id,
                        "offset": offset_samples,
                        "length": length,
                        "params": list(params),
                    }
                )
                offset_samples += length

        sys.stdout.write("\n")

        meta = {
            "version": CACHE_VERSION,
            "dataset_type": "signaltrain_la2a",
            "subset": subset,
            "sample_rate": sample_rate,
            "nparams": 2,
            "param_names": ["limit", "peak_reduction"],
            "files": files_meta,
        }
        with open(tmp_index, "w") as f:
            json.dump(meta, f)

        os.replace(tmp_input, input_bin)
        os.replace(tmp_target, target_bin)
        os.replace(tmp_index, index_path)
    except BaseException:
        for p in (tmp_input, tmp_target, tmp_index):
            try:
                os.remove(p)
            except OSError:
                pass
        raise


class SignalTrainLA2ADataset(torch.utils.data.Dataset):
    """SignalTrain LA2A dataset. Source: [10.5281/zenodo.3824876](https://zenodo.org/record/3824876).

    Decodes every WAV once into a pair of flat int16 files under ``cache_dir`` and
    memory-maps them at load time. Workers share the OS page cache instead of
    each holding their own decoded copy, so ``num_workers`` no longer multiplies
    RAM usage.
    """

    def __init__(
        self,
        root_dir,
        subset="train",
        length=16384,
        dtype=torch.float32,
        cache_dir=None,
    ):
        """
        Args:
            root_dir (str): SignalTrain dataset root.
            subset (str): "train" | "val" | "test" | "full". (Default: "train")
            length (int): Samples per returned example. (Default: 16384)
            dtype (torch.dtype): Output dtype. Returned tensors are in [-1, 1].
                (Default: float32)
            cache_dir (str, optional): Where to keep the mmap store.
                Defaults to ``{root_dir}/.cache``.
        """
        self.root_dir = root_dir
        self.subset = subset
        self.length = length
        self.dtype = dtype
        self.cache_dir = cache_dir or os.path.join(root_dir, ".cache")

        if subset == "full":
            target_files = sorted(
                glob.glob(os.path.join(root_dir, "**", "target_*.wav"))
            )
            input_files = sorted(
                glob.glob(os.path.join(root_dir, "**", "input_*.wav"))
            )
        else:
            target_files = sorted(
                glob.glob(os.path.join(root_dir, subset.capitalize(), "target_*.wav"))
            )
            input_files = sorted(
                glob.glob(os.path.join(root_dir, subset.capitalize(), "input_*.wav"))
            )

        if len(input_files) == 0:
            raise RuntimeError(
                f"No input files found under {root_dir} for subset={subset!r}"
            )
        if len(input_files) != len(target_files):
            raise RuntimeError(
                f"Input/target file count mismatch: {len(input_files)} vs {len(target_files)}"
            )

        if not _cache_is_valid(self.cache_dir, subset):
            _build_cache(input_files, target_files, self.cache_dir, subset)

        input_bin, target_bin, index_path = _cache_paths(self.cache_dir, subset)
        with open(index_path) as f:
            self.meta = json.load(f)
        self.sample_rate = self.meta["sample_rate"]
        self.nparams = self.meta.get("nparams", 2)
        self.param_names = self.meta.get("param_names", ["limit", "peak_reduction"])
        self.dataset_type = self.meta.get("dataset_type", "signaltrain_la2a")
        self.input_store = np.memmap(input_bin, dtype=np.int16, mode="r")
        self.target_store = np.memmap(target_bin, dtype=np.int16, mode="r")

        self.examples = []
        for file_meta in self.meta["files"]:
            file_offset = file_meta["offset"]
            file_length = file_meta["length"]
            params = tuple(file_meta["params"])
            for n_patch in range(file_length // self.length):
                self.examples.append(
                    {
                        "global_offset": file_offset + n_patch * self.length,
                        "params": params,
                    }
                )

        self.minutes = ((self.length * len(self.examples)) / self.sample_rate) / 60
        print(
            f"Located {len(self.examples)} examples totaling {self.minutes:0.2f} min in the {subset} subset."
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        start = ex["global_offset"]
        end = start + self.length

        # np.array materializes the memmap slice into a fresh, writable buffer
        # so torch.from_numpy doesn't produce a read-only tensor.
        input_np = np.array(self.input_store[start:end])
        target_np = np.array(self.target_store[start:end])

        # int16 → float in [-1, 1], matching the model's tanh output range
        input_t = (torch.from_numpy(input_np).to(self.dtype) / 32768.0).unsqueeze(0)
        target_t = (torch.from_numpy(target_np).to(self.dtype) / 32768.0).unsqueeze(0)

        if np.random.rand() > 0.5:
            input_t = -input_t
            target_t = -target_t

        params = torch.tensor(ex["params"]).unsqueeze(0)
        params[:, 1] /= 100

        return input_t, target_t, params


# ---------------------------------------------------------------------------
# Loader registry
# ---------------------------------------------------------------------------

def _egfx_factory(data_dir, subset, length, dtype):
    from microtcn.egfx import EGFxDataset
    return EGFxDataset(cache_dir=data_dir, subset=subset, length=length, dtype=dtype)


def _la2a_factory(data_dir, subset, length, dtype):
    return SignalTrainLA2ADataset(
        root_dir=data_dir, subset=subset, length=length, dtype=dtype
    )


DATASET_REGISTRY: dict[str, dict] = {
    "signaltrain_la2a": {
        "factory": _la2a_factory,
        "data_dir_help": "Raw SignalTrain dataset root (cache auto-built under .cache/).",
    },
    "egfx": {
        "factory": _egfx_factory,
        "data_dir_help": "Prebuilt EGFx cache directory (see `microtcn build-egfx`).",
    },
}


def _read_cache_meta(data_dir: str, subset: str):
    """Try to load subset index.json from either the dir itself or its .cache/ subdir.

    Returns (meta_dict, path_used) or (None, None) if nothing is found.
    """
    candidates = [
        os.path.join(data_dir, f"{subset}_index.json"),
        os.path.join(data_dir, ".cache", f"{subset}_index.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f), p
            except (OSError, json.JSONDecodeError):
                continue
    return None, None


def load_dataset(
    data_dir: str,
    subset: str = "train",
    length: int = 16384,
    *,
    loader: str | None = None,
    dtype: torch.dtype = torch.float32,
):
    """Open a dataset by auto-detecting its loader from cache metadata.

    Args:
        data_dir: dataset-specific location — raw root or prebuilt cache
                  (see each loader's ``data_dir_help``).
        subset: "train" | "val" | "test" | "full".
        length: samples per example.
        loader: optional explicit loader name, overriding metadata. Useful when
                the cache predates the ``dataset_type`` field or to force a
                mismatching loader for debugging.
        dtype: returned tensor dtype.

    Resolution order:
      1. If ``loader`` is given, use it directly.
      2. Otherwise read ``dataset_type`` from the cache's ``{subset}_index.json``.
      3. Raise a structured error naming available loaders and how to register
         a new one.
    """
    if loader is None:
        meta, meta_path = _read_cache_meta(data_dir, subset)
        if meta is None:
            avail = ", ".join(sorted(DATASET_REGISTRY))
            raise FileNotFoundError(
                f"No dataset cache index.json found at {data_dir!r} for subset={subset!r}.\n"
                f"Looked at: {os.path.join(data_dir, f'{subset}_index.json')} and "
                f"{os.path.join(data_dir, '.cache', f'{subset}_index.json')}.\n"
                f"Either build a cache first (e.g. `microtcn build-egfx --cache-dir {data_dir}`) "
                f"or pass --loader explicitly. Available loaders: {avail}."
            )
        loader = meta.get("dataset_type")
        if loader is None:
            avail = ", ".join(sorted(DATASET_REGISTRY))
            raise RuntimeError(
                f"Cache at {meta_path} has no `dataset_type` field — it predates the loader registry.\n"
                f"Fix: rebuild the cache (bump CACHE_VERSION already forces this on next run) "
                f"or pass --loader explicitly. Available loaders: {avail}."
            )

    if loader not in DATASET_REGISTRY:
        avail = "\n  - ".join(
            f"{k}: {v['data_dir_help']}" for k, v in sorted(DATASET_REGISTRY.items())
        )
        raise ValueError(
            f"Unknown dataset loader {loader!r}.\n"
            f"Available loaders:\n  - {avail}\n\n"
            f"To add a new loader:\n"
            f"  1. Write a torch.utils.data.Dataset class (see microtcn/egfx.py EGFxDataset for a template).\n"
            f"  2. Add an entry to DATASET_REGISTRY in microtcn/data.py with a factory callable.\n"
            f"  3. Have your builder write `dataset_type: \"<your_key>\"` into the cache's index.json.\n"
            f"  4. Expose `nparams` / `param_names` attributes on the Dataset so the trainer can size the model."
        )

    factory = DATASET_REGISTRY[loader]["factory"]
    return factory(data_dir, subset, length, dtype)

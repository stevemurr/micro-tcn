import os
import sys
import glob
from functools import lru_cache

import torch
import numpy as np
from torchcodec.decoders import AudioDecoder


@lru_cache(maxsize=64)
def _decode_full_file(path: str):
    """Decode a WAV once and cache the full-file tensor.

    Keeps the legacy ``torchaudio.load(..., normalize=False)`` magnitude
    (values in ``[-32768, 32767]``) as float32 so consumers can slice cheaply.
    The cache is per-process, so each DataLoader worker maintains its own; 64
    files of mono 44.1 kHz audio is ~3-6 GB per worker depending on length.
    """
    dec = AudioDecoder(path)
    sr = dec.metadata.sample_rate
    samples = dec.get_all_samples()
    data = (samples.data * 32768.0).clamp(-32768, 32767).contiguous()
    return data, sr


def _load_int16(path: str, start: int = 0, length: int | None = None):
    """Return a float32 view of shape (C, N) into the cached decoded file."""
    data, sr = _decode_full_file(path)
    if length is None:
        return data, sr
    return data[..., start:start + length], sr


class SignalTrainLA2ADataset(torch.utils.data.Dataset):
    """ SignalTrain LA2A dataset. Source: [10.5281/zenodo.3824876](https://zenodo.org/record/3824876)."""
    def __init__(self, root_dir, subset="train", length=16384, preload=False, dtype=torch.float32, fraction=1.0):
        """
        Args:
            root_dir (str): Path to the root directory of the SignalTrain dataset.
            subset (str, optional): Pull data either from "train", "val", "test", or "full" subsets. (Default: "train")
            length (int, optional): Number of samples in the returned examples. (Default: 40)
            preload (bool, optional): Read in all data into RAM during init. (Default: False)
            dtype (torch.dtype, optional): Output dtype for audio tensors. Use
                ``torch.bfloat16`` / ``torch.float16`` to halve host→GPU
                bandwidth under mixed-precision training. (Default: float32)
            fraction (float, optional): Fraction of the data to load from the subset. (Default: 1.0)
        """
        self.root_dir = root_dir
        self.subset = subset
        self.length = length
        self.preload = preload
        self.dtype = dtype
        self.fraction = fraction

        if self.subset == "full":
            self.target_files = glob.glob(os.path.join(self.root_dir, "**", "target_*.wav"))
            self.input_files  = glob.glob(os.path.join(self.root_dir, "**", "input_*.wav"))
        else:
            # get all the target files files in the directory first
            self.target_files = glob.glob(os.path.join(self.root_dir, self.subset.capitalize(), "target_*.wav"))
            self.input_files  = glob.glob(os.path.join(self.root_dir, self.subset.capitalize(), "input_*.wav"))

        self.examples = []
        self.minutes = 0  # total number of hours of minutes in the subset

        # ensure that the sets are ordered correctlty
        self.target_files.sort()
        self.input_files.sort()

        # get the parameters
        self.params = [(float(f.split("__")[1].replace(".wav","")), float(f.split("__")[2].replace(".wav",""))) for f in self.target_files]

        sample_rate = None

        # loop over files to count total length
        for idx, (tfile, ifile, params) in enumerate(zip(self.target_files, self.input_files, self.params)):

            ifile_id = int(os.path.basename(ifile).split("_")[1])
            tfile_id = int(os.path.basename(tfile).split("_")[1])
            if ifile_id != tfile_id:
                raise RuntimeError(f"Found non-matching file ids: {ifile_id} != {tfile_id}! Check dataset.")

            md = AudioDecoder(tfile).metadata
            sample_rate = md.sample_rate
            num_frames = int(md.duration_seconds * sample_rate)

            if self.preload:
                sys.stdout.write(f"* Pre-loading... {idx+1:3d}/{len(self.target_files):3d} ...\r")
                sys.stdout.flush()
                input, sr  = self.load(ifile)
                target, sr = self.load(tfile)

                num_frames = int(np.min([input.shape[-1], target.shape[-1]]))
                if input.shape[-1] != target.shape[-1]:
                    print(os.path.basename(ifile), input.shape[-1], os.path.basename(tfile), target.shape[-1])
                    raise RuntimeError("Found potentially corrupt file!")
                input = input.to(self.dtype)
                target = target.to(self.dtype)
            else:
                input = None
                target = None

            # create one entry for each patch
            self.file_examples = []
            for n in range((num_frames // self.length)):
                offset = int(n * self.length)
                end = offset + self.length
                self.file_examples.append({"idx": idx,
                                           "target_file" : tfile,
                                           "input_file" : ifile,
                                           "input_audio" : input[:,offset:end] if input is not None else None,
                                           "target_audio" : target[:,offset:end] if input is not None else None,
                                           "params" : params,
                                           "offset": offset,
                                           "frames" : num_frames})

            # add to overall file examples
            self.examples += self.file_examples

        # use only a fraction of the subset data if applicable
        if self.subset == "train":
            classes = set([ex['params'] for ex in self.examples])
            n_classes = len(classes) # number of unique compressor configurations
            fraction_examples = int(len(self.examples) * self.fraction)
            n_examples_per_class = int(fraction_examples / n_classes)
            n_min_total = ((self.length * n_examples_per_class * n_classes) / sample_rate) / 60
            n_min_per_class = ((self.length * n_examples_per_class) / sample_rate) / 60
            print(sorted(classes))
            print(f"Total Examples: {len(self.examples)}     Total classes: {n_classes}")
            print(f"Fraction examples: {fraction_examples}    Examples/class: {n_examples_per_class}")
            print(f"Training with {n_min_per_class:0.2f} min per class    Total of {n_min_total:0.2f} min")

            if n_examples_per_class <= 0:
                raise ValueError(f"Fraction `{self.fraction}` set too low. No examples selected.")

            sampled_examples = []

            for config_class in classes: # select N examples from each class
                class_examples = [ex for ex in self.examples if ex["params"] == config_class]
                example_indices = np.random.randint(0, high=len(class_examples), size=n_examples_per_class)
                class_examples = [class_examples[idx] for idx in example_indices]
                extra_factor = int(1/self.fraction)
                sampled_examples += class_examples * extra_factor

            self.examples = sampled_examples

        self.minutes = ((self.length * len(self.examples)) / sample_rate) / 60

        # we then want to get the input files
        print(f"Located {len(self.examples)} examples totaling {self.minutes:0.2f} min in the {self.subset} subset.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if self.preload:
            audio_idx = self.examples[idx]["idx"]
            offset = self.examples[idx]["offset"]
            input = self.examples[idx]["input_audio"]
            target = self.examples[idx]["target_audio"]
        else:
            offset = self.examples[idx]["offset"]
            input, _ = _load_int16(self.examples[idx]["input_file"],
                                   start=offset, length=self.length)
            target, _ = _load_int16(self.examples[idx]["target_file"],
                                    start=offset, length=self.length)
            if input.dtype != self.dtype:
                input = input.to(self.dtype)
                target = target.to(self.dtype)

        # at random with p=0.5 flip the phase; out-of-place because the loader
        # may hand back views into the LRU-cached full-file tensors.
        if np.random.rand() > 0.5:
            input = -input
            target = -target

        # then get the tuple of parameters
        params = torch.tensor(self.examples[idx]["params"]).unsqueeze(0)
        params[:,1] /= 100

        return input, target, params

    def load(self, filename):
        return _load_int16(filename)

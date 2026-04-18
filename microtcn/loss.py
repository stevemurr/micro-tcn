"""L1 + STFT magnitude loss."""
import auraloss
import torch
import torch.nn as nn


class L1STFT(nn.Module):
    def __init__(self, l1_weight: float = 1.0, stft_weight: float = 1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.stft = auraloss.freq.STFTLoss()
        self.l1_w = l1_weight
        self.stft_w = stft_weight

    def forward(self, pred, target):
        l1 = self.l1(pred, target)
        stft = self.stft(pred, target)
        total = self.l1_w * l1 + self.stft_w * stft
        return total, {"l1": l1.detach(), "stft": stft.detach()}

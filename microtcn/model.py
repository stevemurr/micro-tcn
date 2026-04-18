"""TCN model with FiLM conditioning."""
import torch
import torch.nn as nn

from microtcn.utils import causal_crop, center_crop


class FiLM(nn.Module):
    def __init__(self, num_features: int, cond_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.adaptor = nn.Linear(cond_dim, num_features * 2)

    def forward(self, x, cond):
        cond = self.adaptor(cond)
        g, b = torch.chunk(cond, 2, dim=-1)
        g = g.permute(0, 2, 1)
        b = b.permute(0, 2, 1)
        return self.bn(x) * g + b


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, causal: bool):
        super().__init__()
        self.causal = causal
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size=kernel_size, padding=0,
            dilation=dilation, bias=False,
        )
        self.film = FiLM(out_ch, cond_dim=32)
        self.relu = nn.PReLU(out_ch)
        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1, groups=in_ch, bias=False)

    def forward(self, x, cond):
        x_in = x
        x = self.relu(self.film(self.conv1(x), cond))
        crop = causal_crop if self.causal else center_crop
        return x + crop(self.res(x_in), x.shape[-1])


class TCN(nn.Module):
    """Dilated causal/non-causal TCN with FiLM param conditioning and tanh output.

    Output range is [-1, 1]; train on audio normalized to the same range.
    """

    def __init__(
        self,
        nparams: int = 2,
        ninputs: int = 1,
        noutputs: int = 1,
        nblocks: int = 4,
        kernel_size: int = 13,
        dilation_growth: int = 10,
        channel_width: int = 32,
        causal: bool = True,
    ):
        super().__init__()
        self.nparams = nparams
        self.nblocks = nblocks
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.channel_width = channel_width
        self.causal = causal

        self.gen = nn.Sequential(
            nn.Linear(nparams, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
        )

        self.blocks = nn.ModuleList()
        in_ch = ninputs
        for n in range(nblocks):
            out_ch = channel_width
            dilation = dilation_growth ** n
            self.blocks.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, causal))
            in_ch = out_ch

        self.output = nn.Conv1d(out_ch, noutputs, kernel_size=1)

    def forward(self, x, p):
        cond = self.gen(p)
        for block in self.blocks:
            x = block(x, cond)
        return torch.tanh(self.output(x))

    def receptive_field(self) -> int:
        rf = self.kernel_size
        for n in range(1, self.nblocks):
            rf += (self.kernel_size - 1) * (self.dilation_growth ** n)
        return rf

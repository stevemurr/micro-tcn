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
    """Dilated causal/non-causal TCN with FiLM param conditioning.

    Two output heads:
      arch="direct":  y = tanh(conv(features))                   — free-form prediction
      arch="hybrid":  y = sigmoid(g) · x + α · tanh(d)           — gain-modulator + learned
                                                                    coloration residual
    The hybrid head encodes the compressor's physics (gain modulation) as a structural
    prior; the scalar α is initialized to 0 so the network starts as a pure gain-modulator
    and only reaches for additive coloration if the loss demands it.
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
        arch: str = "direct",
    ):
        super().__init__()
        if arch not in ("direct", "hybrid"):
            raise ValueError(f"arch must be 'direct' or 'hybrid', got {arch!r}")
        self.nparams = nparams
        self.noutputs = noutputs
        self.nblocks = nblocks
        self.kernel_size = kernel_size
        self.dilation_growth = dilation_growth
        self.channel_width = channel_width
        self.causal = causal
        self.arch = arch

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

        final_ch = 2 * noutputs if arch == "hybrid" else noutputs
        self.output = nn.Conv1d(out_ch, final_ch, kernel_size=1)

        if arch == "hybrid":
            # Gain channels start with large positive bias so sigmoid(~4) ≈ 0.98;
            # delta channels start at 0. Network begins near-identity.
            with torch.no_grad():
                self.output.bias.zero_()
                self.output.bias[:noutputs].fill_(4.0)
            self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, p):
        x_orig = x
        cond = self.gen(p)
        for block in self.blocks:
            x = block(x, cond)
        raw = self.output(x)

        if self.arch == "direct":
            return torch.tanh(raw)

        no = self.noutputs
        gain = torch.sigmoid(raw[:, :no, :])
        delta = torch.tanh(raw[:, no:, :])
        crop = causal_crop if self.causal else center_crop
        x_aligned = crop(x_orig, gain.shape[-1])
        return gain * x_aligned + self.alpha * delta

    def receptive_field(self) -> int:
        rf = self.kernel_size
        for n in range(1, self.nblocks):
            rf += (self.kernel_size - 1) * (self.dilation_growth ** n)
        return rf

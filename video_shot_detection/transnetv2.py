from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as functional


class Conv3DConfigurable(nn.Module):
    def __init__(self, in_filters: int, filters: int, dilation_rate: int, separable: bool = True, use_bias: bool = True):
        super().__init__()
        if separable:
            self.layers = nn.ModuleList([
                nn.Conv3d(in_filters, 2 * filters, kernel_size=(1, 3, 3),
                          dilation=(1, 1, 1), padding=(0, 1, 1), bias=False),
                nn.Conv3d(2 * filters, filters, kernel_size=(3, 1, 1),
                          dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 0, 0), bias=use_bias),
            ])
        else:
            self.layers = nn.ModuleList([
                nn.Conv3d(in_filters, filters, kernel_size=3,
                          dilation=(dilation_rate, 1, 1), padding=(dilation_rate, 1, 1), bias=use_bias),
            ])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class DilatedDCNNV2(nn.Module):
    def __init__(self, in_filters: int, filters: int, batch_norm: bool = True, activation=None):
        super().__init__()
        self.Conv3D_1 = Conv3DConfigurable(in_filters, filters, 1, use_bias=not batch_norm)
        self.Conv3D_2 = Conv3DConfigurable(in_filters, filters, 2, use_bias=not batch_norm)
        self.Conv3D_4 = Conv3DConfigurable(in_filters, filters, 4, use_bias=not batch_norm)
        self.Conv3D_8 = Conv3DConfigurable(in_filters, filters, 8, use_bias=not batch_norm)
        self.bn = nn.BatchNorm3d(filters * 4, eps=1e-3) if batch_norm else None
        self.activation = activation

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([
            self.Conv3D_1(inputs),
            self.Conv3D_2(inputs),
            self.Conv3D_4(inputs),
            self.Conv3D_8(inputs),
        ], dim=1)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class StackedDDCNNV2(nn.Module):
    def __init__(
        self,
        in_filters: int,
        n_blocks: int,
        filters: int,
        shortcut: bool = True,
        pool_type: str = "avg",
        stochastic_depth_drop_prob: float = 0.0,
    ):
        super().__init__()
        assert pool_type in ("max", "avg")
        self.shortcut = shortcut
        self.DDCNN = nn.ModuleList([
            DilatedDCNNV2(
                in_filters if i == 1 else filters * 4,
                filters,
                activation=functional.relu if i != n_blocks else None,
            )
            for i in range(1, n_blocks + 1)
        ])
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2)) if pool_type == "max" else nn.AvgPool3d(kernel_size=(1, 2, 2))
        self.stochastic_depth_drop_prob = stochastic_depth_drop_prob

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        shortcut = None
        for block in self.DDCNN:
            x = block(x)
            if shortcut is None:
                shortcut = x
            x = functional.relu(x)
        if self.shortcut and shortcut is not None:
            if self.stochastic_depth_drop_prob != 0.0:
                if self.training:
                    x = shortcut if random.random() < self.stochastic_depth_drop_prob else x + shortcut
                else:
                    x = (1 - self.stochastic_depth_drop_prob) * x + shortcut
            else:
                x += shortcut
        return self.pool(x)


class FrameSimilarity(nn.Module):
    def __init__(self, in_filters: int, similarity_dim: int = 128, lookup_window: int = 101, output_dim: int = 128, use_bias: bool = False):
        super().__init__()
        assert lookup_window % 2 == 1
        self.projection = nn.Linear(in_filters, similarity_dim, bias=use_bias)
        self.fc = nn.Linear(lookup_window, output_dim)
        self.lookup_window = lookup_window

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat([torch.mean(f, dim=[3, 4]) for f in inputs], dim=1).transpose(1, 2)
        x = functional.normalize(self.projection(x), p=2, dim=2)
        batch_size, time_window = x.shape[0], x.shape[1]
        sim = functional.pad(torch.bmm(x, x.transpose(1, 2)), [(self.lookup_window - 1) // 2] * 2)
        hw = self.lookup_window
        bi = torch.arange(batch_size, device=x.device).view(batch_size, 1, 1).expand(batch_size, time_window, hw)
        ti = torch.arange(time_window, device=x.device).view(1, time_window, 1).expand(batch_size, time_window, hw)
        li = torch.arange(hw, device=x.device).view(1, 1, hw).expand(batch_size, time_window, hw) + ti
        return functional.relu(self.fc(sim[bi, ti, li]))


class ColorHistograms(nn.Module):
    def __init__(self, lookup_window: int = 101, output_dim: int | None = None):
        super().__init__()
        assert lookup_window % 2 == 1
        self.fc = nn.Linear(lookup_window, output_dim) if output_dim is not None else None
        self.lookup_window = lookup_window

    @staticmethod
    def _compute(frames: torch.Tensor) -> torch.Tensor:
        frames = frames.int()
        batch_size, time_window, height, width, _ = frames.shape
        flat = frames.view(batch_size * time_window, height * width, 3)
        binned = ((flat[:, :, 0] >> 5) << 6) + ((flat[:, :, 1] >> 5) << 3) + (flat[:, :, 2] >> 5)
        prefix = (torch.arange(batch_size * time_window, device=frames.device) << 9).view(-1, 1)
        binned = (binned + prefix).view(-1)
        hist = torch.zeros(batch_size * time_window * 512, dtype=torch.int32, device=frames.device)
        hist.scatter_add_(0, binned, torch.ones(len(binned), dtype=torch.int32, device=frames.device))
        return functional.normalize(hist.view(batch_size, time_window, 512).float(), p=2, dim=2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self._compute(inputs)
        batch_size, time_window = x.shape[0], x.shape[1]
        sim = functional.pad(torch.bmm(x, x.transpose(1, 2)), [(self.lookup_window - 1) // 2] * 2)
        hw = self.lookup_window
        bi = torch.arange(batch_size, device=x.device).view(batch_size, 1, 1).expand(batch_size, time_window, hw)
        ti = torch.arange(time_window, device=x.device).view(1, time_window, 1).expand(batch_size, time_window, hw)
        li = torch.arange(hw, device=x.device).view(1, 1, hw).expand(batch_size, time_window, hw) + ti
        sim = sim[bi, ti, li]
        return functional.relu(self.fc(sim)) if self.fc is not None else sim


class TransNetV2(nn.Module):
    """Shot transition detector. Input: [B, T, 27, 48, 3] uint8; output: (one_hot_logits [B, T, 1], {"many_hot": [B, T, 1]})."""

    INPUT_H = 27
    INPUT_W = 48

    def __init__(self, F: int = 16, L: int = 3, S: int = 2, D: int = 1024):
        super().__init__()
        self.SDDCNN = nn.ModuleList(
            [StackedDDCNNV2(in_filters=3, n_blocks=S, filters=F, stochastic_depth_drop_prob=0.0)] +
            [StackedDDCNNV2(in_filters=(F * 2 ** (i - 1)) * 4, n_blocks=S, filters=F * 2 ** i) for i in range(1, L)]
        )
        self.frame_sim_layer = FrameSimilarity(
            sum([(F * 2 ** i) * 4 for i in range(L)]), lookup_window=101, output_dim=128, similarity_dim=128, use_bias=True,
        )
        self.color_hist_layer = ColorHistograms(lookup_window=101, output_dim=128)
        self.dropout = nn.Dropout(0.5)
        output_dim = ((F * 2 ** (L - 1)) * 4) * 3 * 6 + 128 + 128
        self.fc1 = nn.Linear(output_dim, D)
        self.cls_layer1 = nn.Linear(D, 1)
        self.cls_layer2 = nn.Linear(D, 1)
        self.eval()

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        assert (
            isinstance(inputs, torch.Tensor)
            and list(inputs.shape[2:]) == [self.INPUT_H, self.INPUT_W, 3]
            and inputs.dtype == torch.uint8
        ), f"expected [batch, T, {self.INPUT_H}, {self.INPUT_W}, 3] uint8, got {inputs.shape} {inputs.dtype}"

        x = inputs.permute([0, 4, 1, 2, 3]).float().div_(255.0)
        block_features = []
        for block in self.SDDCNN:
            x = block(x)
            block_features.append(x)

        x = x.permute(0, 2, 3, 4, 1).reshape(x.shape[0], x.shape[2], -1)
        x = torch.cat([self.frame_sim_layer(block_features), x], dim=2)
        x = torch.cat([self.color_hist_layer(inputs), x], dim=2)
        x = functional.relu(self.fc1(x))
        x = self.dropout(x)
        return self.cls_layer1(x), {"many_hot": self.cls_layer2(x)}

# -*- coding: utf-8 -*-            
# @Author : Hao Fan
# @Time : 2024/7/25
import warnings

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


class TiSSD(nn.Module):
    def __init__(self,
                 d_model: int,
                 seq_len: int,
                 d_state: int = 128,
                 d_conv=4,
                 conv_init=None,
                 expand: int = 2,
                 head_dim: int = 64,
                 d_ssm=None,
                 A_init_range=(1, 16),
                 D_has_hdim: bool = False,
                 rms_norm: bool = True,
                 norm_before_gate: bool = False,
                 dt_min: int = 0.001,
                 dt_max: int = 0.1,
                 dt_init_floor: float = 1e-4,
                 dt_limit=(0.0, float("inf")),
                 bias: bool = True,
                 conv_bias: bool = True,
                 chunk_size: int = 256,
                 time_drop_out: float = 0.0,
                 is_time: bool = True,
                 p2p_residual: bool = False,
                 norm_eps: float = 1e-12,
                 ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand

        self.d_inner = self.expand * self.d_model

        self.head_dim = head_dim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm

        assert self.d_ssm % self.head_dim == 0, f'd_ssm {self.d_ssm} is not divisible by head_dim {self.head_dim}!'
        self.n_heads = self.d_ssm // self.head_dim

        if self.n_heads % 8 != 0:
            self.use_equivalent_conv1d = True
            warnings.warn(f'n_heads {self.n_heads} not divisible by 8, actually use \'nn.Conv1d\'!')
        else:
            self.use_equivalent_conv1d = False

        self.D_has_hdim = D_has_hdim
        self.rms_norm = rms_norm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = 'swish'
        self.chunk_size = chunk_size
        self.is_time = is_time

        # Order: [z, x, B, C, dt]
        d_in_proj = 2 * self.d_inner + 2 * self.d_state + self.n_heads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias)

        # Computed convolution dimension
        conv_dim = self.d_ssm + 2 * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1
        )

        if is_time:
            self.conv1d4time_1 = nn.Conv1d(
                in_channels=self.n_heads,
                out_channels=self.n_heads,
                kernel_size=d_conv,
                bias=conv_bias,
                groups=self.n_heads,
                padding=d_conv - 1
            )

            self.conv1d4time_2 = nn.Conv1d(
                in_channels=self.n_heads,
                out_channels=self.n_heads,
                kernel_size=d_conv,
                bias=conv_bias,
                groups=self.n_heads,
                padding=d_conv - 1
            )

            # [batch_size, n_heads, seq_len] -> [batch, n_heads, seq_len]
            self.mlp4time_scale = nn.Sequential(
                nn.Dropout(p=time_drop_out),
                nn.Linear(in_features=seq_len, out_features=seq_len, bias=True),
                nn.SiLU(inplace=True),
                nn.Linear(in_features=seq_len, out_features=seq_len, bias=True),
                # nn.Hardswish(),
            )

            self.time_gate_residual = nn.Sequential(
                nn.Dropout(p=time_drop_out),
                nn.Linear(in_features=seq_len, out_features=(seq_len if p2p_residual else 1)),
                nn.Sigmoid()
            )

            self.time_layer_norm = nn.LayerNorm(seq_len, eps=norm_eps)

        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        if causal_conv1d_fn is None or self.use_equivalent_conv1d:
            self.act = nn.Hardswish(inplace=True)

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.n_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert 0 < A_init_range[0] <= A_init_range[1]
        A = torch.empty(self.n_heads, dtype=torch.float32).uniform_(*A_init_range)
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.n_heads))
        self.D._no_weight_decay = True

        if self.rms_norm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=norm_eps, norm_before_gate=self.norm_before_gate)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, u, time_diff=None):
        """

        :param u: [batch_size, seq_len, d_model]
        :param time_diff: [batch_size, seq_len]
        :return:
        """
        assert self.is_time is not True or time_diff is not None, 'When is_time is True, time_diff cannot be None!'
        z_x_B_C_dt = self.in_proj(u)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (n_heads) or (d_inner, d_state)

        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        # d_mlp = d_inner - d_ssm
        d_mlp = (z_x_B_C_dt.shape[-1] - 2 * self.d_ssm - 2 * self.d_state - self.n_heads) // 2

        z0, x0, z, xBC, dt = torch.split(
            z_x_B_C_dt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.d_state, self.n_heads],
            dim=-1
        )

        if causal_conv1d_fn is None or self.use_equivalent_conv1d or self.activation not in ["silu", "swish"]:
            xBC = self.act(
                self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1), :]
            )

            if self.is_time:
                time_scale = self.mlp4time_scale(time_diff)
                # [batch_size, num_heads, 1] * [batch_size, num_heads, seq_len] -> [batch_size, num_heads, seq_len]
                time_dt = time_diff * time_scale

                time_dt = self.act(
                    self.conv1d4time_1(time_dt)[:, :, :-(self.d_conv - 1)]
                )
                time_dt = self.act(
                    self.conv1d4time_2(time_dt).transpose(1, 2)[:, :-(self.d_conv - 1), :]
                )
                final_dt = dt * time_dt
            else:
                final_dt = dt

        else:
            xBC = causal_conv1d_fn(
                xBC.transpose(1, 2).contiguous(),
                rearrange(self.conv1d.weight, 'd 1 w -> d w'),
                bias=self.conv1d.bias,
                activation=self.activation
            ).transpose(1, 2)

            if self.is_time:
                time_scale = self.mlp4time_scale(time_diff)
                time_dt = time_scale * time_diff

                time_dt = causal_conv1d_fn(
                    time_dt,
                    rearrange(self.conv1d4time_1.weight, 'd 1 w -> d w'),
                    bias=self.conv1d4time_1.bias,
                    activation=self.activation
                )
                time_dt = causal_conv1d_fn(
                    time_dt,
                    rearrange(self.conv1d4time_2.weight, 'd 1 w -> d w'),
                    bias=self.conv1d4time_2.bias,
                    activation=self.activation
                ).transpose(1, 2)
                final_dt = dt * time_dt
            else:
                final_dt = dt

        x, B, C = torch.split(xBC, [self.d_ssm, self.d_state, self.d_state], dim=-1)
        y = mamba_chunk_scan_combined(
            rearrange(x, 'b l (h p) -> b l h p', p=self.head_dim),
            final_dt,
            A,
            rearrange(B, 'b l (g n) -> b l g n', g=1),
            rearrange(C, 'b l (g n) -> b l g n', g=1),
            chunk_size=self.chunk_size,
            D=rearrange(self.D, '(h p) -> h p', p=self.head_dim) if self.D_has_hdim else self.D,
            z=rearrange(z, 'b l (h p) -> b l h p', p=self.head_dim) if not self.rms_norm else None,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            **dt_limit_kwargs
        )

        y = rearrange(y, 'b l h p -> b l (h p)')

        if self.rms_norm:
            y = self.norm(y, z)

        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)

        out = self.out_proj(y)

        if self.is_time:
            time_gate = self.time_gate_residual(time_diff)
            time_dt = self.time_layer_norm(time_gate * time_diff + (1 - time_gate) * time_dt.transpose(-1, -2))
            return out, time_dt
        else:
            return out, None


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_test = torch.randn(8, 50, 64).to(device)
    n_heads = 8
    h_dim = 64 * 2 // n_heads
    time_test = torch.randn(8, n_heads, 50).to(device)
    model = TiSSD(d_model=64, d_state=32, d_conv=4, expand=2, head_dim=h_dim, seq_len=50).to(device)
    y_test, _ = model(x_test, time_test)
    print(y_test.shape)

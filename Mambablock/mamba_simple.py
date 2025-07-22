import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from torch.distributions.dirichlet import Dirichlet as Dir


class Mamba(nn.Module):
    """
    bimamba_type:
        v1: Mamba with single input
        v2: CrossMamba with two inputs (x, z | B, C, t)
    """
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
        use_norm=False,
        input_h=128,
        input_w=128,
        input_c=128,
        patch_size=4
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out
        self.use_norm = use_norm
        self.input_h = input_h
        self.input_w = input_w
        self.input_c = input_c

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        if bimamba_type == "v2":
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
            self.in_proj_b = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)

        else:
            self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # A_b = repeat(
        #     torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
        #     "n -> d n",
        #     d=self.d_inner,
        # ).contiguous()
        # A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        # self.A_b_log = nn.Parameter(A_b_log)
        # self.A_b_log._no_weight_decay = True

        A_c = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_c_log = torch.log(A_c)  # Keep A_c_log in fp32
        self.A_c_log = nn.Parameter(A_c_log)
        self.A_c_log._no_weight_decay = True

        # A_d = repeat(
        #     torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
        #     "n -> d n",
        #     d=self.d_inner,
        # ).contiguous()
        # A_d_log = torch.log(A_d)  # Keep A_d_log in fp32
        # self.A_d_log = nn.Parameter(A_d_log)
        # self.A_d_log._no_weight_decay = True

        A_e = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_e_log = torch.log(A_e)
        self.A_e_log = nn.Parameter(A_e_log)
        self.A_e_log._no_weight_decay = True

        # A_f = repeat(
        #     torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
        #     "n -> d n",
        #     d=self.d_inner,
        # ).contiguous()
        # A_f_log = torch.log(A_f)
        # self.A_f_log = nn.Parameter(A_f_log)
        # self.A_f_log._no_weight_decay = True

        A_g = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_g_log = torch.log(A_g)
        self.A_g_log = nn.Parameter(A_g_log)
        self.A_g_log._no_weight_decay = True

        # A_h = repeat(
        #     torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
        #     "n -> d n",
        #     d=self.d_inner,
        # ).contiguous()
        # A_h_log = torch.log(A_h)
        # self.A_h_log = nn.Parameter(A_h_log)
        # self.A_h_log._no_weight_decay = True

        A_i = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_i_log = torch.log(A_i)
        self.A_i_log = nn.Parameter(A_i_log)
        self.A_i_log._no_weight_decay = True

        # A_j = repeat(
        #     torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
        #     "n -> d n",
        #     d=self.d_inner,
        # ).contiguous()
        # A_j_log = torch.log(A_j)
        # self.A_j_log = nn.Parameter(A_j_log)
        # self.A_j_log._no_weight_decay = True

        A_k = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_k_log = torch.log(A_k)
        self.A_k_log = nn.Parameter(A_k_log)
        self.A_k_log._no_weight_decay = True

        # A_l = repeat(
        #     torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
        #     "n -> d n",
        #     d=self.d_inner,
        # ).contiguous()
        # A_l_log = torch.log(A_l)
        # self.A_l_log = nn.Parameter(A_l_log)
        # self.A_l_log._no_weight_decay = True

        # self.conv1d_b = nn.Conv1d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=d_conv,
        #     groups=self.d_inner,
        #     padding=d_conv - 1,
        #     **factory_kwargs,
        # )

        self.conv1d_c = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        # self.conv1d_d = nn.Conv1d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=d_conv,
        #     groups=self.d_inner,
        #     padding=d_conv - 1,
        #     **factory_kwargs,
        # )
        self.conv1d_e = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        # self.conv1d_f = nn.Conv1d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=d_conv,
        #     groups=self.d_inner,
        #     padding=d_conv - 1,
        #     **factory_kwargs,
        # )
        self.conv1d_g = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d_h = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.conv1d_i = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d_j = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d_k = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.conv1d_l = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        # self.x_proj_b = nn.Linear(
        #     self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        # )
        self.x_proj_c = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        # self.x_proj_d = nn.Linear(
        #     self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        # )
        self.x_proj_e = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        # self.x_proj_f = nn.Linear(
        #     self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        # )
        self.x_proj_g = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        # self.x_proj_h = nn.Linear(
        #     self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        # )
        self.x_proj_i = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        # self.x_proj_j = nn.Linear(
        #     self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        # )
        self.x_proj_k = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        # self.x_proj_l = nn.Linear(
        #     self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        # )

        # self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.dt_proj_c = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        # self.dt_proj_d = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.dt_proj_e = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        # self.dt_proj_f = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.dt_proj_g = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        # self.dt_proj_h = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.dt_proj_i = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        # self.dt_proj_j = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.dt_proj_k = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        # self.dt_proj_l = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        # self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_c = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        # self.D_d = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_e = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        # self.D_f = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_g = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        # self.D_h = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_i = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        # self.D_j = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_k = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        # self.D_l = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True
        # self.D_b._no_weight_decay = True
        self.D_c._no_weight_decay = True
        # self.D_d._no_weight_decay = True
        self.D_e._no_weight_decay = True
        # self.D_f._no_weight_decay = True
        self.D_g._no_weight_decay = True
        # self.D_h._no_weight_decay = True
        self.D_i._no_weight_decay = True
        # self.D_j._no_weight_decay = True
        self.D_k._no_weight_decay = True
        # self.D_l._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        if use_norm:
            self.norm = nn.LayerNorm(self.d_inner)

    def forward(self, hidden_states, extra_emb=None, inference_params=None, alpha=None, MV_flag=None):
        """
        hidden_states: (B, L, D)
        extra_emb: (B, L, D)
        alpha: (1,12) weight distrubution parameter
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None

        


        outs = []

        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        if extra_emb is not None:
            extra_emb = rearrange(
                self.in_proj_b.weight @ rearrange(extra_emb, "b l d -> d (b l)"),
                "d (b l) -> b d l",
                l=seqlen,
            )
            if self.in_proj_b.bias is not None:
                extra_emb = extra_emb + rearrange(self.in_proj_b.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states

            if self.bimamba_type == "v1":
                

                # xz_b = rearrange(xz, "b d (h w c) -> b d (w h c)", h=self.input_h, w=self.input_w, c=self.input_c)

                # xz_e = rearrange(xz, "b d (h w c) -> b d (w c h)", h=self.input_h, w=self.input_w, c=self.input_c)

                # A_c = -torch.exp(self.A_c_log.float())
                # A_d = -torch.exp(self.A_d_log.float())

                # A_i = -torch.exp(self.A_i_log.float())
                # A_j = -torch.exp(self.A_j_log.float())
              

                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,
                    None,
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                
                # out_c = mamba_inner_fn_no_out_proj(
                #     xz_b,
                #     self.conv1d_c.weight,
                #     self.conv1d_c.bias,
                #     self.x_proj_c.weight,
                #     self.dt_proj_c.weight,
                #     A_c,
                #     None,
                #     None,
                #     self.D_c.float(),
                #     delta_bias=self.dt_proj_c.bias.float(),
                #     delta_softplus=True,
                # )
                # out_d = mamba_inner_fn_no_out_proj(
                #     xz_b.flip([-1]),
                #     self.conv1d_d.weight,
                #     self.conv1d_d.bias,
                #     self.x_proj_d.weight,
                #     self.dt_proj_d.weight,
                #     A_d,
                #     None,
                #     None,
                #     self.D_d.float(),
                #     delta_bias=self.dt_proj_d.bias.float(),
                #     delta_softplus=True,
                # )
               
                # out_i = mamba_inner_fn_no_out_proj(
                #     xz_e,
                #     self.conv1d_i.weight,
                #     self.conv1d_i.bias,
                #     self.x_proj_i.weight,
                #     self.dt_proj_i.weight,
                #     A_i,
                #     None,
                #     None,
                #     self.D_i.float(),
                #     delta_bias=self.dt_proj_i.bias.float(),
                #     delta_softplus=True,
                # )
                # out_j = mamba_inner_fn_no_out_proj(
                #     xz_e.flip([-1]),
                #     self.conv1d_j.weight,
                #     self.conv1d_j.bias,
                #     self.x_proj_j.weight,
                #     self.dt_proj_j.weight,
                #     A_j,
                #     None,
                #     None,
                #     self.D_j.float(),
                #     delta_bias=self.dt_proj_j.bias.float(),
                #     delta_softplus=True,
                # )
            

                # out_c = rearrange(out_c, "b d (w h c) -> b d (h w c)", w=self.input_w, h=self.input_h, c=self.input_c)
                # out_d = rearrange(out_d.flip([-1]), "b d (w h c) -> b d (h w c)", w=self.input_w, h=self.input_h, c=self.input_c)
                
                # out_i = rearrange(out_i, "b d (w c h) -> b d (h w c)", w=self.input_w, h=self.input_h, c=self.input_c)
                # out_j = rearrange(out_j.flip([-1]), "b d (w c h) -> b d (h w c)", w=self.input_w, h=self.input_h, c=self.input_c)



                # if self.use_norm:
                #     out = F.linear(self.norm(rearrange( out_c + out_d, "b d l -> b l d") / 2),
                #                     self.out_proj.weight, self.out_proj.bias)
                # else:
                #     out = F.linear(rearrange(out_c + out_d, "b d l -> b l d") / 2,
                #                     self.out_proj.weight, self.out_proj.bias)
                    
                if self.use_norm:
                    out = F.linear(self.norm(rearrange( out, "b d l -> b l d")), self.out_proj.weight, self.out_proj.bias)
                else:
                    out = F.linear(rearrange(out, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)   
                outs.append(rearrange(out, "b d l -> b l d"))
                # outs.append(rearrange(out_c, "b d l -> b l d"))
                # outs.append(rearrange(out_d, "b d l -> b l d"))
                # outs.append(rearrange(out_i, "b d l -> b l d"))
                # outs.append(rearrange(out_j, "b d l -> b l d"))

            elif self.bimamba_type == "v2":
                if len(alpha)==0:
                    x, z = xz.chunk(2, dim=1)
                    extra = extra_emb
                    # direction 1
                    if causal_conv1d_fn is None:
                        x = self.act(self.conv1d(x)[..., :seqlen])
                    else:
                        assert self.activation in ["silu", "swish"]
                        x = causal_conv1d_fn(
                            x=x,
                            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                            bias=self.conv1d.bias,
                            activation=self.activation,
                        )
                    x_dbl = self.x_proj(rearrange(extra, "b d l -> (b l) d"))  # (bl d)
                    dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                    dt = self.dt_proj.weight @ dt.t()
                    dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
                    B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    assert self.activation in ["silu", "swish"]
                    y = selective_scan_fn(
                        x,
                        dt,
                        A,
                        B,
                        C,
                        self.D.float(),
                        z=z,
                        delta_bias=self.dt_proj.bias.float(),
                        delta_softplus=True,
                        return_last_state=False,
                    )
                    if self.use_norm:
                        out = F.linear(self.norm(rearrange(y, "b d l -> b l d")), self.out_proj.weight, self.out_proj.bias)
                    else:
                        out = F.linear(rearrange(y, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                    outs.append(rearrange(y,"b d l -> b l d"))

                else:
                    thetas = Dir(torch.exp(alpha)).rsample(torch.Size([1])).squeeze(0)
                    x, z = xz.chunk(2, dim=1)
                    # x_b, z_b = xz.flip([-1]).chunk(2, dim=1)
                    extra = extra_emb
                    # extra_b = extra_emb.flip([-1])
                    xz_b = rearrange(xz, "b d (h w c) -> b d (w h c)", h=self.input_h, w=self.input_w, c=self.input_c)
                    x_c, z_c = xz_b.chunk(2, dim=1)
                    # x_d, z_d = xz_b.flip([-1]).chunk(2, dim=1)
                    extra_emb_b = rearrange(extra_emb, "b d (h w c) -> b d (w h c)", h=self.input_h, w=self.input_w, c=self.input_c)
                    extra_c = extra_emb_b
                    # extra_d = extra_emb_b.flip([-1])

                    xz_c = rearrange(xz, "b d (h w c) -> b d (c w h)", h=self.input_h, w=self.input_w, c=self.input_c)
                    x_e, z_e = xz_c.chunk(2, dim=1)
                    # x_f, z_f = xz_c.flip([-1]).chunk(2, dim=1)
                    extra_emb_c = rearrange(extra_emb, "b d (h w c) -> b d (c w h)", h=self.input_h, w=self.input_w, c=self.input_c)

                    extra_e = extra_emb_c
                    # extra_f = extra_emb_c.flip([-1])

                    xz_d = rearrange(xz, "b d (h w c) -> b d (h c w)", h=self.input_h, w=self.input_w, c=self.input_c)
                    x_g, z_g = xz_d.chunk(2, dim=1)
                    # x_h, z_h = xz_d.flip([-1]).chunk(2, dim=1)
                    extra_emb_d = rearrange(extra_emb, "b d (h w c) -> b d (h c w)", h=self.input_h, w=self.input_w, c=self.input_c)
                    extra_g = extra_emb_d
                    # extra_h = extra_emb_d.flip([-1])

                    xz_e = rearrange(xz, "b d (h w c) -> b d (w c h)", h=self.input_h, w=self.input_w, c=self.input_c)
                    x_i, z_i = xz_e.chunk(2, dim=1)
                    # x_j, z_j = xz_e.flip([-1]).chunk(2, dim=1)
                    extra_emb_e = rearrange(extra_emb, "b d (h w c) -> b d (w c h)", h=self.input_h, w=self.input_w, c=self.input_c)
                    extra_i = extra_emb_e
                    # extra_j = extra_emb_e.flip([-1])

                    xz_f = rearrange(xz, "b d (h w c) -> b d (c h w)", h=self.input_h, w=self.input_w, c=self.input_c)
                    x_k, z_k = xz_f.chunk(2, dim=1)
                    # x_l, z_l = xz_f.flip([-1]).chunk(2, dim=1)
                    extra_emb_f = rearrange(extra_emb, "b d (h w c) -> b d (c h w)", h=self.input_h, w=self.input_w, c=self.input_c)
                    extra_k = extra_emb_f
                    # extra_l = extra_emb_f.flip([-1])

                    # direction 1
                    if causal_conv1d_fn is None:
                        x = self.act(self.conv1d(x)[..., :seqlen])
                    else:
                        assert self.activation in ["silu", "swish"]
                        x = causal_conv1d_fn(
                            x=x,
                            weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                            bias=self.conv1d.bias,
                            activation=self.activation,
                        )
                    x_dbl = self.x_proj(rearrange(extra, "b d l -> (b l) d"))  # (bl d)
                    dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                    dt = self.dt_proj.weight @ dt.t()
                    dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
                    B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    assert self.activation in ["silu", "swish"]
                    y = selective_scan_fn(
                        x,
                        dt,
                        A,
                        B,
                        C,
                        self.D.float(),
                        z=z,
                        delta_bias=self.dt_proj.bias.float(),
                        delta_softplus=True,
                        return_last_state=False,
                    )

                    # direction 2
                    # A_b = -torch.exp(self.A_b_log.float())
                    # if causal_conv1d_fn is None:
                    #     x_b = self.act(self.conv1d_b(x_b)[..., :seqlen])
                    # else:
                    #     assert self.activation in ["silu", "swish"]
                    #     x_b = causal_conv1d_fn(
                    #         x=x_b,
                    #         weight=rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                    #         bias=self.conv1d_b.bias,
                    #         activation=self.activation,
                    #     )
                    # x_dbl_b = self.x_proj_b(rearrange(extra_b, "b d l -> (b l) d"))  # (bl d)
                    # dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                    # dt_b = self.dt_proj_b.weight @ dt_b.t()
                    # dt_b = rearrange(dt_b, "d (b l) -> b d l", l=seqlen)
                    # B_b = rearrange(B_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    # C_b = rearrange(C_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    # assert self.activation in ["silu", "swish"]
                    # y_b = selective_scan_fn(
                    #     x_b,
                    #     dt_b,
                    #     A_b,
                    #     B_b,
                    #     C_b,
                    #     self.D_b.float(),
                    #     z=z_b,
                    #     delta_bias=self.dt_proj_b.bias.float(),
                    #     delta_softplus=True,
                    #     return_last_state=False,
                    # )

                    # direction 3
                    A_c = -torch.exp(self.A_c_log.float())
                    if causal_conv1d_fn is None:
                        x_c = self.act(self.conv1d_c(x_c)[..., :seqlen])
                    else:
                        assert self.activation in ["silu", "swish"]
                        x_c = causal_conv1d_fn(
                            x=x_c,
                            weight=rearrange(self.conv1d_c.weight, "d 1 w -> d w"),
                            bias=self.conv1d_c.bias,
                            activation=self.activation,
                        )
                    x_dbl_c = self.x_proj_c(rearrange(extra_c, "b d l -> (b l) d"))  # (bl d)
                    dt_c, B_c, C_c = torch.split(x_dbl_c, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                    dt_c = self.dt_proj_c.weight @ dt_c.t()
                    dt_c = rearrange(dt_c, "d (b l) -> b d l", l=seqlen)
                    B_c = rearrange(B_c, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    C_c = rearrange(C_c, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    assert self.activation in ["silu", "swish"]
                    y_c = selective_scan_fn(
                        x_c,
                        dt_c,
                        A_c,
                        B_c,
                        C_c,
                        self.D_c.float(),
                        z=z_c,
                        delta_bias=self.dt_proj_c.bias.float(),
                        delta_softplus=True,
                        return_last_state=False,
                    )

                    # direction 4
                    # A_d = -torch.exp(self.A_d_log.float())
                    # if causal_conv1d_fn is None:
                    #     x_d = self.act(self.conv1d_d(x_d)[..., :seqlen])
                    # else:
                    #     assert self.activation in ["silu", "swish"]
                    #     x_d = causal_conv1d_fn(
                    #         x=x_d,
                    #         weight=rearrange(self.conv1d_d.weight, "d 1 w -> d w"),
                    #         bias=self.conv1d_d.bias,
                    #         activation=self.activation,
                    #     )
                    # x_dbl_d = self.x_proj_d(rearrange(extra_d, "b d l -> (b l) d"))  # (bl d)
                    # dt_d, B_d, C_d = torch.split(x_dbl_d, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                    # dt_d = self.dt_proj_d.weight @ dt_d.t()
                    # dt_d = rearrange(dt_d, "d (b l) -> b d l", l=seqlen)
                    # B_d = rearrange(B_d, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    # C_d = rearrange(C_d, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    # assert self.activation in ["silu", "swish"]
                    # y_d = selective_scan_fn(
                    #     x_d,
                    #     dt_d,
                    #     A_d,
                    #     B_d,
                    #     C_d,
                    #     self.D_d.float(),
                    #     z=z_d,
                    #     delta_bias=self.dt_proj_d.bias.float(),
                    #     delta_softplus=True,
                    #     return_last_state=False,
                    # )

                    # direction 5
                    A_e = -torch.exp(self.A_e_log.float())
                    if causal_conv1d_fn is None:
                        x_e = self.act(self.conv1d_e(x_e)[..., :seqlen])
                    else:
                        assert self.activation in ["silu", "swish"]
                        x_e = causal_conv1d_fn(
                            x=x_e,
                            weight=rearrange(self.conv1d_e.weight, "d 1 w -> d w"),
                            bias=self.conv1d_e.bias,
                            activation=self.activation,
                        )
                    x_dbl_e = self.x_proj_e(rearrange(extra_e, "b d l -> (b l) d"))  # (bl d)
                    dt_e, B_e, C_e = torch.split(x_dbl_e, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                    dt_e = self.dt_proj_e.weight @ dt_e.t()
                    dt_e = rearrange(dt_e, "d (b l) -> b d l", l=seqlen)
                    B_e = rearrange(B_e, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    C_e = rearrange(C_e, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    assert self.activation in ["silu", "swish"]
                    y_e = selective_scan_fn(
                        x_e,
                        dt_e,
                        A_e,
                        B_e,
                        C_e,
                        self.D_e.float(),
                        z=z_e,
                        delta_bias=self.dt_proj_e.bias.float(),
                        delta_softplus=True,
                        return_last_state=False,
                    )

                    # direction 6
                    # A_f = -torch.exp(self.A_f_log.float())
                    # if causal_conv1d_fn is None:
                    #     x_f = self.act(self.conv1d_f(x_f)[..., :seqlen])
                    # else:
                    #     assert self.activation in ["silu", "swish"]
                    #     x_f = causal_conv1d_fn(
                    #         x=x_f,
                    #         weight=rearrange(self.conv1d_f.weight, "d 1 w -> d w"),
                    #         bias=self.conv1d_f.bias,
                    #         activation=self.activation,
                    #     )
                    # x_dbl_f = self.x_proj_f(rearrange(extra_f, "b d l -> (b l) d"))  # (bl d)
                    # dt_f, B_f, C_f = torch.split(x_dbl_f, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                    # dt_f = self.dt_proj_f.weight @ dt_f.t()
                    # dt_f = rearrange(dt_f, "d (b l) -> b d l", l=seqlen)
                    # B_f = rearrange(B_f, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    # C_f = rearrange(C_f, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    # assert self.activation in ["silu", "swish"]
                    # y_f = selective_scan_fn(
                    #     x_f,
                    #     dt_f,
                    #     A_f,
                    #     B_f,
                    #     C_f,
                    #     self.D_f.float(),
                    #     z=z_f,
                    #     delta_bias=self.dt_proj_f.bias.float(),
                    #     delta_softplus=True,
                    #     return_last_state=False,
                    # )

                    # direction 7
                    A_g = -torch.exp(self.A_g_log.float())
                    if causal_conv1d_fn is None:
                        x_g = self.act(self.conv1d_g(x_g)[..., :seqlen])
                    else:
                        assert self.activation in ["silu", "swish"]
                        x_g = causal_conv1d_fn(
                            x=x_g,
                            weight=rearrange(self.conv1d_g.weight, "d 1 w -> d w"),
                            bias=self.conv1d_g.bias,
                            activation=self.activation,
                        )
                    x_dbl_g = self.x_proj_g(rearrange(extra_g, "b d l -> (b l) d"))  # (bl d)
                    dt_g, B_g, C_g = torch.split(x_dbl_g, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                    dt_g = self.dt_proj_g.weight @ dt_g.t()
                    dt_g = rearrange(dt_g, "d (b l) -> b d l", l=seqlen)
                    B_g = rearrange(B_g, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    C_g = rearrange(C_g, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    assert self.activation in ["silu", "swish"]
                    y_g = selective_scan_fn(
                        x_g,
                        dt_g,
                        A_g,
                        B_g,
                        C_g,
                        self.D_g.float(),
                        z=z_g,
                        delta_bias=self.dt_proj_g.bias.float(),
                        delta_softplus=True,
                        return_last_state=False,
                    )

                    # direction 8
                    # A_h = -torch.exp(self.A_h_log.float())
                    # if causal_conv1d_fn is None:
                    #     x_h = self.act(self.conv1d_h(x_h)[..., :seqlen])
                    # else:
                    #     assert self.activation in ["silu", "swish"]
                    #     x_h = causal_conv1d_fn(
                    #         x=x_h,
                    #         weight=rearrange(self.conv1d_h.weight, "d 1 w -> d w"),
                    #         bias=self.conv1d_h.bias,
                    #         activation=self.activation,
                    #     )
                    # x_dbl_h = self.x_proj_h(rearrange(extra_h, "b d l -> (b l) d"))  # (bl d)
                    # dt_h, B_h, C_h = torch.split(x_dbl_h, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                    # dt_h = self.dt_proj_h.weight @ dt_h.t()
                    # dt_h = rearrange(dt_h, "d (b l) -> b d l", l=seqlen)
                    # B_h = rearrange(B_h, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    # C_h = rearrange(C_h, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    # assert self.activation in ["silu", "swish"]
                    # y_h = selective_scan_fn(
                    #     x_h,
                    #     dt_h,
                    #     A_h,
                    #     B_h,
                    #     C_h,
                    #     self.D_h.float(),
                    #     z=z_h,
                    #     delta_bias=self.dt_proj_h.bias.float(),
                    #     delta_softplus=True,
                    #     return_last_state=False,
                    # )

                    # direction 9
                    A_i = -torch.exp(self.A_i_log.float())
                    if causal_conv1d_fn is None:
                        x_i = self.act(self.conv1d_i(x_i)[..., :seqlen])
                    else:
                        assert self.activation in ["silu", "swish"]
                        x_i = causal_conv1d_fn(
                            x=x_i,
                            weight=rearrange(self.conv1d_i.weight, "d 1 w -> d w"),
                            bias=self.conv1d_i.bias,
                            activation=self.activation,
                        )
                    x_dbl_i = self.x_proj_i(rearrange(extra_i, "b d l -> (b l) d"))  # (bl d)
                    dt_i, B_i, C_i = torch.split(x_dbl_i, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                    dt_i = self.dt_proj_i.weight @ dt_i.t()
                    dt_i = rearrange(dt_i, "d (b l) -> b d l", l=seqlen)
                    B_i = rearrange(B_i, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    C_i = rearrange(C_i, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    assert self.activation in ["silu", "swish"]
                    y_i = selective_scan_fn(
                        x_i,
                        dt_i,
                        A_i,
                        B_i,
                        C_i,
                        self.D_i.float(),
                        z=z_i,
                        delta_bias=self.dt_proj_i.bias.float(),
                        delta_softplus=True,
                        return_last_state=False,
                    )

                    # direction 10
                    # A_j = -torch.exp(self.A_j_log.float())
                    # if causal_conv1d_fn is None:
                    #     x_j = self.act(self.conv1d_j(x_j)[..., :seqlen])
                    # else:
                    #     assert self.activation in ["silu", "swish"]
                    #     x_j = causal_conv1d_fn(
                    #         x=x_j,
                    #         weight=rearrange(self.conv1d_j.weight, "d 1 w -> d w"),
                    #         bias=self.conv1d_j.bias,
                    #         activation=self.activation,
                    #     )
                    # x_dbl_j = self.x_proj_j(rearrange(extra_j, "b d l -> (b l) d"))  # (bl d)
                    # dt_j, B_j, C_j = torch.split(x_dbl_j, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                    # dt_j = self.dt_proj_j.weight @ dt_j.t()
                    # dt_j = rearrange(dt_j, "d (b l) -> b d l", l=seqlen)
                    # B_j = rearrange(B_j, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    # C_j = rearrange(C_j, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    # assert self.activation in ["silu", "swish"]
                    # y_j = selective_scan_fn(
                    #     x_j,
                    #     dt_j,
                    #     A_j,
                    #     B_j,
                    #     C_j,
                    #     self.D_j.float(),
                    #     z=z_j,
                    #     delta_bias=self.dt_proj_j.bias.float(),
                    #     delta_softplus=True,
                    #     return_last_state=False,
                    # )

                    # direction 11
                    A_k = -torch.exp(self.A_k_log.float())
                    if causal_conv1d_fn is None:
                        x_k = self.act(self.conv1d_k(x_k)[..., :seqlen])
                    else:
                        assert self.activation in ["silu", "swish"]
                        x_k = causal_conv1d_fn(
                            x=x_k,
                            weight=rearrange(self.conv1d_k.weight, "d 1 w -> d w"),
                            bias=self.conv1d_k.bias,
                            activation=self.activation,
                        )
                    x_dbl_k = self.x_proj_k(rearrange(extra_k, "b d l -> (b l) d"))  # (bl d)
                    dt_k, B_k, C_k = torch.split(x_dbl_k, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                    dt_k = self.dt_proj_k.weight @ dt_k.t()
                    dt_k = rearrange(dt_k, "d (b l) -> b d l", l=seqlen)
                    B_k = rearrange(B_k, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    C_k = rearrange(C_k, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    assert self.activation in ["silu", "swish"]
                    y_k = selective_scan_fn(
                        x_k,
                        dt_k,
                        A_k,
                        B_k,
                        C_k,
                        self.D_k.float(),
                        z=z_k,
                        delta_bias=self.dt_proj_k.bias.float(),
                        delta_softplus=True,
                        return_last_state=False,
                    )

                    # direction 12
                    # A_l = -torch.exp(self.A_l_log.float())
                    # if causal_conv1d_fn is None:
                    #     x_l = self.act(self.conv1d_l(x_l)[..., :seqlen])
                    # else:
                    #     assert self.activation in ["silu", "swish"]
                    #     x_l = causal_conv1d_fn(
                    #         x=x_l,
                    #         weight=rearrange(self.conv1d_l.weight, "d 1 w -> d w"),
                    #         bias=self.conv1d_l.bias,
                    #         activation=self.activation,
                    #     )
                    # x_dbl_l = self.x_proj_l(rearrange(extra_l, "b d l -> (b l) d"))  # (bl d)
                    # dt_l, B_l, C_l = torch.split(x_dbl_l, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                    # dt_l = self.dt_proj_l.weight @ dt_l.t()
                    # dt_l = rearrange(dt_l, "d (b l) -> b d l", l=seqlen)
                    # B_l = rearrange(B_l, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    # C_l = rearrange(C_l, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    # assert self.activation in ["silu", "swish"]
                    # y_l = selective_scan_fn(
                    #     x_l,
                    #     dt_l,
                    #     A_l,
                    #     B_l,
                    #     C_l,
                    #     self.D_l.float(),
                    #     z=z_l,
                    #     delta_bias=self.dt_proj_l.bias.float(),
                    #     delta_softplus=True,
                    #     return_last_state=False,
                    # )

                    # combination
                    y_c = rearrange(y_c, "b d (w h c) -> b d w h c", w=self.input_w, h=self.input_h, c=self.input_c)
                    y_c = rearrange(y_c.transpose(-3, -2).contiguous(),
                                    "b d h w c -> b d (h w c)",
                                    h=self.input_h,
                                    w=self.input_w,
                                    c=self.input_c)
                    # y_d = rearrange(y_d.flip([-1]), "b d (w h c) -> b d w h c", w=self.input_w, h=self.input_h, c=self.input_c)
                    # y_d = rearrange(y_d.transpose(-3, -2).contiguous(),
                    #                 "b d h w c -> b d (h w c)",
                    #                 h=self.input_h,
                    #                 w=self.input_w,
                    #                 c=self.input_c)

                    y_e = rearrange(y_e, "b d (c w h) -> b d (h w c)", w=self.input_w, h=self.input_h, c=self.input_c)
                    # y_f = rearrange(y_f.flip([-1]), "b d (c w h) -> b d (h w c)", w=self.input_w, h=self.input_h, c=self.input_c)

                    y_g = rearrange(y_g, "b d (h c w) -> b d (h w c)", w=self.input_w, h=self.input_h, c=self.input_c)
                    # y_h = rearrange(y_h.flip([-1]), "b d (h c w) -> b d (h w c)", w=self.input_w, h=self.input_h, c=self.input_c)

                    y_i = rearrange(y_i, "b d (w c h) -> b d (h w c)", w=self.input_w, h=self.input_h, c=self.input_c)
                    # y_j = rearrange(y_j.flip([-1]), "b d (w c h) -> b d (h w c)", w=self.input_w, h=self.input_h, c=self.input_c)

                    y_k = rearrange(y_k, "b d (c h w) -> b d (h w c)", w=self.input_w, h=self.input_h, c=self.input_c)
                    # y_l = rearrange(y_l.flip([-1]), "b d (c h w) -> b d (h w c)", w=self.input_w, h=self.input_h, c=self.input_c)

                    # if self.use_norm:
                    #     out = F.linear(self.norm(rearrange(y * thetas[0] + y_b.flip([-1]) * thetas[1] + y_c * thetas[2] + y_d * thetas[3]     + y_e * thetas[4] + y_f * thetas[5] + y_g * thetas[6] + y_h * thetas[7] + y_i * thetas[8] + y_j * thetas[9] + y_k *     thetas[10] + y_l * thetas[11], "b d l -> b l d")),
                    #                     self.out_proj.weight, self.out_proj.bias)
                    # else:
                    #     out = F.linear(rearrange(y * thetas[0] + y_b.flip([-1]) * thetas[1] + y_c * thetas[2] + y_d * thetas[3] + y_e *   thetas[4] + y_f * thetas[5] + y_g * thetas[6] + y_h * thetas[7] + y_i * thetas[8] + y_j * thetas[9] + y_k * thetas[10]    + y_l * thetas[11], "b d l -> b l d"),
                    #                     self.out_proj.weight, self.out_proj.bias)

                    if self.use_norm:
                        out = F.linear(self.norm(rearrange(y * thetas[0] + y_c * thetas[1] + y_e * thetas[2] + y_g * thetas[3] + y_i *  thetas[4] + y_k * thetas[5], "b d l -> b l d")),
                                        self.out_proj.weight, self.out_proj.bias)
                    else:
                        out = F.linear(rearrange(y * thetas[0] + y_c * thetas[1] + y_e * thetas[2] + y_g * thetas[3] + y_i * thetas[4] +y_k     * thetas[5], "b d l -> b l d"),
                                        self.out_proj.weight, self.out_proj.bias)

                    outs.append(rearrange(y,"b d l -> b l d"))
                    # outs.append(rearrange(y_b.flip([-1]), "b d l -> b l d"))
                    outs.append(rearrange(y_c,"b d l -> b l d"))
                    # outs.append(rearrange(y_d,"b d l -> b l d"))
                    outs.append(rearrange(y_e, "b d l -> b l d"))
                    # outs.append(rearrange(y_f, "b d l -> b l d"))
                    outs.append(rearrange(y_g, "b d l -> b l d"))
                    # outs.append(rearrange(y_h, "b d l -> b l d"))
                    outs.append(rearrange(y_i, "b d l -> b l d"))
                    # outs.append(rearrange(y_j, "b d l -> b l d"))
                    outs.append(rearrange(y_k, "b d l -> b l d"))
                    # outs.append(rearrange(y_l, "b d l -> b l d"))
            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )

        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)

        if self.init_layer_scale is not None:
                out = out * self.gamma
        return out, outs

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)




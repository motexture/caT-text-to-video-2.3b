# Adapted from https://github.com/lucidrains/make-a-video-pytorch/blob/main/make_a_video_pytorch/make_a_video.py
# Borrowed from https://github.com/xuduo35/MakeLongVideo/blob/main/makelongvideo/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass
from typing import Optional
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.models.attention import BasicTransformerBlock
from torch import nn
 
class TemporalConvLayer(nn.Module):
    """
    Temporal convolutional layer that can be used for video (sequence of images) input Code mostly copied from:
    https://github.com/modelscope/modelscope/blob/1509fdb973e5871f37148a4b5e5964cafd43e64d/modelscope/models/multi_modal/video_synthesis/unet_sd.py#L1016

    Parameters:
        in_dim (`int`): Number of input channels.
        out_dim (`int`): Number of output channels.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        out_dim = out_dim or in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.conv1 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, in_dim),
            nn.SiLU(),
            nn.Conv3d(in_dim, out_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv3 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )
        self.conv4 = nn.Sequential(
            nn.GroupNorm(norm_num_groups, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv3d(out_dim, in_dim, (3, 1, 1), padding=(1, 0, 0)),
        )

        nn.init.zeros_(self.conv4[-1].weight)
        nn.init.zeros_(self.conv4[-1].bias)

    def forward(self, hidden_states: torch.Tensor, num_frames: int = 1) -> torch.Tensor:
        hidden_states = (
            hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
        )

        identity = hidden_states

        hidden_states = self.conv1(hidden_states)
        hidden_states = self.conv2(hidden_states)
        hidden_states = self.conv3(hidden_states)
        hidden_states = self.conv4(hidden_states)
        
        hidden_states = identity + hidden_states

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            (hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:]
        )
        return hidden_states
    
class PositionalEncodings(torch.nn.Module):
    def __init__(self):
        super(PositionalEncodings, self).__init__()

    def _get_embeddings(self, seq_len, d_model):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        normalized_position = (position - (seq_len - 1) / 2) / ((seq_len - 1) / 2)

        weights = torch.cos(math.pi * normalized_position)

        for i in range(d_model):
            pe[:, i] = weights.squeeze()
        
        return pe.unsqueeze(0)

    def forward(self, x):
        _, seq_len, d_model = x.shape
        pe = self._get_embeddings(seq_len, d_model)
        return pe.to(x.device)
    
@dataclass
class caTConditioningTransformerOutput(BaseOutput):
    sample: torch.FloatTensor

class caTConditioningTransformerModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        cross_attention_dim: int = 1280,
        num_layers: int = 2,
        only_cross_attention: bool = True,
        dropout: float = 0.0,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        norm_elementwise_affine: bool = True,
    ):
        super().__init__()

        if num_layers == 1:
            raise ValueError("The number of transformer layers (`num_layers`) must be greater than 1.")

        self.encoder_conv_in = nn.Conv3d(4, cross_attention_dim, kernel_size=(1, 1, 1))

        self.hidden_ln = nn.LayerNorm(in_channels)
        self.hidden_proj_in = nn.Linear(in_channels, cross_attention_dim)

        self.positional_encoding = PositionalEncodings()

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    cross_attention_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    norm_elementwise_affine=norm_elementwise_affine
                )
                for _ in range(num_layers)
            ]
        )

        self.hidden_proj_out = nn.Linear(cross_attention_dim, in_channels)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        num_frames,
        return_dict: bool = True,       
    ):
        if encoder_hidden_states is None:
            if not return_dict:
                return (hidden_states,)

            return caTConditioningTransformerOutput(sample=hidden_states)
            
        if hidden_states.size(2) <= 1:
            if not return_dict:
                return (hidden_states,)

            return caTConditioningTransformerOutput(sample=hidden_states)
        
        hidden_states = (
            hidden_states[None, :].reshape((-1, num_frames) + hidden_states.shape[1:]).permute(0, 2, 1, 3, 4)
        )
        
        h_b, h_c, h_f, h_h, h_w = hidden_states.shape

        residual = hidden_states

        encoder_hidden_states = torch.nn.functional.interpolate(encoder_hidden_states, size=(encoder_hidden_states.shape[2], h_h, h_w), mode='trilinear', align_corners=False)
        if encoder_hidden_states.shape[0] < h_b:
            encoder_hidden_states = encoder_hidden_states.repeat_interleave(repeats=h_b, dim=0)
        encoder_hidden_states = self.encoder_conv_in(encoder_hidden_states)

        e_b, e_c, e_f, e_h, e_w = encoder_hidden_states.shape

        encoder_hidden_states = encoder_hidden_states.permute(0, 3, 4, 2, 1)
        encoder_hidden_states = encoder_hidden_states.reshape(e_b * e_h * e_w, e_f, e_c)

        hidden_states = hidden_states.permute(0, 3, 4, 2, 1)
        hidden_states = hidden_states.reshape(h_b * h_h * h_w, h_f, h_c)

        hidden_states = self.hidden_ln(hidden_states)
        hidden_states = self.hidden_proj_in(hidden_states)

        encoder_hidden_states = encoder_hidden_states + self.positional_encoding(
            torch.cat((encoder_hidden_states, hidden_states), dim=1)
        )[:, :e_f, :]

        hidden_states = hidden_states + self.positional_encoding(
            torch.cat((encoder_hidden_states, hidden_states), dim=1)
        )[:, e_f:, :]

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states
            )

        hidden_states = self.hidden_proj_out(hidden_states)

        hidden_states = hidden_states.view(h_b, h_h, h_w, h_f, h_c).contiguous()
        hidden_states = hidden_states.permute(0, 4, 3, 1, 2)

        hidden_states += residual

        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            (hidden_states.shape[0] * hidden_states.shape[2], -1) + hidden_states.shape[3:]
        )
        
        output = hidden_states

        if not return_dict:
            return (output,)

        return caTConditioningTransformerOutput(sample=output)
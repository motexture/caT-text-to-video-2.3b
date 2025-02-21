# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_3d_blocks.py

import torch
from torch import nn
from .caT_modules import caTConditioningTransformerModel
from diffusers.models.resnet import Downsample2D, ResnetBlock2D, TemporalConvLayer, Upsample2D
from diffusers.models.transformers.transformer_2d  import Transformer2DModel
from diffusers.models.transformers.transformer_temporal  import TransformerTemporalModel

def get_down_block(
    down_block_type,
    num_layers,
    transformer_layers,
    conditioning_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attention_head_dim,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
):
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "CrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock3D")
        return CrossAttnDownBlock3D(
            num_layers=num_layers,
            transformer_layers=transformer_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            conditioning_layers=conditioning_layers,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    raise ValueError(f"{down_block_type} does not exist.")

def get_up_block(
    up_block_type,
    num_layers,
    transformer_layers,
    conditioning_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attention_head_dim,
    resnet_groups=None,
    cross_attention_dim=None,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
):
    if up_block_type == "UpBlock3D":
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "CrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock3D")
        return CrossAttnUpBlock3D(
            num_layers=num_layers,
            transformer_layers=transformer_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            conditioning_layers=conditioning_layers,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    raise ValueError(f"{up_block_type} does not exist.")

class UNetMidBlock3DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers: int = 1,
        conditioning_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim: int = 64,
        output_scale_factor: float = 1.0,
        cross_attention_dim: int = 1024,
        upcast_attention: bool = False
    ):
        super().__init__()

        self.has_cross_attention = True
        self.attention_head_dim = attention_head_dim
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        temp_convs = [
            TemporalConvLayer(
                in_channels,
                in_channels,
                dropout=0.1
            )
        ]
        
        cond_attentions = []
        temp_attentions = []
        attentions = []

        for _ in range(num_layers):
            attentions.append(
                Transformer2DModel(
                    in_channels // attention_head_dim,
                    attention_head_dim,
                    in_channels=in_channels,
                    num_layers=transformer_layers,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    upcast_attention=upcast_attention,
                    use_linear_projection=True
                )
            )
            cond_attentions.append(
                caTConditioningTransformerModel(
                    in_channels // attention_head_dim,
                    attention_head_dim,
                    in_channels=in_channels,
                    num_layers=conditioning_layers
                )
            )
            temp_attentions.append(
                TransformerTemporalModel(
                    in_channels // attention_head_dim,
                    attention_head_dim,
                    in_channels=in_channels,
                    norm_num_groups=resnet_groups
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    in_channels,
                    in_channels,
                    dropout=0.1
                )
            )
        
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.cond_attentions = nn.ModuleList(cond_attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)
        
    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        num_frames = 16,
        conditioning_hidden_states=None
    ):
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.temp_convs[0](hidden_states, num_frames=num_frames)
            
        for attn, cond_attn, temp_attn, resnet, temp_conv in zip(self.attentions, self.cond_attentions, self.temp_attentions, self.resnets[1:], self.temp_convs[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample
            hidden_states = cond_attn(hidden_states, conditioning_hidden_states, num_frames=num_frames).sample
            hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        return hidden_states

class CrossAttnDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers: int = 1,
        conditioning_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 1536,
        output_scale_factor: float = 1.0,
        downsample_padding: int = 1,
        add_downsample:bool = True,
        only_cross_attention: bool = False,
        upcast_attention: bool = False
    ):
        super().__init__()

        resnets = []
        temp_convs = []
        attentions = []
        cond_attentions = []
        temp_attentions = []

        self.has_cross_attention = True
        self.attention_head_dim = attention_head_dim

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1
                )
            )
            attentions.append(
                Transformer2DModel(
                    out_channels // attention_head_dim,
                    attention_head_dim,
                    in_channels=out_channels,
                    num_layers=transformer_layers,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    use_linear_projection=True
                )
            )
            cond_attentions.append(
                caTConditioningTransformerModel(
                    out_channels // attention_head_dim,
                    attention_head_dim,
                    in_channels=out_channels,
                    num_layers=conditioning_layers
                )
            )
            temp_attentions.append(
                TransformerTemporalModel(
                    out_channels // attention_head_dim,
                    attention_head_dim,
                    in_channels=out_channels,
                    norm_num_groups=resnet_groups
                )
            )
            
        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.cond_attentions = nn.ModuleList(cond_attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(
        self,
        hidden_states,
        temb=None,
        encoder_hidden_states=None,
        num_frames = 16,
        conditioning_hidden_states=None
    ):
        output_states = ()

        for resnet, temp_conv, attn, cond_attn, temp_attn in zip(self.resnets, self.temp_convs, self.attentions, self.cond_attentions, self.temp_attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample
            hidden_states = cond_attn(hidden_states, conditioning_hidden_states, num_frames=num_frames).sample
            hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states

class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample:bool = True,
        downsample_padding: int = 1,
    ):
        super().__init__()

        resnets = []
        temp_convs = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states, temb=None, num_frames=16):
        output_states = ()

        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states

class CrossAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        transformer_layers: int = 1,   
        conditioning_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 2048,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        only_cross_attention: bool = False,
        upcast_attention: bool = False
    ):
        super().__init__()

        resnets = []
        temp_convs = []
        attentions = []
        cond_attentions = []
        temp_attentions = []

        self.has_cross_attention = True
        self.attention_head_dim = attention_head_dim

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1
                )
            )
            attentions.append(
                Transformer2DModel(
                    out_channels // attention_head_dim,
                    attention_head_dim,
                    in_channels=out_channels,
                    num_layers=transformer_layers,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    use_linear_projection=True
                )
            )
            cond_attentions.append(
                caTConditioningTransformerModel(
                    out_channels // attention_head_dim,
                    attention_head_dim,
                    in_channels=out_channels,
                    num_layers=conditioning_layers
                )
            )
            temp_attentions.append(
                TransformerTemporalModel(
                    out_channels // attention_head_dim,
                    attention_head_dim,
                    in_channels=out_channels,
                    norm_num_groups=resnet_groups
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)
        self.attentions = nn.ModuleList(attentions)
        self.cond_attentions = nn.ModuleList(cond_attentions)
        self.temp_attentions = nn.ModuleList(temp_attentions)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        num_frames=16,
        conditioning_hidden_states=None,
        upsample_size=None,
    ):
        for resnet, temp_conv, attn, cond_attn, temp_attn in zip(self.resnets, self.temp_convs, self.attentions, self.cond_attentions, self.temp_attentions):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample
            hidden_states = cond_attn(hidden_states, conditioning_hidden_states, num_frames=num_frames).sample
            hidden_states = temp_attn(hidden_states, num_frames=num_frames).sample

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states

class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True
    ):
        super().__init__()
        resnets = []
        temp_convs = []
        
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            temp_convs.append(
                TemporalConvLayer(
                    out_channels,
                    out_channels,
                    dropout=0.1
                )
            )

        self.resnets = nn.ModuleList(resnets)
        self.temp_convs = nn.ModuleList(temp_convs)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, num_frames=16):
        for resnet, temp_conv in zip(self.resnets, self.temp_convs):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = temp_conv(hidden_states, num_frames=num_frames)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states
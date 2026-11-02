"""
Mixed adaptation from:

    Liu et al. 2022, A ConvNet for the 2020s.
    Source: https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Ronneberger et al., 2015. Convolutional Networks for Biomedical Image Segmentation.

If you use this implementation, please cite original work above.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    """
    Implementation of the ConvNext block for the ROMS pacts30 dataset.
    It is assumed, that the data has the shape (Batch, Channel, Width, Height).
    Drop path is ommited since it was not experimented with.
    """

    def __init__(
            self, 
            dim, 
            layer_scale_init_value=1.0e-6
        ):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1.0e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True,
        ) if layer_scale_init_value > 0 else None
    
    def forward(self, x):
        skip = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 3, 1, 2)
        x = skip + x
        return x

class Stage(nn.Module):
    """
    Implementation of a ConvNeXt Stage for the ROMS pactcs30 dataset.
    skip_projection is needed because the bottleneck stage increases the
    feature dimensions by 2.

    Args:
        dim_in (int): Number of input channels.
        dim_out (int): Number of output channels.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        mode (str): Down, Up, Neck. Default: "down"
        skip_projection (Boolean): Whether to reduce feature numbers by half.
    """

    def __init__(
            self,
            dim_in,
            dim_out,
            depth=2,
            mode="down",
            layer_scale_init_value=1.0e-6,
            skip_projection=False
    ):
        super().__init__()

        if skip_projection:
            self.preprocess = nn.Conv2d(2 * dim_in, dim_in, kernel_size=1)
        else:
            self.preprocess = nn.Identity()

        match mode:
            case "down":
                self.resample = nn.Sequential(
                    LayerNorm(dim_in, eps=1.0e-6),
                    nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2)
                )
            case "up":
                self.resample = nn.Sequential(
                    LayerNorm(dim_in, eps=1.0e-6),
                    nn.ConvTranspose2d(dim_in, dim_out, kernel_size=2, stride=2)
                )
            case "neck":
                self.resample = nn.Identity()
        
        self.blocks = nn.ModuleList([
            Block(dim_in, layer_scale_init_value) for _ in range(depth)
        ])

    def forward(self, x):
        x = self.preprocess(x)

        for b in self.blocks:
            x = b(x)

        x = self.resample(x)
        return x
    
class CNUnet(nn.Module):
    """
    Implementation of a ConvNeXt model for the ROMS pactsc30 dataset.

    Args:
        dim_in (int): Number of input channels.
        dim_out (int): Number of output channels.
        stages (int): number of stages.
        blocks_per_stage (int): Number of blocks per stage.
        blocks_at_neck (int): Number of blocks at the bottleneck.
        init_features (int): Number of Channels before UNet.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        stages=4,
        blocks_per_stage=2,
        blocks_at_neck=1,
        init_features=42,
    ):
        super().__init__()

        encoder_dims = [init_features * 2**i for i in range(stages + 1)]
        decoder_dims = [init_features * 2**i for i in range(stages, -1, -1)]

        encoder = []
        decoder = []

        self.in_proj = nn.Conv2d(dim_in, init_features, kernel_size=3, padding=1)
        self.out_proj = nn.Conv2d(2 * init_features, dim_out, kernel_size=3, padding=1)

        for i in range(stages):
            encoder.append(
                Stage(
                    encoder_dims[i],
                    encoder_dims[i+1],
                    depth=blocks_per_stage,
                    mode="down"
                )
            )

            decoder.append(
                Stage(
                    decoder_dims[i],
                    decoder_dims[i+1],
                    depth=blocks_per_stage,
                    mode="up",
                    skip_projection=(i != 0)
                )
            )

        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.neck = Stage(
            encoder_dims[-1],
            encoder_dims[-1],
            depth=blocks_at_neck,
            mode="neck"
        )

    def forward(self, x):
        x = self.in_proj(x)
        skips = []

        for enc in self.encoder:
            skips.append(x)
            x = enc(x)

        x = self.neck(x)

        for i, dec in enumerate(self.decoder):
            x = dec(x)
            x = torch.cat([skips[-(i+1)], x], dim=1)
        
        x = self.out_proj(x)
        return x
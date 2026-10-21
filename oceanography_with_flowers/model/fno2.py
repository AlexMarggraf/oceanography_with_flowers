import torch
import torch.nn as nn
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.channel_mlp import ChannelMLP
from neuralop.layers.skip_connections import skip_connection

class FiLM(nn.Module):
    def __init__(self, num_channels, param_dim=1):
        super().__init__()
        
        self.weight = nn.Linear(param_dim, num_channels)
        self.bias = nn.Linear(param_dim, num_channels)

    def forward(self, x, param):
        param = param.view(-1, 1)

        weight = self.weight(param)
        bias = self.bias(param)

        while weight.dim() < x.dim():
            weight = weight.unsqueeze(-1)
            bias = bias.unsqueeze(-1)

        return weight * x + bias

class FNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes, param_dim=1, use_film=True, use_norm = True):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_film = use_film
        self.use_norm = use_norm

        self.skip = skip_connection(
            in_features=in_channels,
            out_features=out_channels
        )

        self.spectral_conv = SpectralConv(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=modes
        )

        if use_film:
            self.film = FiLM(
                num_channels=out_channels,
                param_dim=param_dim
            )
        
        if use_norm:
            self.norm = nn.InstanceNorm2d(num_features=out_channels)

        self.activation = nn.GELU()
    
    def forward(self, x, param=None):
        x_spec = self.spectral_conv(x)
        x_skip = self.skip(x)

        out = x_spec + x_skip

        if self.use_norm:
            out = self.norm(out)

        if self.use_film and param is not None:
            out = self.film(out, param)

        out = self.activation(out)
        return out

class Pactcs30FNO(nn.Module):
    def __init__(self, start_channels, end_channels, hidden_channels, modes, layers, param_dim=1, use_film=True, use_norm=True):
        super().__init__()
        
        self.in_channels = start_channels
        self.out_channels = end_channels
        self.hidden_channels = hidden_channels
        self.layers = layers
        self.use_film = use_film
        self.param_dim = param_dim

        self.lift = ChannelMLP(
            in_channels=start_channels,
            out_channels=hidden_channels,
            hidden_channels=2*hidden_channels,
            n_layers=2
        )

        self.blocks = nn.ModuleList([
            FNOBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                modes=modes,
                param_dim=param_dim,
                use_film=use_film,
                use_norm=use_norm
            )
            for _ in range(layers)
        ])

        self.project = ChannelMLP(
            in_channels=hidden_channels,
            out_channels=end_channels,
            hidden_channels=2*hidden_channels,
            n_layers=2
        )

    def forward(self, x, param=None):
        x = self.lift(x)

        for b in self.blocks:
            x = b(x, param)

        x = self.project(x)
        return x

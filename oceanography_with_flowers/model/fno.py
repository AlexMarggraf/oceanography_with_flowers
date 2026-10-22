from torch import nn
from neuralop.layers.fno_block import FNOBlocks

class Pactcs30FNO(nn.Module):
    """
    FNO architecture for predicting pactcs30 data.

    Args:
        start_channels (int): Amount of channels, i.e. data variables plus encodings
        end_channels (int): Amount of channels in the label, i.e. data variables
        hidden_channels (int): Amount of hidden channels after lifting layer
        modes (int): Amount of modes to keep during spectral convolutions
        layers (int): Depth of the network
    """

    def __init__(self, start_channels, end_channels, hidden_channels, modes, layers):
        super().__init__()
        self.fno = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=(modes, modes),
            n_layers=layers,
            norm="instance_norm"
        )
        self.lift = nn.Conv2d(start_channels, hidden_channels, 1)
        self.proj = nn.Conv2d(hidden_channels, end_channels, 1)

    def forward(self, x):
        x = self.lift(x)
        x = self.fno(x)
        x = self.proj(x)
        return x

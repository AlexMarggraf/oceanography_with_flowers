from torch import nn
from neuralop.layers.fno_block import FNOBlocks

class Pactcs30Fno(nn.Module):
    """
    Represents the FNO used to predict pactcs30 data.
    
    
    """

    def __init__(self):
        super().__init__()
        self.fno = FNOBlocks(
            in_channels=16,
            out_channels=16,
            n_modes=(16, 16),
            n_layers=4,
        )
        self.lift = nn.Conv2d(10, 16, 1)
        self.proj = nn.Conv2d(16, 8, 1)

    def forward(self, x):
        x = self.lift(x)
        x = self.fno(x)
        x = self.proj(x)
        return x

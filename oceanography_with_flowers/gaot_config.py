"""
This implementation was obtained from:

    Wen, S., Kumbhat, A., Lingsch, L., Mousavi, S., Zhao, Y., Chandrashekar, P., & Mishra, S. (2025). 
    Geometry Aware Operator Transformer as an Efficient and Accurate Neural Surrogate for PDEs on Arbitrary Domains.
    Source: https://github.com/camlab-ethz/GAOT/tree/main/src/model/gaot.py

If you want to use this implementation, please cite the original work above.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Union, Tuple

@dataclass
class MagnoConfig:
    """Simplified MAGNO Configuration with reduced parameters"""
    
    # --- Core Parameters ---
    coord_dim: int = 2                                      # Coordinate dimension (2 for 2D, 3 for 3D)
    radius: float = 0.033                                   # Radius for neighbor search
    hidden_size: int = 64                                   # Base hidden size for all MLPs
    mlp_layers: int = 3                                     # Number of MLP layers (consistent across all MLPs)
    lifting_channels: int = 32                              # Number of channels after Encoder

    
    # --- Multi-scale ---
    scales: List[float] = field(default_factory=lambda: [1.0])  # Multi-scale factors
    use_scale_weights: bool = False                         # Whether to use learnable scale weights
    
    # --- Attention and Embedding ---
    use_attention: bool = True                              # Enable attention mechanism  
    attention_type: str= 'cosine'                           # Attention type, support ['cosine', 'dot_product']
    use_geoembed: bool = True                              # Enable geometric embedding
    embedding_method: str = 'statistical'                   # Geometric embedding method, support ['statistical', 'pointnet']
    pooling: str = 'max'                                    # Pooling for pointnet embedding, support ['max', 'mean', 'sum']
    
    # --- Transform and Sampling ---
    transform_type: str = 'linear'                          # Transform type for both encoder and decoder, support ['linear', 'nonlinear']
    sampling_strategy: Optional[str] = None                 # Edge sampling strategy, support ['max_neighbors', 'ratio']
    max_neighbors: Optional[int] = None                     # Max neighbors for sampling
    sample_ratio: Optional[float] = None                    # Sample ratio for edge drop
    
    # --- Advanced ---
    node_embedding: bool = False                            # Use positional node embedding
    neighbor_search_method: str = 'auto'                   # Neighbor search method, support ['open3d', 'torch_cluster', 'grid', 'chunked', 'native']
    use_torch_scatter: bool = True                          # Use torch_scatter backend
    neighbor_strategy: str = 'radius'                      # Neighbor search strategy, support ['radius', 'knn']
    precompute_edges: bool = False                          # Whether edges are precomputed
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.coord_dim not in [2, 3]:
            raise ValueError(f"coord_dim must be 2 or 3, got {self.coord_dim}")
        if self.sampling_strategy == 'ratio' and (self.sample_ratio is None or not 0 < self.sample_ratio <= 1):
            raise ValueError("sample_ratio must be in (0, 1] when using 'ratio' sampling")
        if self.sampling_strategy == 'max_neighbors' and (self.max_neighbors is None or self.max_neighbors <= 0):
            raise ValueError("max_neighbors must be > 0 when using 'max_neighbors' sampling")

@dataclass
class AttentionConfig:
    num_heads: int = 8                      # Number of attention heads (for multi-head attention)
    num_kv_heads: int = 8                   # Number of attention heads for Key and Value (Grouped Query Attention)
    use_conditional_norm: bool = False      # Whether to use time conditional normalization
    cond_norm_hidden_size: int = 4          # Hidden size for the time conditional normalization
    atten_dropout: float = 0.0              # Dropout probability in the attention module

@dataclass
class TransformerConfig:
    patch_size: int = 8                              # Size of the patches for the structured latent tokens
    hidden_size: int = 256                           # Hidden size of the transformer
    use_attn_norm: bool = True                       # Whether to use normalization in the attention module
    use_ffn_norm: bool = True                        # Whether to use normalization in the feedforward network
    norm_eps: float = 1e-6                           # Epsilon value for layer normalization
    num_layers: int = 3                              # Number of transformer blocks
    positional_embedding: str = 'absolute'           # Positional embedding type, supports ['absolute', 'rope']
    use_long_range_skip: bool = True                 # Set it to True for UViT processor
    ffn_multiplier: int = 4                          # FFN hidden size multiplier (ffn_hidden = hidden_size * ffn_multiplier)
    attn_config: AttentionConfig = field(default_factory=AttentionConfig)   # Configuration for the attention sub-module

@dataclass
class ModelArgsConfig:
    """Model arguments configuration."""
    magno: MagnoConfig = field(default_factory=MagnoConfig)                     # MAGNO encoder/decoder config
    transformer: TransformerConfig = field(default_factory=TransformerConfig)   # Transformer processor config


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "gaot"                                             # Model name: "gaot"
    use_conditional_norm: bool = False                             # Time-conditional normalization
    latent_tokens_size: Tuple[int, int] = (64, 64)                 # Latent token dimensions (H,W) or (H,W,D)
    dim_in: int = 13                                               # Input dimensions
    dim_out: int = 10                                              # Output dimensions
    args: ModelArgsConfig = field(default_factory=ModelArgsConfig) # Model component configurations

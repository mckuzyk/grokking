from dataclasses import dataclass


@dataclass
class TransformerConfig:
    # Prime number to mod by
    P: int = 113

    # Dimension of the embedded vector (the residual stream)
    d_model: int = 128

    # Attention settings
    n_blocks: int = 1
    n_heads: int = 4

    # Dimension of hidden layer of MLP
    d_mlp: int = 512

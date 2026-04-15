from pathlib import Path
from dataclasses import dataclass, asdict
from abc import ABC
import json
from datetime import datetime


@dataclass
class Config(ABC):
    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r") as f:
            kwargs = json.load(f)
        return cls(**kwargs)


@dataclass
class TransformerConfig(Config):
    # Prime number to mod by
    P: int = 113

    # Dimension of the embedded vector (the residual stream)
    d_model: int = 128

    # Attention settings
    n_blocks: int = 1
    n_heads: int = 4

    # Dimension of hidden layer of MLP
    d_mlp: int = 512


@dataclass
class TrainConfig(Config):
    save_path: str | None = None
    random_seed: int = 42
    epochs: int = 10_000
    checkpoint_every: int = 100
    warmup_steps: int = 10

    # Train/Test data
    train_frac: float = 0.3

    # AdamW arguments
    lr: float = 1e-3  # Specified in paper
    weight_decay: float = 1.0  # Specified in paper
    beta1: float = 0.9  # Default in torch
    beta2: float = 0.98  # Used by Nanda

    def __post_init__(self):
        if self.save_path is None:
            dir_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
            self.save_path = str(Path("runs") / dir_name)

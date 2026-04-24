from pathlib import Path
from scripts.analysis import Results
from grokking.data import ModPDataset, train_test_split
from grokking.config import TrainConfig


path = Path("runs") / "baseline_weight_init_seed_137"
r = Results(path)
train_config = TrainConfig.load(path / "training_config.json")

total = sum(p.numel() for p in r.model.parameters())
trainable = sum(p.numel() for p in r.model.parameters() if p.requires_grad)

print(f"Total parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")

dset = ModPDataset(r.cfg.P)
_, train_labels, _, _ = train_test_split(dset, train_frac=train_config.train_frac)

print(f"Total training samples: {len(train_labels)}")

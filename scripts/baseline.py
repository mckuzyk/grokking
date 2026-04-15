from grokking import model, config
from grokking.train import train
import torch
import matplotlib.pyplot as plt


all_metrics = []
for seed in [0, 137, 28]:
    train_cfg = config.TrainConfig(
        save_path=f"runs/baseline_weight_init_seed_{seed}",
        epochs=40_000,
        random_seed=seed,
    )
    torch.manual_seed(train_cfg.random_seed)

    model_cfg = config.TransformerConfig(P=113)
    trans = model.Transformer(model_cfg)

    trans, metrics = train(trans, train_cfg)
    all_metrics.append(metrics)

plt.figure()
for metrics in all_metrics:
    plt.plot(metrics["train_loss"], color="blue")
    plt.plot(metrics["test_loss"], color="red")
plt.yscale("log")

plt.figure()
for metrics in all_metrics:
    plt.plot(metrics["train_acc"], color="blue")
    plt.plot(metrics["test_acc"], color="red")

plt.show()

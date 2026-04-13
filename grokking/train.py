from pathlib import Path
import json
import torch
from grokking.config import TrainConfig
from grokking.data import ModPDataset, train_test_split


def train(model, cfg: TrainConfig):
    save_path = Path(cfg.save_path)
    save_path.mkdir(parents=True)
    model.config.save(save_path / "model_config.json")
    cfg.save(save_path / "training_config.json")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    dset = ModPDataset(P=model.config.P)
    train_inputs, train_labels, test_inputs, test_labels = train_test_split(
        dset, train_frac=cfg.train_frac
    )
    metrics = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }
    for epoch in range(cfg.epochs):
        optimizer.zero_grad()

        # Test
        with torch.no_grad():
            logits = model.forward(test_inputs)
            t_loss = loss_fn(logits, test_labels).item()
            metrics["test_loss"].append(t_loss)

            test_acc = (logits.argmax(dim=-1) == test_labels).float().mean().item()
            metrics["test_acc"].append(test_acc)

        # Train
        logits = model.forward(train_inputs)
        loss = loss_fn(logits, train_labels)
        loss.backward()
        metrics["train_loss"].append(loss.item())

        with torch.no_grad():
            train_acc = (logits.argmax(dim=-1) == train_labels).float().mean().item()
        metrics["train_acc"].append(train_acc)

        optimizer.step()

        if epoch % cfg.checkpoint_every == 0:
            torch.save(model.state_dict(), save_path / f"model_state_dict_{epoch}.pt")

        if epoch % 100 == 0:
            print(
                f"Epoch: {epoch} | "
                f"Loss (Train): {loss.item()} | "
                f"Loss (Test): {t_loss} | "
                f"Acc (Train): {train_acc} | "
                f"Acc (Test): {test_acc} | "
            )
    # Save final state if needed
    if epoch % cfg.checkpoint_every != 0:
        torch.save(model.state_dict(), save_path / f"model_state_dict_{epoch}.pt")

    with open(save_path / "metrics.json", "w") as f:
        json.dump(metrics, f)

    return model, metrics


if __name__ == "__main__":
    from grokking import model, config
    import matplotlib.pyplot as plt

    train_cfg = config.TrainConfig(save_path="runs/full_test_40k", epochs=40_000)
    torch.manual_seed(train_cfg.random_seed)

    model_cfg = config.TransformerConfig(P=113)
    trans = model.Transformer(model_cfg)

    trans, metrics = train(trans, train_cfg)

    plt.figure()
    plt.plot(metrics["train_loss"])
    plt.plot(metrics["test_loss"])
    plt.yscale("log")

    plt.figure()
    plt.plot(metrics["train_acc"])
    plt.plot(metrics["test_acc"])

    plt.show()

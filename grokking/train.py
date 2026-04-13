from pathlib import Path
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
    train_loss = []
    test_loss = []
    for epoch in range(cfg.epochs):
        optimizer.zero_grad()

        # Test
        with torch.no_grad():
            logits = model.forward(test_inputs)
            t_loss = loss_fn(logits, test_labels).item()
            test_loss.append(t_loss)

        # Train
        logits = model.forward(train_inputs)
        loss = loss_fn(logits, train_labels)
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()

        if epoch % cfg.checkpoint_every == 0:
            torch.save(model.state_dict(), save_path / f"model_state_dict_{epoch}.pt")

        if epoch % 100 == 0:
            print(
                f"Epoch: {epoch} | "
                f"Loss (Train): {loss.item()} | "
                f"Loss (Test): {t_loss} | "
            )
    # Save final state if needed
    if epoch % cfg.checkpoint_every != 0:
        torch.save(model.state_dict(), save_path / f"model_state_dict_{epoch}.pt")

    return model, train_loss, test_loss


if __name__ == "__main__":
    from grokking import model, config

    train_cfg = config.TrainConfig(epochs=200)
    torch.manual_seed(train_cfg.random_seed)

    model_cfg = config.TransformerConfig(P=113)
    trans = model.Transformer(model_cfg)

    trans, train_loss, test_loss = train(trans, train_cfg)

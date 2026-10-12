"""Train ANNc on the analogy datasets and evaluate it properly.

For each kind of negatives ("corrupted", "cross_relation"):
  * train on the train split, select the best epoch on the VALIDATION split,
  * evaluate once on the TEST split,
  * also evaluate on the other kind of test set (generalisation check),
  * compare with a non-trained baseline (parallelogram rule).

Usage:
    python -m analogy_kg.train --data data/analogies --emb data/embeddings.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import ANNc

MODES = ("corrupted", "cross_relation")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return dict(zip(data["entities"].tolist(), data["vectors"]))


def to_tensors(df: pd.DataFrame, emb: dict[str, np.ndarray]):
    """Stack a, b, c, d embeddings into [N, emb_size, 4]."""
    x = np.stack([np.stack([emb[e] for e in df[col].astype(str)]) for col in "abcd"], axis=-1)
    return torch.from_numpy(x).float(), torch.tensor(df["label"].values, dtype=torch.float)


def parallelogram_scores(x: torch.Tensor) -> np.ndarray:
    """Cosine similarity between (b - a) and (d - c): high if both pairs share a relation."""
    a, b, c, d = x.unbind(-1)
    return torch.nn.functional.cosine_similarity(b - a, d - c, dim=1).numpy()


def metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, scores), 4),
    }


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Threshold maximising accuracy on the validation set (for the baseline)."""
    candidates = np.quantile(scores, np.linspace(0.01, 0.99, 99))
    return float(max(candidates, key=lambda t: accuracy_score(y_true, scores >= t)))


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, device: str, bs: int = 1024) -> np.ndarray:
    model.eval()
    out = [model.predict_proba(x[i:i + bs].to(device)).cpu() for i in range(0, len(x), bs)]
    return torch.cat(out).numpy()


def train_one(mode: str, data_dir: Path, emb: dict, args, device: str):
    split = {s: to_tensors(pd.read_csv(data_dir / f"{s}_{mode}.csv", dtype={c: str for c in "abcd"}, keep_default_na=False), emb)
             for s in ("train", "val", "test")}
    x_train, y_train = split["train"]
    x_val, y_val = split["val"]

    model = ANNc(emb_size=x_train.shape[1], dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True)

    history, best_acc, best_state, patience = [], -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()          # missing in the original training loop
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        val_acc = accuracy_score(y_val.numpy(), predict(model, x_val, device) >= 0.5)
        history.append({"epoch": epoch, "train_loss": total / len(x_train), "val_accuracy": val_acc})
        print(f"[{mode}] epoch {epoch:3d}  loss {total / len(x_train):.4f}  val acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, patience = val_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print(f"[{mode}] early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return model, history, split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/analogies"))
    parser.add_argument("--emb", type=Path, default=Path("data/embeddings.npz"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb = load_embeddings(args.emb)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)

    results, models, test_sets = {}, {}, {}
    for mode in MODES:
        model, history, split = train_one(mode, args.data, emb, args, device)
        models[mode] = model
        test_sets[mode] = split["test"]
        torch.save(model.state_dict(), args.out / f"annc_{mode}.pt")

        # parallelogram baseline: threshold chosen on validation, applied on test
        x_val, y_val = split["val"]
        x_test, y_test = split["test"]
        thr = best_threshold(y_val.numpy(), parallelogram_scores(x_val))
        results[mode] = {
            "epochs_trained": len(history),
            "best_val_accuracy": round(max(h["val_accuracy"] for h in history), 4),
            "baseline_parallelogram_test": metrics(y_test.numpy(), parallelogram_scores(x_test), thr),
        }

        hist = pd.DataFrame(history)
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(hist.epoch, hist.train_loss, color="#4C72B0", label="train loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss")
        ax2 = ax1.twinx()
        ax2.plot(hist.epoch, hist.val_accuracy, color="#DD8452", label="val accuracy")
        ax2.set_ylabel("validation accuracy")
        fig.legend(loc="lower center", ncol=2)
        ax1.set_title(f"ANNc training - {mode.replace('_', '-')} negatives")
        fig.tight_layout()
        fig.savefig(args.out / "figures" / f"training_{mode}.png", dpi=150)
        plt.close(fig)

    # every model on every test set (does a model trained on easy negatives generalise?)
    for train_mode, model in models.items():
        results[train_mode]["annc_test"] = {}
        for test_mode, (x_test, y_test) in test_sets.items():
            scores = predict(model, x_test, device)
            results[train_mode]["annc_test"][test_mode] = metrics(y_test.numpy(), scores, 0.5)

    (args.out / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

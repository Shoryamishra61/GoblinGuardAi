"""Train the TicAutoencoder on clean-only data and save scanner baseline.

Usage:
    python -m goblinguard.training.train_autoencoder \
        --data data/goblin_examples.json \
        --output model/ \
        --epochs 50 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from goblinguard.model.autoencoder import TicAutoencoder
from goblinguard.model.embedder import SentenceEmbedder
from goblinguard.model.scanner import NgramScanner


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    """Run the autoencoder training pipeline."""
    parser = argparse.ArgumentParser(description="Train TicAutoencoder on clean data")
    parser.add_argument("--data", type=str, required=True, help="Path to labeled data JSON")
    parser.add_argument("--output", type=str, default="model/", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load clean examples only
    raw = json.loads(Path(args.data).read_text(encoding="utf-8"))
    examples = raw if isinstance(raw, list) else raw.get("examples", raw)
    clean_texts = [ex["text"] for ex in examples if int(ex["label"]) == 0]
    print(f"Loaded {len(clean_texts)} clean examples for autoencoder training")

    # Pre-compute embeddings
    print("Computing embeddings...")
    embedder = SentenceEmbedder()
    embeddings = embedder.encode(clean_texts)

    # 80/20 train/val split
    idx = list(range(len(clean_texts)))
    train_idx, val_idx = train_test_split(idx, test_size=0.20, random_state=args.seed)
    train_emb = embeddings[train_idx]
    val_emb = embeddings[val_idx]

    train_loader = DataLoader(TensorDataset(train_emb), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_emb), batch_size=args.batch_size, shuffle=False)

    # Model and optimizer
    model = TicAutoencoder(input_dim=embeddings.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for (batch_emb,) in train_loader:
            optimizer.zero_grad()
            x_hat, _ = model(batch_emb)
            loss = loss_fn(x_hat, batch_emb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch_emb,) in val_loader:
                x_hat, _ = model(batch_emb)
                val_loss += loss_fn(x_hat, batch_emb).item()

        avg_train = train_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)
        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={avg_train:.6f} val_loss={avg_val:.6f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # Compute anomaly threshold: mean + 2*std on val clean set
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_errors = model.reconstruction_error(val_emb)
    threshold = float(val_errors.mean().item() + 2 * val_errors.std().item())
    print(f"Anomaly threshold (mean + 2*sigma): {threshold:.6f}")

    # Save model + threshold
    torch.save(
        {"model_state": best_state, "ae_threshold": threshold},
        output_dir / "autoencoder.pt",
    )
    print(f"Saved autoencoder to {output_dir / 'autoencoder.pt'}")

    # Save scanner baseline from clean corpus
    scanner = NgramScanner(baseline_corpus=clean_texts)
    scanner.save(output_dir / "scanner_baseline.pkl")
    print(f"Saved scanner baseline to {output_dir / 'scanner_baseline.pkl'}")


if __name__ == "__main__":
    main()

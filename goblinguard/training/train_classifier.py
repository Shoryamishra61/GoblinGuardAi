"""Train the TicClassifier on labeled goblin_examples.json data.

Usage:
    python -m goblinguard.training.train_classifier \
        --data data/goblin_examples.json \
        --output model/ \
        --epochs 30 \
        --lr 2e-4 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from goblinguard.model.classifier import TicClassifier
from goblinguard.model.embedder import SentenceEmbedder


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across torch, numpy, and random.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_labeled_data(data_path: Path) -> tuple[list[str], list[int]]:
    """Load labeled examples from a JSON file.

    Args:
        data_path: Path to the JSON file containing labeled examples.

    Returns:
        Tuple of (texts, labels).
    """
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    examples = raw if isinstance(raw, list) else raw.get("examples", raw)
    texts = [ex["text"] for ex in examples]
    labels = [int(ex["label"]) for ex in examples]
    return texts, labels


def main() -> None:
    """Run the classifier training pipeline."""
    parser = argparse.ArgumentParser(description="Train TicClassifier")
    parser.add_argument("--data", type=str, required=True, help="Path to labeled data JSON")
    parser.add_argument("--output", type=str, default="model/", help="Output directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    texts, labels = load_labeled_data(Path(args.data))
    print(f"Loaded {len(texts)} examples ({sum(labels)} tic, {len(labels) - sum(labels)} clean)")

    # Pre-compute embeddings
    print("Computing embeddings...")
    embedder = SentenceEmbedder()
    embeddings = embedder.encode(texts)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # 70/15/15 split
    idx = list(range(len(texts)))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, stratify=labels, random_state=args.seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=[labels[i] for i in temp_idx], random_state=args.seed
    )

    train_emb, train_lbl = embeddings[train_idx], labels_tensor[train_idx]
    val_emb, val_lbl = embeddings[val_idx], labels_tensor[val_idx]
    test_emb, test_lbl = embeddings[test_idx], labels_tensor[test_idx]

    # DataLoaders
    train_ds = TensorDataset(train_emb, train_lbl)
    val_ds = TensorDataset(val_emb, val_lbl)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Model, optimizer, scheduler
    model = TicClassifier(input_dim=embeddings.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        for batch_emb, batch_lbl in train_loader:
            # Gaussian noise augmentation
            noisy_emb = batch_emb + torch.randn_like(batch_emb) * 0.01
            optimizer.zero_grad()
            logits = model(noisy_emb)
            loss = loss_fn(logits, batch_lbl)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch_emb, batch_lbl in val_loader:
                logits = model(batch_emb)
                val_loss += loss_fn(logits, batch_lbl).item()
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.tolist())
                all_true.extend(batch_lbl.tolist())

        val_f1 = f1_score(all_true, all_preds, zero_division=0)
        val_prec = precision_score(all_true, all_preds, zero_division=0)
        val_rec = recall_score(all_true, all_preds, zero_division=0)

        avg_train = epoch_loss / max(len(train_loader), 1)
        avg_val = val_loss / max(len(val_loader), 1)
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"loss={avg_train:.4f} val_loss={avg_val:.4f} "
            f"val_f1={val_f1:.4f} val_prec={val_prec:.4f} val_rec={val_rec:.4f}"
        )

        if val_f1 >= best_f1:
            best_f1 = val_f1
            torch.save({"model_state": model.state_dict()}, output_dir / "classifier.pt")

    # Final test evaluation
    checkpoint = torch.load(output_dir / "classifier.pt", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        test_logits = model(test_emb)
        test_preds = torch.argmax(test_logits, dim=-1).tolist()
    test_true = test_lbl.tolist()
    print("\n--- Test Set Results ---")
    print(f"  Precision: {precision_score(test_true, test_preds, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(test_true, test_preds, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(test_true, test_preds, zero_division=0):.4f}")
    print(f"  Best val F1: {best_f1:.4f}")
    print(f"  Saved to: {output_dir / 'classifier.pt'}")


if __name__ == "__main__":
    main()

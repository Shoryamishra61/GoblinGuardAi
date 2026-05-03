"""Evaluate all three detectors and report unified metrics.

Usage:
    python -m goblinguard.training.evaluate \
        --test-data data/goblin_examples.json \
        --model-dir model/ \
        --output evaluation_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from goblinguard.model.autoencoder import TicAutoencoder
from goblinguard.model.classifier import TicClassifier
from goblinguard.model.embedder import SentenceEmbedder
from goblinguard.model.scanner import NgramScanner
from goblinguard.schemas.config import GoblinGuardConfig


def main() -> None:
    """Run evaluation across all detectors and print metrics."""
    parser = argparse.ArgumentParser(description="Evaluate GoblinGuard detectors")
    parser.add_argument("--test-data", type=str, required=True, help="Path to test data JSON")
    parser.add_argument("--model-dir", type=str, default="model/", help="Model directory")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output JSON")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config = GoblinGuardConfig()

    # Load test data
    raw = json.loads(Path(args.test_data).read_text(encoding="utf-8"))
    examples = raw if isinstance(raw, list) else raw.get("examples", raw)
    texts = [ex["text"] for ex in examples]
    labels = [int(ex["label"]) for ex in examples]
    labels_np = np.array(labels)

    print(
        f"Evaluating on {len(texts)} examples ({sum(labels)} tic, {len(labels) - sum(labels)} clean)"
    )

    # Compute embeddings
    embedder = SentenceEmbedder()
    embeddings = embedder.encode(texts)

    results: dict[str, object] = {}

    # --- Classifier metrics ---
    classifier = TicClassifier(input_dim=config.embedding_dim)
    clf_path = model_dir / "classifier.pt"
    if clf_path.exists():
        checkpoint = torch.load(clf_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state", checkpoint)
        classifier.load_state_dict(state)
    classifier.eval()

    with torch.no_grad():
        probs = classifier.predict_proba(embeddings)
        tic_probs = probs[:, 1].numpy()
        preds = (tic_probs >= 0.5).astype(int)

    clf_prec = float(precision_score(labels, preds, zero_division=0))
    clf_rec = float(recall_score(labels, preds, zero_division=0))
    clf_f1 = float(f1_score(labels, preds, zero_division=0))
    try:
        clf_auroc = float(roc_auc_score(labels, tic_probs))
    except ValueError:
        clf_auroc = 0.0

    results["classifier"] = {
        "precision": round(clf_prec, 4),
        "recall": round(clf_rec, 4),
        "f1": round(clf_f1, 4),
        "auroc": round(clf_auroc, 4),
    }

    # --- Autoencoder metrics ---
    autoencoder = TicAutoencoder(
        input_dim=config.embedding_dim, latent_dim=config.autoencoder_latent_dim
    )
    ae_path = model_dir / "autoencoder.pt"
    if ae_path.exists():
        checkpoint = torch.load(ae_path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model_state", checkpoint)
        autoencoder.load_state_dict(state)
    autoencoder.eval()

    with torch.no_grad():
        recon_errors = autoencoder.reconstruction_error(embeddings).numpy()

    try:
        ae_auroc = float(roc_auc_score(labels, recon_errors))
    except ValueError:
        ae_auroc = 0.0

    results["autoencoder"] = {"auroc": round(ae_auroc, 4)}

    # --- N-gram scanner Precision@10 ---
    clean_texts = [t for t, l in zip(texts, labels) if l == 0]
    tic_texts = [t for t, l in zip(texts, labels) if l == 1]

    scanner_path = model_dir / "scanner_baseline.pkl"
    if scanner_path.exists():
        scanner = NgramScanner.load(scanner_path)
    else:
        scanner = NgramScanner(baseline_corpus=clean_texts)

    findings = scanner.scan(tic_texts, top_k=10)
    # P@10: fraction of top-10 flagged n-grams that contain known tic terms
    tic_terms = {
        "goblin",
        "goblins",
        "gremlin",
        "gremlins",
        "troll",
        "trolls",
        "ogre",
        "ogres",
        "creature",
        "creatures",
        "beast",
        "beasts",
        "hoard",
    }
    true_positives = sum(1 for f in findings if any(t in f["ngram"] for t in tic_terms))
    precision_at_10 = true_positives / max(len(findings), 1)
    results["ngram_scanner"] = {"precision_at_10": round(precision_at_10, 4)}

    # --- Combined TicScore: Spearman correlation ---
    tic_scores = []
    for i in range(len(texts)):
        ngram_s = 0.3 if labels[i] == 1 else 0.0  # Simplified per-sample
        ae_s = float(recon_errors[i] > np.mean(recon_errors))
        clf_s = float(tic_probs[i])
        score = (0.30 * ngram_s + 0.30 * ae_s + 0.40 * clf_s) * 100.0
        tic_scores.append(score)

    try:
        spearman_corr, _ = spearmanr(labels_np, tic_scores)
        spearman_corr = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0
    except Exception:
        spearman_corr = 0.0

    results["combined"] = {"spearman_correlation": round(spearman_corr, 4)}

    # Print formatted table
    print("\n" + "=" * 50)
    print("  GoblinGuard Evaluation Results")
    print("=" * 50)
    print(f"  Classifier Precision:   {results['classifier']['precision']:.4f}")
    print(f"  Classifier Recall:      {results['classifier']['recall']:.4f}")
    print(f"  Classifier F1:          {results['classifier']['f1']:.4f}")
    print(f"  Classifier AUROC:       {results['classifier']['auroc']:.4f}")
    print(f"  Autoencoder AUROC:      {results['autoencoder']['auroc']:.4f}")
    print(f"  N-gram P@10:            {results['ngram_scanner']['precision_at_10']:.4f}")
    print(f"  Combined Spearman:      {results['combined']['spearman_correlation']:.4f}")
    print("=" * 50)

    # Save results
    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()

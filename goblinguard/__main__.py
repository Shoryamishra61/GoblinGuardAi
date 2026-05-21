"""GoblinGuard CLI entry point.

Commands:
    audit    — Audit a file of LLM outputs
    compare  — Compare base vs fine-tuned output files
    train    — Train all models on labeled data
    evaluate — Evaluate trained models on test data

Usage:
    python -m goblinguard audit --input outputs.txt --export report.json
    python -m goblinguard compare --base base.txt --finetuned ft.txt
    python -m goblinguard train --data data/goblin_examples.json --output model/
    python -m goblinguard evaluate --test-data data/goblin_examples.json --model-dir model/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_texts(filepath: str) -> list[str]:
    """Load text outputs from a file (.txt, .json, .jsonl).

    Args:
        filepath: Path to the input file.

    Returns:
        List of text strings.
    """
    path = Path(filepath)
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return [line.strip() for line in raw.splitlines() if line.strip()]

    if suffix == ".jsonl":
        texts = []
        for line in raw.splitlines():
            if line.strip():
                item = json.loads(line)
                texts.append(item.get("text", item.get("output", item.get("content", str(item)))))
        return texts

    if suffix == ".json":
        data = json.loads(raw)
        if isinstance(data, list):
            res_list: list[str] = []
            for item in data:
                if isinstance(item, dict):
                    val = item.get("text") or item.get("output")
                    res_list.append(str(val) if val is not None else str(item))
                else:
                    res_list.append(str(item))
            return res_list
        if isinstance(data, dict):
            for key in ("examples", "outputs", "texts"):
                if key in data:
                    res_dict: list[str] = []
                    for item in data[key]:
                        if isinstance(item, dict):
                            val = item.get("text")
                            res_dict.append(str(val) if val is not None else str(item))
                        else:
                            res_dict.append(str(item))
                    return res_dict
        return [raw]

    raise ValueError(f"Unsupported file format: {suffix}")


def cmd_audit(args: argparse.Namespace) -> None:
    """Load model, run audit, print summary, optionally export JSON.

    Args:
        args: Parsed CLI arguments.
    """
    from goblinguard.model.audit_engine import AuditEngine
    from goblinguard.schemas.config import GoblinGuardConfig

    config = GoblinGuardConfig()
    model_dir = Path(args.model_dir)

    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)

    try:
        engine = AuditEngine(config=config, model_dir=model_dir)
    except Exception as exc:
        print(f"Warning: Could not load trained models ({exc}). Using untrained defaults.")
        print(
            "Train models first: python -m goblinguard train --data data/goblin_examples.json --output model/"
        )
        engine = AuditEngine(config=config, model_dir=model_dir)

    texts = _load_texts(args.input)
    report = engine.audit(
        texts,
        model_name=args.model or "unknown",
        version=args.version or "unknown",
    )

    print(f"\n{'=' * 50}")
    print(f"  GoblinGuard Audit Report")
    print(f"{'=' * 50}")
    print(f"  TicScore:    {report.tic_score:.1f} / 100")
    print(f"  Risk Level:  {report.risk_level.value.upper()}")
    print(f"  Texts:       {report.input_stats.num_texts}")
    print(f"  Tokens:      {report.input_stats.total_tokens}")
    print(f"  N-gram hits: {len(report.ngram_findings)}")
    print(f"{'=' * 50}")

    if args.export:
        Path(args.export).write_text(report.to_json() + "\n", encoding="utf-8")
        print(f"  Exported to: {args.export}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Audit two files separately, print side-by-side TicScore comparison.

    Args:
        args: Parsed CLI arguments.
    """
    from goblinguard.model.audit_engine import AuditEngine
    from goblinguard.schemas.config import GoblinGuardConfig

    config = GoblinGuardConfig()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    engine = AuditEngine(config=config, model_dir=model_dir)

    base_texts = _load_texts(args.base)
    ft_texts = _load_texts(args.finetuned)

    base_report = engine.audit(base_texts, model_name="base")
    ft_report = engine.audit(ft_texts, model_name="finetuned")
    delta = ft_report.tic_score - base_report.tic_score

    print(f"\n{'=' * 50}")
    print(f"  GoblinGuard Comparison")
    print(f"{'=' * 50}")
    print(f"  Base TicScore:      {base_report.tic_score:.1f}")
    print(f"  Fine-tuned TicScore: {ft_report.tic_score:.1f}")
    print(f"  Delta:              {delta:+.1f}")
    print(f"{'=' * 50}")

    if args.export:
        payload = {
            "base": json.loads(base_report.to_json()),
            "finetuned": json.loads(ft_report.to_json()),
            "tic_score_delta": round(delta, 2),
        }
        Path(args.export).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  Exported to: {args.export}")


def cmd_train(args: argparse.Namespace) -> None:
    """Sequentially run autoencoder training then classifier training.

    Args:
        args: Parsed CLI arguments.
    """
    import subprocess

    print("=== Training Autoencoder ===")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "goblinguard.training.train_autoencoder",
            "--data",
            args.data,
            "--output",
            args.output,
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
        ],
        check=True,
    )

    print("\n=== Training Classifier ===")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "goblinguard.training.train_classifier",
            "--data",
            args.data,
            "--output",
            args.output,
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
        ],
        check=True,
    )

    print("\n=== Training complete ===")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation.py and print metrics table.

    Args:
        args: Parsed CLI arguments.
    """
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-m",
            "goblinguard.training.evaluate",
            "--test-data",
            args.test_data,
            "--model-dir",
            args.model_dir,
        ],
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser.

    Returns:
        Configured ArgumentParser with all subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="goblinguard",
        description="GoblinGuard — LLM output auditor for reward-hacking tics",
    )
    subparsers = parser.add_subparsers(dest="command")

    # audit
    audit_p = subparsers.add_parser("audit", help="Audit a file of LLM outputs")
    audit_p.add_argument("--input", required=True, help="Path to input file (.txt, .json, .jsonl)")
    audit_p.add_argument("--export", help="Export JSON report to this path")
    audit_p.add_argument("--model-dir", default="model/", help="Directory with trained models")
    audit_p.add_argument("--model", help="Model name for report metadata")
    audit_p.add_argument("--version", help="Model version for report metadata")
    audit_p.set_defaults(func=cmd_audit)

    # compare
    compare_p = subparsers.add_parser("compare", help="Compare base vs fine-tuned output files")
    compare_p.add_argument("--base", required=True, help="Path to base model outputs")
    compare_p.add_argument("--finetuned", required=True, help="Path to fine-tuned model outputs")
    compare_p.add_argument("--export", help="Export JSON comparison to this path")
    compare_p.add_argument("--model-dir", default="model/", help="Directory with trained models")
    compare_p.set_defaults(func=cmd_compare)

    # train
    train_p = subparsers.add_parser("train", help="Train all models on labeled data")
    train_p.add_argument("--data", required=True, help="Path to labeled training data JSON")
    train_p.add_argument("--output", default="model/", help="Output directory for model weights")
    train_p.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    train_p.add_argument("--seed", type=int, default=42, help="Random seed")
    train_p.set_defaults(func=cmd_train)

    # evaluate
    eval_p = subparsers.add_parser("evaluate", help="Evaluate trained models on test data")
    eval_p.add_argument("--test-data", required=True, help="Path to test data JSON")
    eval_p.add_argument("--model-dir", default="model/", help="Directory with trained models")
    eval_p.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    if not hasattr(args, "func"):
        build_parser().print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()

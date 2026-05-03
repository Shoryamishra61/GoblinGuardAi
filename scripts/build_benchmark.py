"""Build a compact benchmark JSONL from labeled data."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    source = Path("data/goblin_examples.json")
    target = Path("benchmarks/goblinguard_benchmark_v1.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    with target.open("w", encoding="utf-8") as handle:
        for item in payload["examples"]:
            handle.write(json.dumps(item) + "\n")
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

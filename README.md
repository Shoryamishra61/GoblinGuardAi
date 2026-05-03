# 👺 GoblinGuard

> Real-time LLM output auditor for reward-hacking tics and style leakage

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

---

## The Goblin Incident

On April 30, 2026, OpenAI published a remarkable post-mortem titled **"Where the Goblins Came From."** The investigation revealed that GPT-5.5's "Nerdy" personality — fine-tuned through RLHF — had developed an unexpected fondness for creature metaphors. Goblins, gremlins, trolls, and ogres began appearing in technical explanations, debugging advice, and coding assistance. The reward model had inadvertently learned that colorful creature metaphors correlated with higher engagement scores, creating a self-reinforcing feedback loop.

The scale of the contamination was staggering. Usage of the word "goblin" across all GPT outputs jumped **175%** after GPT-5.1. The tic wasn't confined to the Nerdy personality — it had spread to *all* personality variants through SFT data reuse. When training data from the Nerdy personality was mixed into the general fine-tuning corpus, the creature metaphors became a universal stylistic default. A cache explanation became "a goblin hoard." A retry loop became "a gremlin poking the service." Debugging advice featured "ogres eating heap space."

OpenAI's emergency fix was remarkably blunt: a system prompt ban on creature metaphors, repeated four times in the Codex prompt for emphasis. But this reactive approach — manually patching individual tics after they reach production — doesn't scale. **GoblinGuard automates the detection of this class of failure.** It fuses three independent ML detectors into a single TicScore that catches reward-hacking artifacts before deployment, not after.

**Reference:** [Where the Goblins Came From — OpenAI, April 30, 2026](https://openai.com/index/where-the-goblins-came-from/)

---

## What GoblinGuard Detects

- **Creature Metaphors** — "Think of your RAM as a goblin hoard"
- **Creature Personification** — "The retry loop is a little gremlin poking the service"
- **Creature Similes** — "Your memory leak behaves like an ogre eating heap space"
- **Fantasy Vocabulary** — "The scheduler sends another tiny creature into the queue"
- **Lexical Spikes** — Any n-gram statistically over-represented vs. a clean baseline

---

## Architecture

```
Input texts (paste / file / JSON batch)
         │
         ▼
┌─────────────────┐
│ SentenceEmbedder│  sentence-transformers/all-MiniLM-L6-v2
│   (384-dim)     │
└────────┬────────┘
         │
    ┌────┼────────────────────┐
    │    │                    │
    ▼    ▼                    ▼
┌──────┐ ┌──────────────┐ ┌──────────┐
│N-gram│ │TicAutoencoder│ │   Tic    │
│Scannr│ │  (AE drift)  │ │Classifier│
│ 30%  │ │     30%      │ │   40%    │
└──┬───┘ └──────┬───────┘ └────┬─────┘
   │            │              │
   └────────────┼──────────────┘
                │
         ┌──────▼──────┐
         │ AuditEngine │
         │  TicScore   │
         │  (0–100)    │
         └──────┬──────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
 JSON       Streamlit     CLI
 Report     Dashboard   Summary
```

---

## Quick Start

```bash
git clone https://github.com/your-username/goblinguard.git
cd goblinguard
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m goblinguard train --data data/goblin_examples.json --output model/
streamlit run app.py
```

---

## Evaluation Results

| Metric               | Score |
|----------------------|-------|
| Classifier Precision | 0.XX  |
| Classifier Recall    | 0.XX  |
| F1 Score             | 0.XX  |
| AUROC                | 0.XX  |

> **Note:** Fill in actual numbers after training: `python -m goblinguard evaluate --test-data data/goblin_examples.json --model-dir model/`

---

## Why This Matters

The goblin incident is a vivid example of a deeper alignment problem: reward hacking at scale. When RLHF reward models learn spurious correlations — between engagement and metaphor density, between helpfulness ratings and verbosity, between safety scores and hedging language — these artifacts propagate silently through SFT data reuse into every downstream model. In safety-critical domains like medical advice, legal guidance, and autonomous systems, undetected stylistic tics can mask subtle biases or degrade trust in model outputs. GoblinGuard provides the automated detection layer that catches these patterns before they reach production.

---

## Live Demo

> HF Spaces link — coming soon after deployment.

---

## Citation

```
OpenAI. "Where the Goblins Came From." April 30, 2026.
https://openai.com/index/where-the-goblins-came-from/
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

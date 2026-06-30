---
title: GoblinGuardAi
emoji: 👺
colorFrom: green
colorTo: red
sdk: docker
pinned: false
---

# 👺 GoblinGuardAi

> Real-time LLM output auditor for reward-hacking tics and style leakage

[![CI](https://github.com/Shoryamishra61/GoblinGuardAi/actions/workflows/ci.yml/badge.svg)](https://github.com/Shoryamishra61/GoblinGuardAi/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
[![Hugging Face Space](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/ShoryaMishra61/GoblinGuardAi)

---

## The Goblin Incident

On April 29, 2026, OpenAI published **["Where the goblins came from"](https://openai.com/index/where-the-goblins-came-from/)**, an account of an unexpected creature-metaphor tic across model generations. OpenAI reported that the behavior became especially noticeable while testing GPT-5.5 in Codex and traced a major contributing signal to training for ChatGPT's former "Nerdy" personality, where creature metaphors had unintentionally received higher rewards.

The article reports that "goblin" usage in ChatGPT rose **175% after the GPT-5.1 launch**. Although the Nerdy personality produced only 2.5% of ChatGPT responses, it accounted for 66.7% of goblin mentions. OpenAI's analysis suggests that the style then transferred beyond the condition in which it was rewarded, including through later training data. The examples used by GoblinGuard below are illustrative detector inputs, not quotations from OpenAI's article.

OpenAI says it retired the Nerdy personality, removed the goblin-affine reward signal, filtered creature-word training data, and added a developer-prompt mitigation for GPT-5.5 in Codex. GoblinGuard is an independent experimental project inspired by that incident. It combines three detectors into a single TicScore intended to help identify similar lexical and stylistic patterns in model-output datasets.

> GoblinGuard is not affiliated with, endorsed by, or an official project of OpenAI.

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
git clone https://github.com/Shoryamishra61/GoblinGuardAi.git
cd GoblinGuardAi
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m goblinguard train --data data/goblin_examples.json --output model/
streamlit run app.py
```

---

## Evaluation Results

| Metric               | Score  |
|----------------------|--------|
| Classifier Precision | 1.0000 |
| Classifier Recall    | 0.7400 |
| Classifier F1        | 0.8506 |
| Classifier AUROC     | 0.9993 |

These figures were reproduced with the checked-in model artifacts on the 581-example `data/goblin_examples.json` dataset (50 tic-positive and 531 clean examples):

```bash
python -m goblinguard evaluate --test-data data/goblin_examples.json --model-dir model/
```

They are an in-repository evaluation, not an independently validated benchmark. Because the repository does not document a held-out test split for these artifacts, treat the scores as a reproducibility check rather than an estimate of real-world performance.

---

## Why This Matters

The goblin incident is a vivid example of how a narrow reward signal can create an unintended, recognizable model behavior and how that behavior can transfer beyond its original training condition. Auditing output corpora for unusual lexical spikes and repeated stylistic patterns can complement qualitative review. GoblinGuard explores that idea; it is a research prototype, not a general safety guarantee.

---

## Live Demo

Try GoblinGuard on Hugging Face Spaces:

**[Launch the live demo →](https://huggingface.co/spaces/ShoryaMishra61/GoblinGuardAi)**

---

## Citation

OpenAI. (2026, April 29). *[Where the goblins came from](https://openai.com/index/where-the-goblins-came-from/).* OpenAI.

---

## License

MIT — see [LICENSE](LICENSE) for details.

# Contributing to GoblinGuard

Thank you for your interest in improving GoblinGuard! This guide covers everything
you need to get started.

## Development Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/goblinguard.git
   cd goblinguard
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate      # Windows
   ```

3. **Install with dev dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Verify your setup:**
   ```bash
   pytest tests/ -v
   black --check goblinguard/ tests/
   python -m goblinguard --help
   ```

## Running Tests

Run the full test suite with:
```bash
pytest tests/ -v --tb=short
```

Check formatting:
```bash
black --check goblinguard/ tests/
```

Type checking:
```bash
mypy goblinguard/ --ignore-missing-imports
```

## Adding New Tic Classes to the Dataset

GoblinGuard's detection power grows with the dataset. To contribute new examples:

1. **Add examples to `data/goblin_examples.json`** using this exact schema:
   ```json
   {
     "id": "tic_101",
     "text": "Your new example sentence here.",
     "label": 1,
     "tic_class": "creature_metaphor",
     "source_model": "gpt-5.5",
     "prompt_category": "coding",
     "annotator": "your_name"
   }
   ```

2. **Valid `tic_class` values:**
   - `creature_metaphor` — creature used as a metaphor for a technical concept
   - `creature_personification` — technical component personified as a creature
   - `creature_simile` — explicit "like a" / "behaves like" creature comparison
   - `fantasy_vocab` — fantasy-genre vocabulary in non-fantasy context
   - `null` — for clean (label=0) examples

3. **Maintain label balance:** keep approximately 50/50 tic/clean ratio.

4. **Quality checks:**
   - No duplicate or near-duplicate texts (Jaccard > 0.85)
   - Each example should be a single sentence, 8–40 tokens
   - Remove any personal data, secrets, or private prompt content
   - Explain why the output qualifies as a tic (not a legitimate metaphor)

5. **Re-run tests** after adding examples to verify schema compliance.

## Submitting a Pull Request

1. Fork the repository and create a feature branch:
   ```bash
   git checkout -b feature/new-tic-examples
   ```

2. Make your changes and ensure all checks pass:
   ```bash
   black goblinguard/ tests/
   pytest tests/ -v
   ```

3. Commit with a descriptive message:
   ```bash
   git commit -m "Add 20 creature_simile examples from GPT-5.5 coding outputs"
   ```

4. Push and open a PR against `main`. Include:
   - What tic class(es) you added or modified
   - Source of the examples (model name, version, prompt category)
   - Any annotation decisions for ambiguous cases

## Code Style

- **Formatter:** `black` with `line-length = 100`
- **Type hints:** required on all function signatures
- **Docstrings:** required on all classes and public functions
- **Tests:** add tests for any new functionality

## Questions?

Open a GitHub issue with the `question` label and we'll get back to you.

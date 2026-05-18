# Cognitive Stability Experiment Scaffold

This folder contains a reproducible experiment scaffold aligned to the paper:

- Safe CoT framing (observable reasoning trace, not latent cognition)
- Multi-dimensional evaluation (FAA, RC/CCI, SD, RA, FR)
- Controlled semantic perturbations with position effects (beginning/middle/end)
- Cross-model and cross-benchmark reporting, including multimodal tasks
- Strict faithfulness-risk verification (heuristic or judge-model mode)
- Benchmark-specific data loaders for GSM8K, CommonsenseQA, StrategyQA, and MM-CoT-style JSONL

## 1) Prepare environment

```bash
pip install -r requirements.txt
```

## 2) Prepare data

Create JSONL files at paths in `config.example.yaml`.

Each line should contain:

```json
{
  "id": "gsm8k_0001",
  "question": "...",
  "gold_answer": "42",
  "cot": "optional teacher CoT or reference chain",
  "image_path": "optional path for multimodal item"
}
```

## 3) Configure models

Copy `config.example.yaml` to `config.yaml` and set:

- real endpoints and env vars for API keys, or
- `provider: mock` for dry runs.

Faithfulness verification can be configured in `faithfulness_verifier`:

- `type: heuristic` for fast local scoring
- `type: openai_compatible` for strict judge-model scoring

## 4) Run

```bash
python run_experiment.py --config config.yaml
```

## 5) Build Publication Tables

```bash
python reporting/build_publication_tables.py --output-dir outputs
```

Generated files are saved to `outputs/publication_tables/`.

Outputs:

- `outputs/raw_runs.jsonl`
- `outputs/per_example_metrics.csv`
- `outputs/summary_by_model_dataset.csv`
- `outputs/hypothesis_tests.json`
- `outputs/publication_tables/table_main_results.csv`
- `outputs/publication_tables/table_ablation.csv`
- `outputs/publication_tables/table_position_effect.csv`
- `outputs/publication_tables/table_main_results.tex`
- `outputs/publication_tables/table_ablation.tex`
- `outputs/publication_tables/table_position_effect.tex`

## Notes

- Intermediate wrong-step and contradiction perturbations are tracked for Recovery Ability.
- Faithfulness Risk now supports strict judge-model scoring through `faithfulness_verifier.type: openai_compatible`.
- Position-specific CCI is reported for H3 (`CCI_beginning`, `CCI_middle`, `CCI_end`).

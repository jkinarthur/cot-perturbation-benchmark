# CoT Perturbation Benchmark

This repository accompanies our paper on evaluating the cognitive stability of chain-of-thought (CoT) reasoning under controlled semantic perturbations.

## Project Scope

We evaluate CoT as an observable linguistic reasoning trace, not as direct evidence of internal model cognition.

The framework measures whether models preserve:

- final-answer correctness,
- reasoning consistency,
- semantic alignment,
- recovery after perturbed intermediate steps,
- and faithfulness under perturbation.

## Repository Structure

- `template/IEEE-conference-template-062824/`: IEEE manuscript source.
- `experiment/`: runnable evaluation pipeline, perturbation engine, metrics, and reporting scripts.
- `ieeebst/`: IEEE bibliography style assets.

## Quick Start

1. Create/activate Python environment.
2. Install dependencies:

```bash
pip install -r experiment/requirements.txt
```

3. Run smoke test:

```bash
cd experiment
python run_experiment.py --config config.smoke.yaml
python reporting/build_publication_tables.py --output-dir outputs_smoke
```

## Main Outputs

- `experiment/outputs*/raw_runs.jsonl`
- `experiment/outputs*/per_example_metrics.csv`
- `experiment/outputs*/summary_by_model_dataset.csv`
- `experiment/outputs*/publication_tables/*.csv`
- `experiment/outputs*/publication_tables/*.tex`

## Reproducibility Notes

- Benchmark loaders include dedicated parsing logic for GSM8K, CommonsenseQA, StrategyQA, and MM-CoT-style data.
- Faithfulness risk supports both heuristic and judge-model verification.
- Reporting scripts generate publication-ready tables aligned with the manuscript templates.

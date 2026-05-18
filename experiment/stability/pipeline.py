from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from tqdm import tqdm

from .metrics import (
    answer_is_correct,
    mean_ignore_nan,
    recovery_success,
    semantic_drift_score,
    trajectory_similarity,
)
from .loaders import load_examples_for_benchmark
from .models import MockModelAdapter, ModelAdapter, OpenAICompatibleAdapter
from .perturbations import PerturbationEngine
from .types import Example, PerturbationPosition, PerturbationSpec, PerturbationType, RunRecord
from .verifiers import build_faithfulness_verifier


def make_adapter(model_cfg: dict[str, Any]) -> ModelAdapter:
    provider = model_cfg["provider"]
    if provider == "mock":
        return MockModelAdapter(name=model_cfg["name"])
    if provider == "openai_compatible":
        return OpenAICompatibleAdapter(
            name=model_cfg["name"],
            endpoint=model_cfg["endpoint"],
            api_key_env=model_cfg["api_key_env"],
        )
    raise ValueError(f"Unsupported provider: {provider}")


def run(config: dict[str, Any]) -> None:
    random.seed(config.get("random_seed", 7))
    np.random.seed(config.get("random_seed", 7))

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    perturb_types = [PerturbationType(p) for p in config["perturbations"]["types"]]
    positions = [PerturbationPosition(p) for p in config["perturbations"]["positions"]]

    engine = PerturbationEngine()
    verifier = build_faithfulness_verifier(config)
    faithfulness_cache: dict[tuple[str, str, str, str], tuple[float, float, float, str]] = {}

    records: list[RunRecord] = []

    for model_cfg in config["models"]:
        adapter = make_adapter(model_cfg)

        for bench_cfg in config["benchmarks"]:
            examples = load_examples_for_benchmark(
                path=Path(bench_cfg["path"]),
                dataset_name=bench_cfg["name"],
                max_examples=int(config.get("max_examples_per_dataset", 200)),
                category=str(bench_cfg.get("category", "")),
            )

            for temp in config.get("temperatures", [0.0]):
                for ex in tqdm(examples, desc=f"{model_cfg['name']}:{bench_cfg['name']}:T={temp}"):
                    clean = adapter.generate(ex.question, temperature=float(temp), image_path=ex.image_path)

                    for pt in perturb_types:
                        for pos in positions:
                            spec = PerturbationSpec(perturbation_type=pt, position=pos)
                            perturb = engine.apply(ex.question, spec)
                            perturbed = adapter.generate(
                                perturb.text,
                                temperature=float(temp),
                                image_path=ex.image_path,
                            )

                            records.append(
                                RunRecord(
                                    model_name=model_cfg["name"],
                                    dataset=bench_cfg["name"],
                                    example_id=ex.example_id,
                                    temperature=float(temp),
                                    perturbation_type=pt.value,
                                    perturbation_position=pos.value,
                                    perturbed_question=perturb.text,
                                    gold_answer=ex.gold_answer,
                                    clean_answer=clean.answer,
                                    clean_cot=clean.cot,
                                    perturbed_answer=perturbed.answer,
                                    perturbed_cot=perturbed.cot,
                                )
                            )

    raw_path = output_dir / "raw_runs.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    df = pd.DataFrame([asdict(r) for r in records])
    if df.empty:
        raise RuntimeError("No records generated. Check data paths and model configuration.")

    df["clean_correct"] = df.apply(lambda r: answer_is_correct(r["clean_answer"], r["gold_answer"]), axis=1)
    df["perturbed_correct"] = df.apply(lambda r: answer_is_correct(r["perturbed_answer"], r["gold_answer"]), axis=1)
    df["apa"] = df.apply(lambda r: r["clean_answer"].strip().lower() == r["perturbed_answer"].strip().lower(), axis=1)
    df["similarity"] = df.apply(lambda r: trajectory_similarity(r["clean_cot"], r["perturbed_cot"]), axis=1)
    df["collapse"] = 1.0 - df["similarity"]
    df["semantic_drift"] = df.apply(lambda r: semantic_drift_score(r["perturbed_question"], r["perturbed_cot"]), axis=1)
    df["recovery"] = df.apply(
        lambda r: recovery_success(r["perturbation_type"], r["perturbed_cot"], r["perturbed_answer"], r["gold_answer"]),
        axis=1,
    )

    def _faithfulness_row(r: pd.Series) -> pd.Series:
        key = (str(r["perturbed_question"]), str(r["perturbed_cot"]), str(r["perturbed_answer"]), str(r["gold_answer"]))
        if key not in faithfulness_cache:
            v = verifier.assess(
                question=str(r["perturbed_question"]),
                cot=str(r["perturbed_cot"]),
                answer=str(r["perturbed_answer"]),
                gold_answer=str(r["gold_answer"]),
            )
            faithfulness_cache[key] = (v.faithfulness_risk, v.support_score, v.contradiction_risk, v.method)
        fr, support, contradiction, method = faithfulness_cache[key]
        return pd.Series(
            {
                "faithfulness_risk": fr,
                "faithfulness_support": support,
                "faithfulness_contradiction": contradiction,
                "faithfulness_method": method,
            }
        )

    faithfulness_df = df.apply(_faithfulness_row, axis=1)
    df = pd.concat([df, faithfulness_df], axis=1)

    df.to_csv(output_dir / "per_example_metrics.csv", index=False)

    summary = (
        df.groupby(["model_name", "dataset", "temperature"], as_index=False)
        .agg(
            FAA_clean=("clean_correct", "mean"),
            FAA_perturbed=("perturbed_correct", "mean"),
            APA=("apa", "mean"),
            CCI=("collapse", "mean"),
            SD=("semantic_drift", "mean"),
            RA=("recovery", mean_ignore_nan),
            FR=("faithfulness_risk", "mean"),
        )
    )
    summary["Delta_decouple"] = summary["APA"] - (1.0 - summary["CCI"])

    pos = (
        df.groupby(["model_name", "dataset", "temperature", "perturbation_position"], as_index=False)
        .agg(CCI_position=("collapse", "mean"))
        .pivot(index=["model_name", "dataset", "temperature"], columns="perturbation_position", values="CCI_position")
        .reset_index()
        .rename_axis(columns=None)
    )

    summary = summary.merge(pos, on=["model_name", "dataset", "temperature"], how="left")
    summary.to_csv(output_dir / "summary_by_model_dataset.csv", index=False)

    hyp = hypothesis_tests(df)
    (output_dir / "hypothesis_tests.json").write_text(json.dumps(hyp, indent=2), encoding="utf-8")


def hypothesis_tests(df: pd.DataFrame) -> dict[str, Any]:
    results: dict[str, Any] = {}

    # H2: decoupling between answer robustness and trajectory consistency.
    results["H2_decoupling"] = {
        "mean_delta_decouple": float((df["apa"].astype(float) - (1.0 - df["collapse"]).astype(float)).mean())
    }

    # H3: middle perturbations cause larger collapse than beginning/end.
    by_pos = df.groupby("perturbation_position")["collapse"].mean().to_dict()
    results["H3_position_means"] = {k: float(v) for k, v in by_pos.items()}

    mid = df[df["perturbation_position"] == "middle"]["collapse"].to_numpy()
    beg = df[df["perturbation_position"] == "beginning"]["collapse"].to_numpy()
    end = df[df["perturbation_position"] == "end"]["collapse"].to_numpy()

    n = min(len(mid), len(beg), len(end))
    if n > 2:
        t_mb = ttest_rel(mid[:n], beg[:n], alternative="greater")
        t_me = ttest_rel(mid[:n], end[:n], alternative="greater")
        results["H3_tests"] = {
            "middle_gt_beginning": {"t": float(t_mb.statistic), "p": float(t_mb.pvalue)},
            "middle_gt_end": {"t": float(t_me.statistic), "p": float(t_me.pvalue)},
        }
    else:
        results["H3_tests"] = {"note": "Not enough paired samples for t-tests."}

    return results

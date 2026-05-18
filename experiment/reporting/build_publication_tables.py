from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = float("nan")
    return out


def _fmt_pct(x: float) -> float:
    return round(100.0 * float(x), 2)


def _fmt_float(x: float) -> float:
    return round(float(x), 4)


def build_main_results(summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    cols = [
        "model_name",
        "dataset",
        "temperature",
        "FAA_clean",
        "FAA_perturbed",
        "CCI",
        "SD",
        "RA",
        "FR",
        "Delta_decouple",
    ]
    summary = _ensure_columns(summary, cols)
    t = summary[cols].copy()

    t = t.rename(
        columns={
            "model_name": "Model",
            "dataset": "Benchmark",
            "temperature": "Temp",
            "FAA_clean": "Clean FAA (%)",
            "FAA_perturbed": "Perturbed FAA (%)",
            "CCI": "RC/CCI (down)",
            "SD": "SD (down)",
            "RA": "RA (%)",
            "FR": "FR (down)",
            "Delta_decouple": "Delta_decouple",
        }
    )

    t["Clean FAA (%)"] = t["Clean FAA (%)"].map(_fmt_pct)
    t["Perturbed FAA (%)"] = t["Perturbed FAA (%)"].map(_fmt_pct)
    t["RA (%)"] = t["RA (%)"].map(_fmt_pct)
    for c in ["RC/CCI (down)", "SD (down)", "FR (down)", "Delta_decouple"]:
        t[c] = t[c].map(_fmt_float)

    t = t.sort_values(["Model", "Benchmark", "Temp"]).reset_index(drop=True)

    csv_path = out_dir / "table_main_results.csv"
    tex_path = out_dir / "table_main_results.tex"
    t.to_csv(csv_path, index=False)
    t.to_latex(tex_path, index=False, escape=False)
    return t


def build_ablation(per_example: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    grouped = (
        per_example.groupby(["model_name", "perturbation_type"], as_index=False)
        .agg(CCI=("collapse", "mean"))
        .pivot(index="model_name", columns="perturbation_type", values="CCI")
        .reset_index()
        .rename_axis(columns=None)
        .rename(columns={"model_name": "Model"})
    )

    for c in grouped.columns:
        if c != "Model":
            grouped[c] = grouped[c].map(_fmt_float)

    grouped = grouped.sort_values("Model").reset_index(drop=True)

    csv_path = out_dir / "table_ablation.csv"
    tex_path = out_dir / "table_ablation.tex"
    grouped.to_csv(csv_path, index=False)
    grouped.to_latex(tex_path, index=False, escape=False)
    return grouped


def build_position_effect(per_example: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    pos = (
        per_example.groupby(["model_name", "perturbation_position"], as_index=False)
        .agg(CCI=("collapse", "mean"))
        .pivot(index="model_name", columns="perturbation_position", values="CCI")
        .reset_index()
        .rename_axis(columns=None)
    )

    pos = pos.rename(
        columns={
            "model_name": "Model",
            "beginning": "CCI_beginning",
            "middle": "CCI_middle",
            "end": "CCI_end",
        }
    )

    pos = _ensure_columns(pos, ["Model", "CCI_beginning", "CCI_middle", "CCI_end"])

    for c in ["CCI_beginning", "CCI_middle", "CCI_end"]:
        pos[c] = pos[c].map(_fmt_float)

    pos = pos.sort_values("Model").reset_index(drop=True)

    csv_path = out_dir / "table_position_effect.csv"
    tex_path = out_dir / "table_position_effect.tex"
    pos.to_csv(csv_path, index=False)
    pos.to_latex(tex_path, index=False, escape=False)
    return pos


def build_hypothesis_brief(hyp_path: Path, out_dir: Path) -> dict:
    if hyp_path.exists():
        data = json.loads(hyp_path.read_text(encoding="utf-8"))
    else:
        data = {"note": "hypothesis_tests.json not found"}

    out_path = out_dir / "hypothesis_brief.json"
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-ready experiment tables.")
    parser.add_argument("--output-dir", required=True, help="Experiment output directory (contains CSV metrics).")
    parser.add_argument(
        "--report-dir",
        default="",
        help="Directory for generated publication tables. Defaults to <output-dir>/publication_tables.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir) if args.report_dir else output_dir / "publication_tables"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary_by_model_dataset.csv"
    per_example_path = output_dir / "per_example_metrics.csv"
    hyp_path = output_dir / "hypothesis_tests.json"

    if not summary_path.exists() or not per_example_path.exists():
        raise FileNotFoundError("Required files missing. Run experiment first to generate summary and per-example metrics.")

    summary = pd.read_csv(summary_path)
    per_example = pd.read_csv(per_example_path)

    build_main_results(summary, report_dir)
    build_ablation(per_example, report_dir)
    build_position_effect(per_example, report_dir)
    build_hypothesis_brief(hyp_path, report_dir)

    print(f"Publication tables generated in: {report_dir}")


if __name__ == "__main__":
    main()

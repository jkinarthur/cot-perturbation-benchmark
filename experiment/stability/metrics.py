from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable


@dataclass
class MetricBundle:
    faa_clean: float
    faa_perturbed: float
    apa: float
    cci: float
    semantic_drift: float
    recovery_ability: float
    faithfulness_risk: float


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def answer_is_correct(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def trajectory_similarity(clean_cot: str, perturbed_cot: str) -> float:
    a = token_set(clean_cot)
    b = token_set(perturbed_cot)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def semantic_drift_score(question: str, perturbed_cot: str) -> float:
    q = token_set(question)
    c = token_set(perturbed_cot)
    if not q:
        return 0.0
    retained = len(q & c) / len(q)
    return 1.0 - retained


def recovery_success(perturbation_type: str, perturbed_cot: str, pred: str, gold: str) -> float:
    if perturbation_type not in {"intermediate_wrong_step", "contradictory_evidence"}:
        return math.nan
    repaired_signal = any(k in perturbed_cot.lower() for k in ["correct", "revise", "instead", "fix"])
    return 1.0 if repaired_signal and answer_is_correct(pred, gold) else 0.0


def faithfulness_risk_score(perturbed_cot: str, pred: str) -> float:
    pred_toks = token_set(pred)
    cot_toks = token_set(perturbed_cot)
    if not pred_toks:
        return 1.0
    support = len(pred_toks & cot_toks) / len(pred_toks)
    return 1.0 - support


def mean_ignore_nan(vals: Iterable[float]) -> float:
    xs = [v for v in vals if not math.isnan(v)]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)

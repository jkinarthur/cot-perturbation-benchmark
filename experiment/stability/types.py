from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PerturbationType(str, Enum):
    LEXICAL_PARAPHRASE = "lexical_paraphrase"
    SEMANTIC_DISTRACTOR = "semantic_distractor"
    INTERMEDIATE_WRONG_STEP = "intermediate_wrong_step"
    SKIPPED_STEP = "skipped_step"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    UNIT_VALUE_CHANGE = "unit_value_change"


class PerturbationPosition(str, Enum):
    BEGINNING = "beginning"
    MIDDLE = "middle"
    END = "end"


@dataclass(frozen=True)
class Example:
    example_id: str
    dataset: str
    question: str
    gold_answer: str
    cot: str = ""
    image_path: Optional[str] = None


@dataclass(frozen=True)
class PerturbationSpec:
    perturbation_type: PerturbationType
    position: PerturbationPosition


@dataclass
class ModelOutput:
    answer: str
    cot: str
    raw_text: str


@dataclass
class RunRecord:
    model_name: str
    dataset: str
    example_id: str
    temperature: float
    perturbation_type: str
    perturbation_position: str
    perturbed_question: str
    gold_answer: str
    clean_answer: str
    clean_cot: str
    perturbed_answer: str
    perturbed_cot: str

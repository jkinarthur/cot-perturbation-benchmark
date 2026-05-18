from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .types import Example


def load_examples_for_benchmark(path: Path, dataset_name: str, max_examples: int, category: str = "") -> list[Example]:
    rows = _load_jsonl(path, max_examples)
    lname = dataset_name.lower()

    if "gsm8k" in lname:
        return [_parse_gsm8k(r, dataset_name, i) for i, r in enumerate(rows)]
    if "strategyqa" in lname:
        return [_parse_strategyqa(r, dataset_name, i) for i, r in enumerate(rows)]
    if "commonsenseqa" in lname or "csqa" in lname:
        return [_parse_commonsenseqa(r, dataset_name, i) for i, r in enumerate(rows)]
    if "mmcot" in lname or category == "multimodal":
        return [_parse_mmcot(r, dataset_name, i) for i, r in enumerate(rows)]

    return [_parse_generic(r, dataset_name, i) for i, r in enumerate(rows)]


def _load_jsonl(path: Path, max_examples: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= max_examples:
                break
            items.append(json.loads(line))
    return items


def _parse_gsm8k(raw: dict[str, Any], dataset: str, idx: int) -> Example:
    q = str(raw.get("question", raw.get("problem", ""))).strip()
    ans_raw = str(raw.get("gold_answer", raw.get("answer", raw.get("final_answer", ""))))
    gold = _extract_gsm8k_answer(ans_raw)
    return Example(
        example_id=str(raw.get("id", raw.get("question_id", f"{dataset}_{idx:06d}"))),
        dataset=dataset,
        question=q,
        gold_answer=gold,
        cot=str(raw.get("cot", raw.get("rationale", ""))),
        image_path=raw.get("image_path"),
    )


def _parse_strategyqa(raw: dict[str, Any], dataset: str, idx: int) -> Example:
    q = str(raw.get("question", raw.get("input", ""))).strip()
    ans = raw.get("gold_answer", raw.get("answer", raw.get("target", "")))
    gold = _normalize_bool_answer(ans)
    return Example(
        example_id=str(raw.get("id", raw.get("qid", f"{dataset}_{idx:06d}"))),
        dataset=dataset,
        question=q,
        gold_answer=gold,
        cot=str(raw.get("cot", raw.get("facts", ""))),
        image_path=raw.get("image_path"),
    )


def _parse_commonsenseqa(raw: dict[str, Any], dataset: str, idx: int) -> Example:
    question_obj = raw.get("question", {})
    stem = str(question_obj.get("stem", raw.get("question", ""))).strip()

    choices = question_obj.get("choices", raw.get("choices", []))
    choice_lines: list[str] = []
    if isinstance(choices, list):
        for c in choices:
            if isinstance(c, dict):
                label = c.get("label", "")
                text = c.get("text", "")
                choice_lines.append(f"({label}) {text}")
            else:
                choice_lines.append(str(c))

    question_text = stem
    if choice_lines:
        question_text += "\nOptions: " + " | ".join(choice_lines)

    gold = str(raw.get("gold_answer", raw.get("answerKey", raw.get("answer", "")))).strip()
    return Example(
        example_id=str(raw.get("id", raw.get("question_id", f"{dataset}_{idx:06d}"))),
        dataset=dataset,
        question=question_text,
        gold_answer=gold,
        cot=str(raw.get("cot", raw.get("rationale", ""))),
        image_path=raw.get("image_path"),
    )


def _parse_mmcot(raw: dict[str, Any], dataset: str, idx: int) -> Example:
    q = str(raw.get("question", raw.get("query", ""))).strip()
    image_path = raw.get("image_path", raw.get("image", raw.get("image_file", None)))
    gold = str(raw.get("gold_answer", raw.get("answer", raw.get("target", "")))).strip()
    return Example(
        example_id=str(raw.get("id", f"{dataset}_{idx:06d}")),
        dataset=dataset,
        question=q,
        gold_answer=gold,
        cot=str(raw.get("cot", raw.get("rationale", ""))),
        image_path=image_path,
    )


def _parse_generic(raw: dict[str, Any], dataset: str, idx: int) -> Example:
    return Example(
        example_id=str(raw.get("id", f"{dataset}_{idx:06d}")),
        dataset=dataset,
        question=str(raw.get("question", "")).strip(),
        gold_answer=str(raw.get("gold_answer", raw.get("answer", ""))).strip(),
        cot=str(raw.get("cot", "")),
        image_path=raw.get("image_path"),
    )


def _extract_gsm8k_answer(ans_raw: str) -> str:
    # GSM8K often stores final answer after #### marker.
    if "####" in ans_raw:
        return ans_raw.split("####")[-1].strip()

    m = re.search(r"-?\d+(?:[\.,]\d+)?", ans_raw)
    if m:
        return m.group(0).replace(",", "")
    return ans_raw.strip()


def _normalize_bool_answer(ans: Any) -> str:
    s = str(ans).strip().lower()
    if s in {"true", "yes", "1"}:
        return "yes"
    if s in {"false", "no", "0"}:
        return "no"
    return s

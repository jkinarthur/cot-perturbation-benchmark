from __future__ import annotations

import re
from dataclasses import dataclass

from .types import PerturbationPosition, PerturbationSpec, PerturbationType


@dataclass
class PerturbationResult:
    text: str
    injected_marker: str


class PerturbationEngine:
    """Applies controlled semantic perturbations with explicit injection position."""

    _LEXICAL_MAP = {
        "therefore": "thus",
        "because": "since",
        "purchase": "buy",
        "calculate": "compute",
        "show": "demonstrate",
    }

    def apply(self, question: str, spec: PerturbationSpec) -> PerturbationResult:
        base = question.strip()
        if spec.perturbation_type == PerturbationType.LEXICAL_PARAPHRASE:
            rewritten = self._lexical_paraphrase(base)
            return PerturbationResult(self._insert_by_position(base, rewritten, spec.position), "lexical")

        if spec.perturbation_type == PerturbationType.SEMANTIC_DISTRACTOR:
            distractor = "Context note: A nearby city opened a new museum last year."
            return PerturbationResult(self._inject_segment(base, distractor, spec.position), "distractor")

        if spec.perturbation_type == PerturbationType.INTERMEDIATE_WRONG_STEP:
            wrong_step = "Interim claim: adding both sides gives 17 before simplification."
            return PerturbationResult(self._inject_segment(base, wrong_step, spec.position), "wrong_step")

        if spec.perturbation_type == PerturbationType.SKIPPED_STEP:
            shortened = self._drop_middle_sentence(base)
            return PerturbationResult(shortened, "skipped_step")

        if spec.perturbation_type == PerturbationType.CONTRADICTORY_EVIDENCE:
            contradiction = "Additional statement: one source now says the opposite condition is true."
            return PerturbationResult(self._inject_segment(base, contradiction, spec.position), "contradiction")

        if spec.perturbation_type == PerturbationType.UNIT_VALUE_CHANGE:
            changed = self._unit_or_value_change(base)
            return PerturbationResult(changed, "unit_or_value")

        return PerturbationResult(base, "none")

    def _lexical_paraphrase(self, text: str) -> str:
        out = text
        for src, dst in self._LEXICAL_MAP.items():
            out = re.sub(rf"\\b{re.escape(src)}\\b", dst, out, flags=re.IGNORECASE)
        return out

    def _unit_or_value_change(self, text: str) -> str:
        out = re.sub(r"\\bkm\\b", "m", text)
        out = re.sub(r"\\bmeters\\b", "kilometers", out, flags=re.IGNORECASE)

        def bump(match: re.Match[str]) -> str:
            val = int(match.group(0))
            return str(val + 1)

        return re.sub(r"\\b\\d+\\b", bump, out, count=1)

    def _drop_middle_sentence(self, text: str) -> str:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\\s+", text) if p.strip()]
        if len(parts) < 3:
            return text
        mid = len(parts) // 2
        return " ".join(parts[:mid] + parts[mid + 1 :])

    def _inject_segment(self, text: str, segment: str, position: PerturbationPosition) -> str:
        if position == PerturbationPosition.BEGINNING:
            return f"{segment} {text}"
        if position == PerturbationPosition.END:
            return f"{text} {segment}"
        return self._insert_middle(text, segment)

    def _insert_by_position(self, original: str, rewritten: str, position: PerturbationPosition) -> str:
        if position == PerturbationPosition.BEGINNING:
            return rewritten
        if position == PerturbationPosition.END:
            return rewritten
        return rewritten

    def _insert_middle(self, text: str, segment: str) -> str:
        words = text.split()
        if len(words) < 8:
            return f"{text} {segment}"
        idx = len(words) // 2
        return " ".join(words[:idx] + [segment] + words[idx:])

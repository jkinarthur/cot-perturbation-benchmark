from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Protocol

import requests

from .types import ModelOutput


class ModelAdapter(Protocol):
    def generate(self, question: str, temperature: float, image_path: str | None = None) -> ModelOutput:
        ...


@dataclass
class MockModelAdapter:
    name: str

    def generate(self, question: str, temperature: float, image_path: str | None = None) -> ModelOutput:
        cot = (
            "Step 1: parse constraints. "
            "Step 2: reason over entities. "
            "Step 3: produce final answer."
        )
        answer = "mock_answer"
        return ModelOutput(answer=answer, cot=cot, raw_text=f"CoT: {cot}\nFinalAnswer: {answer}")


@dataclass
class OpenAICompatibleAdapter:
    name: str
    endpoint: str
    api_key_env: str

    def generate(self, question: str, temperature: float, image_path: str | None = None) -> ModelOutput:
        api_key = os.getenv(self.api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {self.api_key_env}")

        user_payload = question
        if image_path:
            user_payload += f"\n[image_path={image_path}]"

        payload = {
            "model": self.name,
            "temperature": temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Provide concise reasoning and end with 'FinalAnswer: <value>'. "
                        "CoT is an observable reasoning trace and may be imperfect."
                    ),
                },
                {"role": "user", "content": user_payload},
            ],
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = requests.post(self.endpoint, headers=headers, data=json.dumps(payload), timeout=120)
        resp.raise_for_status()
        data = resp.json()

        text = data["choices"][0]["message"]["content"]
        answer = self._extract_final_answer(text)
        cot = self._extract_cot(text)
        return ModelOutput(answer=answer, cot=cot, raw_text=text)

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        marker = "FinalAnswer:"
        if marker not in text:
            return text.strip().splitlines()[-1].strip()
        return text.split(marker, 1)[1].strip().splitlines()[0].strip()

    @staticmethod
    def _extract_cot(text: str) -> str:
        marker = "FinalAnswer:"
        if marker in text:
            return text.split(marker, 1)[0].strip()
        return text.strip()

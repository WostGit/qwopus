"""Thin adapter around upstream cisnlp/multypo.

This module intentionally delegates typo generation to the upstream MulTypo
implementation (keyboard-aware perturbations). It only normalizes invocation for
CI reliability across minor API shape differences.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MulTypoAdapter:
    """Wrapper that always uses upstream MulTypo for English keyboard typos."""

    language: str = "english"
    typo_rate: float = 0.2

    def __post_init__(self) -> None:
        try:
            from multypo import MultiTypoGenerator  # type: ignore

            self._generator = MultiTypoGenerator(language=self.language)
            self._mode = "generator"
        except Exception:
            from multypo import generate_typos  # type: ignore

            self._generate_typos = generate_typos
            self._mode = "function"

    def perturb(self, text: str) -> str:
        """Generate typos via upstream MulTypo (no local typo logic)."""
        if self._mode == "generator":
            return self._generator.insert_typos_in_text(text=text, typo_rate=self.typo_rate)
        return self._generate_typos(
            text=text,
            language=self.language,
            typo_rate=self.typo_rate,
        )

#!/usr/bin/env python3
"""CI-friendly typo robustness harness using upstream MulTypo.

This script intentionally uses `multypo.generate_typos` from the public
`cisnlp/multypo` package for English keyboard-aware perturbations.
It does not reimplement typo logic locally.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from multypo import generate_typos


@dataclass
class Example:
    id: str
    prompt: str
    target: str


def load_examples(path: Path, sample_size: int) -> list[Example]:
    examples: list[Example] = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            examples.append(Example(id=row["id"], prompt=row["prompt"], target=str(row["target"])))
            if len(examples) >= sample_size:
                break
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def run_llama_cli(llama_cli: Path, model: Path, prompt: str, threads: int, seed: int) -> str:
    cmd = [
        str(llama_cli),
        "-m",
        str(model),
        "--no-cnv",
        "--log-disable",
        "--seed",
        str(seed),
        "--temp",
        "0",
        "--top-k",
        "1",
        "-n",
        "8",
        "-t",
        str(threads),
        "-ngl",
        "0",
        "-p",
        prompt,
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def extract_digits(text: str) -> str:
    m = re.search(r"(-?\d+)(?!.*-?\d)", text)
    return m.group(1) if m else ""


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llama-cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=120)
    parser.add_argument("--typo-rate", type=float, default=0.2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.input, args.sample_size)

    perturbed_rows: list[dict] = []
    raw_rows: list[dict] = []
    scored_rows: list[dict] = []

    clean_correct = 0
    typo_correct = 0

    for idx, ex in enumerate(examples, start=1):
        typo_prompt = generate_typos(text=ex.prompt, language="english", typo_rate=args.typo_rate)
        perturbed_rows.append({"id": ex.id, "clean_prompt": ex.prompt, "typo_prompt": typo_prompt, "target": ex.target})

        clean_out = run_llama_cli(args.llama_cli, args.model, ex.prompt, args.threads, args.seed)
        typo_out = run_llama_cli(args.llama_cli, args.model, typo_prompt, args.threads, args.seed)

        clean_pred = extract_digits(clean_out)
        typo_pred = extract_digits(typo_out)

        clean_ok = clean_pred == ex.target
        typo_ok = typo_pred == ex.target

        clean_correct += int(clean_ok)
        typo_correct += int(typo_ok)

        raw_rows.extend(
            [
                {"id": ex.id, "variant": "clean", "prompt": ex.prompt, "raw_output": clean_out},
                {"id": ex.id, "variant": "typo", "prompt": typo_prompt, "raw_output": typo_out},
            ]
        )
        scored_rows.append(
            {
                "id": ex.id,
                "target": ex.target,
                "clean_pred": clean_pred,
                "typo_pred": typo_pred,
                "clean_correct": clean_ok,
                "typo_correct": typo_ok,
            }
        )

        if idx % 10 == 0 or idx == len(examples):
            print(
                f"[progress] {idx}/{len(examples)} "
                f"clean_acc={clean_correct/idx:.3f} typo_acc={typo_correct/idx:.3f}",
                flush=True,
            )

    clean_acc = clean_correct / len(examples)
    typo_acc = typo_correct / len(examples)
    robustness_drop = clean_acc - typo_acc

    write_jsonl(args.output_dir / "perturbed_prompts.jsonl", perturbed_rows)
    write_jsonl(args.output_dir / "raw_generations.jsonl", raw_rows)
    write_jsonl(args.output_dir / "scored_results.jsonl", scored_rows)

    with (args.output_dir / "aggregate.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["samples", "clean_accuracy", "typo_accuracy", "robustness_drop"])
        writer.writeheader()
        writer.writerow(
            {
                "samples": len(examples),
                "clean_accuracy": f"{clean_acc:.6f}",
                "typo_accuracy": f"{typo_acc:.6f}",
                "robustness_drop": f"{robustness_drop:.6f}",
            }
        )

    report = f"""# MulTypo Robustness Report

- Upstream typo generator: `multypo.generate_typos` (cisnlp/multypo)
- Language: english
- Samples: {len(examples)}
- Typo rate: {args.typo_rate}
- Deterministic decoding: `temp=0`, `top_k=1`, fixed `seed={args.seed}`

| Metric | Value |
|---|---:|
| Clean accuracy | {clean_acc:.4f} |
| Typo accuracy | {typo_acc:.4f} |
| Robustness drop (clean - typo) | {robustness_drop:.4f} |
"""
    report_path = args.output_dir / "report.md"
    report_path.write_text(report)

    print("\n=== ROBUSTNESS SUMMARY ===")
    print(report)


if __name__ == "__main__":
    main()

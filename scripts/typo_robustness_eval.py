#!/usr/bin/env python3
"""CI-sized typo robustness evaluation harness for llama.cpp.

Typo generation is sourced from upstream `cisnlp/multypo` via the public
`multypo.generate_typos` API (keyboard-aware perturbations for English).
This script only provides a thin local evaluation harness and does NOT
re-implement the typo logic.
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
class PromptItem:
    id: str
    prompt: str
    answer: str
    task: str


def load_dataset(path: Path, limit: int) -> list[PromptItem]:
    items: list[PromptItem] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(PromptItem(**row))
            if len(items) >= limit:
                break
    if not items:
        raise ValueError(f"No prompts found in {path}")
    return items


def run_llama_cli(llama_cli: Path, model: Path, prompt: str, n_predict: int, seed: int) -> str:
    cmd = [
        str(llama_cli),
        "-m",
        str(model),
        "-p",
        prompt,
        "-n",
        str(n_predict),
        "--temp",
        "0",
        "--top-k",
        "1",
        "--seed",
        str(seed),
        "--no-display-prompt",
        "-ngl",
        "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = proc.stdout.strip()
    # llama-cli often appends perf lines on stderr; stdout is mainly generation.
    return out


def extract_answer(text: str) -> str:
    match = re.search(r"-?\d+", text)
    return match.group(0) if match else ""


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llama-cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=180)
    parser.add_argument("--typo-rate", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n-predict", type=int, default=16)
    args = parser.parse_args()

    if not (100 <= args.limit <= 300):
        raise ValueError("--limit must be within 100..300 for CI-sized benchmark")

    random.seed(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_dataset(args.dataset, args.limit)

    perturbed_rows = []
    raw_rows = []
    score_rows = []

    clean_correct = 0
    perturbed_correct = 0

    print(f"Loaded {len(prompts)} clean prompts")
    print("Generating keyboard-aware perturbations via upstream multypo.generate_typos(language='english')")

    for idx, item in enumerate(prompts, start=1):
        perturbed_prompt = generate_typos(
            text=item.prompt,
            language="english",
            typo_rate=args.typo_rate,
            sentence_tokenize=False,
            use_gpu=False,
        )

        perturbed_rows.append(
            {
                "id": item.id,
                "task": item.task,
                "clean_prompt": item.prompt,
                "perturbed_prompt": perturbed_prompt,
                "answer": item.answer,
                "typo_rate": args.typo_rate,
            }
        )

        clean_out = run_llama_cli(args.llama_cli, args.model, item.prompt, args.n_predict, args.seed)
        typo_out = run_llama_cli(args.llama_cli, args.model, perturbed_prompt, args.n_predict, args.seed)

        clean_pred = extract_answer(clean_out)
        typo_pred = extract_answer(typo_out)

        clean_ok = clean_pred == item.answer
        typo_ok = typo_pred == item.answer

        clean_correct += int(clean_ok)
        perturbed_correct += int(typo_ok)

        raw_rows.append(
            {
                "id": item.id,
                "clean_output": clean_out,
                "perturbed_output": typo_out,
            }
        )
        score_rows.append(
            {
                "id": item.id,
                "task": item.task,
                "expected": item.answer,
                "clean_pred": clean_pred,
                "perturbed_pred": typo_pred,
                "clean_correct": clean_ok,
                "perturbed_correct": typo_ok,
            }
        )

        print(
            f"[{idx:03d}/{len(prompts)}] {item.id} | exp={item.answer} | "
            f"clean={clean_pred} ({'OK' if clean_ok else 'FAIL'}) | "
            f"typo={typo_pred} ({'OK' if typo_ok else 'FAIL'})"
        )

    total = len(prompts)
    clean_acc = clean_correct / total
    typo_acc = perturbed_correct / total
    drop = clean_acc - typo_acc

    perturbed_path = out_dir / "perturbed_prompts.jsonl"
    raw_path = out_dir / "raw_generations.jsonl"
    scored_path = out_dir / "scored_results.jsonl"
    aggregate_csv = out_dir / "aggregate.csv"
    report_md = out_dir / "robustness_report.md"

    write_jsonl(perturbed_path, perturbed_rows)
    write_jsonl(raw_path, raw_rows)
    write_jsonl(scored_path, score_rows)

    with aggregate_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "total_prompts",
                "clean_correct",
                "perturbed_correct",
                "clean_accuracy",
                "perturbed_accuracy",
                "robustness_drop",
                "typo_rate",
                "generator_source",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "total_prompts": total,
                "clean_correct": clean_correct,
                "perturbed_correct": perturbed_correct,
                "clean_accuracy": f"{clean_acc:.4f}",
                "perturbed_accuracy": f"{typo_acc:.4f}",
                "robustness_drop": f"{drop:.4f}",
                "typo_rate": args.typo_rate,
                "generator_source": "cisnlp/multypo::generate_typos(language='english')",
            }
        )

    report_md.write_text(
        "\n".join(
            [
                "# Typo Robustness Report",
                "",
                "## Upstream MulTypo component used",
                "- `cisnlp/multypo` Python package",
                "- API: `multypo.generate_typos(..., language=\"english\")`",
                "",
                "## Aggregate Metrics",
                f"- Total prompts: **{total}**",
                f"- Clean accuracy: **{clean_acc:.2%}** ({clean_correct}/{total})",
                f"- Perturbed accuracy: **{typo_acc:.2%}** ({perturbed_correct}/{total})",
                f"- Robustness drop: **{drop:.2%}**",
                f"- Typo rate: **{args.typo_rate:.2f}**",
            ]
        ),
        encoding="utf-8",
    )

    print("\n=== TYPO ROBUSTNESS SUMMARY ===")
    print(f"Total prompts        : {total}")
    print(f"Clean correct        : {clean_correct}")
    print(f"Perturbed correct    : {perturbed_correct}")
    print(f"Clean accuracy       : {clean_acc:.4f}")
    print(f"Perturbed accuracy   : {typo_acc:.4f}")
    print(f"Robustness drop      : {drop:.4f}")
    print(f"Wrote: {perturbed_path}")
    print(f"Wrote: {raw_path}")
    print(f"Wrote: {scored_path}")
    print(f"Wrote: {aggregate_csv}")
    print(f"Wrote: {report_md}")


if __name__ == "__main__":
    main()

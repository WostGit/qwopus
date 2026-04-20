#!/usr/bin/env python3
"""CI-sized typo robustness harness using upstream MulTypo.

Typos are generated with the upstream `cisnlp/multypo` implementation
(via the `multypo` Python package), not a custom typo engine.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
from pathlib import Path
from typing import Dict, List

from multypo import generate_typos


NUM_RE = re.compile(r"[-+]?\d+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate typo robustness with MulTypo perturbations")
    p.add_argument("--dataset", required=True, help="JSONL dataset with prompt/answer fields")
    p.add_argument("--llama-cli", required=True, help="Path to llama-cli binary")
    p.add_argument("--model", required=True, help="Path to GGUF model")
    p.add_argument("--output-dir", required=True, help="Directory for artifacts")
    p.add_argument("--typo-rate", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--max-prompts", type=int, default=120)
    return p.parse_args()


def normalize_prediction(text: str) -> str:
    m = NUM_RE.search(text.strip())
    return m.group(0) if m else text.strip().splitlines()[0].strip()


def run_llama(llama_cli: str, model: str, prompt: str) -> str:
    cmd = [
        llama_cli,
        "-m",
        model,
        "-p",
        prompt,
        "-n",
        "8",
        "--temp",
        "0",
        "--top-k",
        "1",
        "--seed",
        "42",
        "--ctx-size",
        "512",
        "--repeat-penalty",
        "1.0",
        "--no-display-prompt",
    ]
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return res.stdout.strip()


def load_dataset(path: Path, max_prompts: int) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if i >= max_prompts:
                break
            rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    dataset = load_dataset(Path(args.dataset), args.max_prompts)

    perturbed_file = out_dir / "perturbed_prompts.jsonl"
    raw_file = out_dir / "raw_generations.jsonl"
    scored_file = out_dir / "scored_results.jsonl"
    aggregate_file = out_dir / "aggregate.csv"
    report_file = out_dir / "report.md"

    clean_correct = 0
    typo_correct = 0

    with perturbed_file.open("w") as pf, raw_file.open("w") as rf, scored_file.open("w") as sf:
        for row in dataset:
            prompt = row["prompt"]
            answer = str(row["answer"]).strip()
            perturbed_prompt = generate_typos(
                text=prompt,
                language="english",
                typo_rate=args.typo_rate,
                sentence_tokenize=False,
            )

            pf.write(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "task": row.get("task", "exact_match"),
                        "clean_prompt": prompt,
                        "perturbed_prompt": perturbed_prompt,
                        "answer": answer,
                        "generator": "cisnlp/multypo::generate_typos(language='english')",
                    }
                )
                + "\n"
            )

            clean_out = run_llama(args.llama_cli, args.model, prompt)
            typo_out = run_llama(args.llama_cli, args.model, perturbed_prompt)

            clean_pred = normalize_prediction(clean_out)
            typo_pred = normalize_prediction(typo_out)

            clean_ok = clean_pred == answer
            typo_ok = typo_pred == answer
            clean_correct += int(clean_ok)
            typo_correct += int(typo_ok)

            rf.write(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "answer": answer,
                        "clean": {"prompt": prompt, "raw_output": clean_out, "prediction": clean_pred},
                        "perturbed": {
                            "prompt": perturbed_prompt,
                            "raw_output": typo_out,
                            "prediction": typo_pred,
                        },
                    }
                )
                + "\n"
            )

            sf.write(
                json.dumps(
                    {
                        "id": row.get("id"),
                        "answer": answer,
                        "clean_correct": clean_ok,
                        "perturbed_correct": typo_ok,
                        "robustness_drop": int(clean_ok) - int(typo_ok),
                    }
                )
                + "\n"
            )

    total = len(dataset)
    clean_acc = clean_correct / total if total else 0.0
    typo_acc = typo_correct / total if total else 0.0
    drop = clean_acc - typo_acc

    with aggregate_file.open("w", newline="") as cf:
        writer = csv.DictWriter(
            cf,
            fieldnames=["total", "clean_correct", "typo_correct", "clean_accuracy", "typo_accuracy", "robustness_drop"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "total": total,
                "clean_correct": clean_correct,
                "typo_correct": typo_correct,
                "clean_accuracy": f"{clean_acc:.4f}",
                "typo_accuracy": f"{typo_acc:.4f}",
                "robustness_drop": f"{drop:.4f}",
            }
        )

    report = f"""# MulTypo Robustness Report

- Prompts evaluated: **{total}**
- Typo generator source: **cisnlp/multypo** (`generate_typos`, language=`english`)
- Typo rate: **{args.typo_rate}**
- Deterministic decode: temp=0, top-k=1, seed=42

## Accuracy

| Metric | Value |
|---|---:|
| Clean accuracy | {clean_acc:.4f} |
| Perturbed accuracy | {typo_acc:.4f} |
| Robustness drop | {drop:.4f} |
"""
    report_file.write_text(report)

    print("=" * 72)
    print("MulTypo robustness evaluation complete")
    print(f"Total prompts      : {total}")
    print(f"Clean correct      : {clean_correct}")
    print(f"Perturbed correct  : {typo_correct}")
    print(f"Clean accuracy     : {clean_acc:.4f}")
    print(f"Perturbed accuracy : {typo_acc:.4f}")
    print(f"Robustness drop    : {drop:.4f}")
    print("Artifacts:")
    for p in [perturbed_file, raw_file, scored_file, aggregate_file, report_file]:
        print(f" - {p}")
    print("=" * 72)


if __name__ == "__main__":
    main()

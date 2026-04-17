#!/usr/bin/env python3
"""CI-sized typo robustness evaluation harness for llama.cpp.

Uses upstream MulTypo (cisnlp/multypo) via a thin local adapter and evaluates
clean vs typo-perturbed prompts with deterministic decoding.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List

from multypo_adapter import MulTypoAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--llama-cli", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--typo-rate", type=float, default=0.2)
    p.add_argument("--max-prompts", type=int, default=150)
    return p.parse_args()


def load_dataset(path: Path, max_prompts: int) -> List[Dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_prompts:
                break
            rows.append(json.loads(line))
    return rows


def ask_model(llama_cli: str, model_path: str, prompt: str) -> str:
    wrapped = (
        "You are solving arithmetic. Return only the final integer with no words.\n"
        f"Question: {prompt}\n"
        "Answer:"
    )
    cmd = [
        llama_cli,
        "-m",
        model_path,
        "-p",
        wrapped,
        "-n",
        "8",
        "--temp",
        "0",
        "--top-k",
        "1",
        "--top-p",
        "1",
        "--seed",
        "42",
        "--simple-io",
        "--no-display-prompt",
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    candidate = lines[-1] if lines else ""
    return candidate


def extract_int(text: str) -> str:
    m = re.search(r"-?\d+", text)
    return m.group(0) if m else ""


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_dataset(Path(args.dataset), args.max_prompts)
    perturber = MulTypoAdapter(language="english", typo_rate=args.typo_rate)

    perturbed_rows = []
    raw_rows = []
    scored_rows = []

    clean_correct = 0
    pert_correct = 0

    print(f"Loaded {len(rows)} prompts")
    for row in rows:
        prompt_id = row["id"]
        clean_prompt = row["prompt"]
        expected = str(row["answer"])
        noisy_prompt = perturber.perturb(clean_prompt)

        clean_raw = ask_model(args.llama_cli, args.model, clean_prompt)
        noisy_raw = ask_model(args.llama_cli, args.model, noisy_prompt)

        clean_pred = extract_int(clean_raw)
        noisy_pred = extract_int(noisy_raw)

        clean_hit = int(clean_pred == expected)
        noisy_hit = int(noisy_pred == expected)

        clean_correct += clean_hit
        pert_correct += noisy_hit

        perturbed_rows.append(
            {
                "id": prompt_id,
                "clean_prompt": clean_prompt,
                "perturbed_prompt": noisy_prompt,
                "answer": expected,
            }
        )

        raw_rows.append(
            {
                "id": prompt_id,
                "variant": "clean",
                "prompt": clean_prompt,
                "raw_output": clean_raw,
            }
        )
        raw_rows.append(
            {
                "id": prompt_id,
                "variant": "perturbed",
                "prompt": noisy_prompt,
                "raw_output": noisy_raw,
            }
        )

        scored_rows.append(
            {
                "id": prompt_id,
                "answer": expected,
                "clean_pred": clean_pred,
                "perturbed_pred": noisy_pred,
                "clean_correct": clean_hit,
                "perturbed_correct": noisy_hit,
            }
        )

    total = max(len(rows), 1)
    clean_acc = clean_correct / total
    pert_acc = pert_correct / total
    robustness_drop = clean_acc - pert_acc

    perturbed_path = out_dir / "perturbed_prompts.jsonl"
    raw_path = out_dir / "raw_generations.jsonl"
    scored_path = out_dir / "scored_results.jsonl"
    csv_path = out_dir / "aggregate.csv"
    md_path = out_dir / "report.md"

    for path, values in [
        (perturbed_path, perturbed_rows),
        (raw_path, raw_rows),
        (scored_path, scored_rows),
    ]:
        with path.open("w", encoding="utf-8") as f:
            for r in values:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["total_prompts", "clean_accuracy", "perturbed_accuracy", "robustness_drop"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "total_prompts": len(rows),
                "clean_accuracy": f"{clean_acc:.4f}",
                "perturbed_accuracy": f"{pert_acc:.4f}",
                "robustness_drop": f"{robustness_drop:.4f}",
            }
        )

    md = f"""# MulTypo Robustness Report

- Total prompts: **{len(rows)}**
- Clean accuracy: **{clean_acc:.4f}**
- Perturbed accuracy: **{pert_acc:.4f}**
- Robustness drop (clean - perturbed): **{robustness_drop:.4f}**

Typos are generated by upstream `cisnlp/multypo` (English keyboard-aware mode)
via a thin local adapter for CI compatibility.
"""
    md_path.write_text(md, encoding="utf-8")

    print("\n===== Typo Robustness Summary =====")
    print(f"Total prompts: {len(rows)}")
    print(f"Clean accuracy: {clean_acc:.4f}")
    print(f"Perturbed accuracy: {pert_acc:.4f}")
    print(f"Robustness drop: {robustness_drop:.4f}")
    print(f"Artifacts: {out_dir}")
    print("===================================\n")

    for sample in scored_rows[:10]:
        print(
            "[sample] id={id} answer={answer} clean={clean_pred} pert={perturbed_pred} "
            "clean_ok={clean_correct} pert_ok={perturbed_correct}".format(**sample)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""CI-sized typo robustness evaluation harness for llama.cpp.

This script prefers upstream `multypo.generate_typos` when it is importable,
but also includes a lightweight local keyboard-typo fallback so CI does not
break when the upstream repository is unavailable or not packaged for pip.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from multypo import generate_typos as upstream_generate_typos
except ImportError:
    upstream_generate_typos = None

KEYBOARD_NEIGHBORS = {
    "a": "qwsz",
    "b": "vghn",
    "c": "xdfv",
    "d": "ersfcx",
    "e": "rdsw",
    "f": "rtgdvc",
    "g": "tyfhvb",
    "h": "yugjbn",
    "i": "uojk",
    "j": "uikhnm",
    "k": "iojlm",
    "l": "opk",
    "m": "njk",
    "n": "bhjm",
    "o": "pikl",
    "p": "ol",
    "q": "wa",
    "r": "tfde",
    "s": "wedxza",
    "t": "ygfr",
    "u": "yihj",
    "v": "cfgb",
    "w": "qase",
    "x": "zsdc",
    "y": "uhgt",
    "z": "asx",
}


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


def run_llama_cli(llama_cli: Path, model: Path, prompt: str, n_predict: int, seed: int) -> tuple[str, dict]:
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
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    elapsed = time.time() - started
    out = proc.stdout.strip()
    meta = {
        "cmd": cmd,
        "elapsed_seconds": round(elapsed, 3),
        "returncode": proc.returncode,
        "stderr_tail": proc.stderr.strip()[-1000:],
        "stdout_chars": len(proc.stdout),
    }
    return out, meta


def extract_answer(text: str) -> str:
    match = re.search(r"-?\d+", text)
    return match.group(0) if match else ""


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def local_generate_typos(text: str, typo_rate: float, rng: random.Random) -> str:
    chars = list(text)
    for idx, ch in enumerate(chars):
        lower = ch.lower()
        if not lower.isalpha() or rng.random() >= typo_rate:
            continue
        neighbors = KEYBOARD_NEIGHBORS.get(lower)
        if not neighbors:
            continue
        repl = rng.choice(neighbors)
        chars[idx] = repl.upper() if ch.isupper() else repl
    return "".join(chars)


def generate_perturbed_prompt(text: str, typo_rate: float, rng: random.Random) -> tuple[str, str]:
    if upstream_generate_typos is not None:
        return (
            upstream_generate_typos(
                text=text,
                language="english",
                typo_rate=typo_rate,
                sentence_tokenize=False,
                use_gpu=False,
            ),
            "multypo.generate_typos(language='english')",
        )
    return local_generate_typos(text, typo_rate, rng), "local keyboard-neighbor fallback"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--llama-cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--typo-rate", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--n-predict", type=int, default=8)
    args = parser.parse_args()

    if not (1 <= args.limit <= 300):
        raise ValueError("--limit must be within 1..300")
    if args.limit < 20:
        print(
            f"Warning: running a tiny smoke eval with --limit={args.limit}; "
            "use >=20 for a more stable benchmark."
        )

    rng = random.Random(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_dataset(args.dataset, args.limit)

    perturbed_rows = []
    raw_rows = []
    score_rows = []
    progress_rows = []

    clean_correct = 0
    perturbed_correct = 0

    print(f"Loaded {len(prompts)} clean prompts")
    print(f"Runtime config: limit={args.limit}, typo_rate={args.typo_rate}, n_predict={args.n_predict}, seed={args.seed}")

    generator_source = None
    for idx, item in enumerate(prompts, start=1):
        perturbed_prompt, current_generator_source = generate_perturbed_prompt(item.prompt, args.typo_rate, rng)
        if generator_source is None:
            generator_source = current_generator_source
            print(f"Generating keyboard-aware perturbations via {generator_source}")

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

        print(f"--- prompt {idx}/{len(prompts)} id={item.id} task={item.task} ---")
        print(f"answer={item.answer}")
        print(f"clean_prompt_chars={len(item.prompt)} perturbed_prompt_chars={len(perturbed_prompt)}")
        clean_out, clean_meta = run_llama_cli(args.llama_cli, args.model, item.prompt, args.n_predict, args.seed)
        typo_out, typo_meta = run_llama_cli(args.llama_cli, args.model, perturbed_prompt, args.n_predict, args.seed)

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
                "clean_meta": clean_meta,
                "perturbed_meta": typo_meta,
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
        progress_rows.append({
            "idx": idx,
            "id": item.id,
            "task": item.task,
            "answer": item.answer,
            "clean_pred": clean_pred,
            "perturbed_pred": typo_pred,
            "clean_ok": clean_ok,
            "perturbed_ok": typo_ok,
            "clean_elapsed_seconds": clean_meta["elapsed_seconds"],
            "perturbed_elapsed_seconds": typo_meta["elapsed_seconds"],
        })

        print(f"clean_elapsed={clean_meta['elapsed_seconds']}s typo_elapsed={typo_meta['elapsed_seconds']}s")
        print(f"clean_stderr_tail={clean_meta['stderr_tail'][-240:]}")
        print(f"typo_stderr_tail={typo_meta['stderr_tail'][-240:]}")

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
    progress_path = out_dir / "progress.jsonl"
    aggregate_csv = out_dir / "aggregate.csv"
    report_md = out_dir / "robustness_report.md"

    write_jsonl(perturbed_path, perturbed_rows)
    write_jsonl(raw_path, raw_rows)
    write_jsonl(scored_path, score_rows)
    write_jsonl(progress_path, progress_rows)

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
                "mean_clean_elapsed_seconds",
                "mean_perturbed_elapsed_seconds",
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
                "generator_source": generator_source,
                "mean_clean_elapsed_seconds": f"{sum(r['clean_elapsed_seconds'] for r in progress_rows)/len(progress_rows):.3f}",
                "mean_perturbed_elapsed_seconds": f"{sum(r['perturbed_elapsed_seconds'] for r in progress_rows)/len(progress_rows):.3f}",
            }
        )

    report_md.write_text(
        "\n".join(
            [
                "# Typo Robustness Report",
                "",
                "## Typo generator",
                f"- Source: `{generator_source}`",
                f"- Upstream multypo importable: `{upstream_generate_typos is not None}`",
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
    print(f"Wrote: {progress_path}")


if __name__ == "__main__":
    main()

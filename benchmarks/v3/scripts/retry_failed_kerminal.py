#!/usr/bin/env python3
"""Retry failed Kerminal evaluations from a previous KernelBench run."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
KERMINAL_SDK_SRC = Path("/workspace/kerminal/sdk/python/src")


def _configure_environment() -> None:
    api_key = os.environ.get("KERMINAL_API_KEY", "").strip()
    if not api_key:
        api_key = getpass.getpass("KERMINAL_API_KEY: ").strip()
    if not api_key:
        raise SystemExit("KERMINAL_API_KEY is required.")
    os.environ["KERMINAL_API_KEY"] = api_key

    if KERMINAL_SDK_SRC.exists():
        sdk_path = str(KERMINAL_SDK_SRC)
        pythonpath = os.environ.get("PYTHONPATH", "")
        parts = [p for p in pythonpath.split(os.pathsep) if p]
        if sdk_path not in parts:
            os.environ["PYTHONPATH"] = os.pathsep.join([sdk_path, *parts])
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)


def _load_results(run_dir: Path) -> list[dict[str, Any]]:
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        raise SystemExit(f"No results found: {results_path}")
    rows: list[dict[str, Any]] = []
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _parse_levels(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


def _selected_failures(
    rows: list[dict[str, Any]],
    *,
    levels: set[int] | None,
    problem_filter: str | None,
) -> list[dict[str, Any]]:
    selected = []
    seen: set[str] = set()
    for row in rows:
        problem = row.get("problem")
        if not problem or problem in seen:
            continue
        if row.get("correct"):
            continue
        if levels is not None and int(row.get("level", 0)) not in levels:
            continue
        if problem_filter and problem_filter not in problem:
            continue
        selected.append(row)
        seen.add(problem)
    return selected


def _problem_path_by_name(hardware: str) -> dict[str, tuple[int, Path]]:
    from src.hardware import get_target

    target = get_target(hardware)
    return {path.name: (level, path) for level, path in target.find_problems(REPO_ROOT)}


def _retry_run_dir(source_run_dir: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("outputs/batch_eval") / f"retry_{source_run_dir.name}_{timestamp}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Retry failed Kerminal tasks from a previous run")
    parser.add_argument("run_dir", help="Previous outputs/batch_eval/run_* directory")
    parser.add_argument("--hardware", default="h100_local")
    parser.add_argument("--model", default="kerminal/default")
    parser.add_argument("--levels", default=None, help="Comma-separated levels to retry, e.g. 1 or 1,4")
    parser.add_argument("--problem", default=None, help="Only retry failures whose filename contains this")
    parser.add_argument("--output-dir", default=None, help="Directory for retry results")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--max-time-seconds",
        type=int,
        default=None,
        help="Override per-problem agent timeout in seconds, e.g. 1800 for 30 minutes",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    source_run_dir = Path(args.run_dir)
    rows = _load_results(source_run_dir)
    levels = _parse_levels(args.levels)
    failures = _selected_failures(rows, levels=levels, problem_filter=args.problem)
    problem_map = _problem_path_by_name(args.hardware)

    missing = [row["problem"] for row in failures if row["problem"] not in problem_map]
    if missing:
        raise SystemExit(f"Problems not found for {args.hardware}: {', '.join(missing)}")

    print(f"Source run: {source_run_dir}")
    print(f"Selected failures: {len(failures)}")
    for row in failures:
        print(f"  L{row.get('level')} {row['problem']} | error={row.get('error')}")

    if args.dry_run or not failures:
        return 0

    _configure_environment()

    from src.batch import run_single_eval

    retry_dir = _retry_run_dir(source_run_dir, args.output_dir)
    retry_dir.mkdir(parents=True, exist_ok=True)
    results_path = retry_dir / "results.jsonl"

    print(f"\nRetry run directory: {retry_dir}")
    start_time = time.time()
    for index, row in enumerate(failures, 1):
        problem = row["problem"]
        level, problem_path = problem_map[problem]
        task_id = f"Kerminal Default_H100_{problem}".replace("/", "-").replace(" ", "_")
        turn_dir = retry_dir / "turns" / task_id
        result = run_single_eval(
            args.hardware,
            args.model,
            level,
            problem_path,
            max_turns=None,
            max_time_seconds=args.max_time_seconds,
            turn_artifact_dir=turn_dir,
            judge_model_key=args.judge_model,
        )
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), default=str) + "\n")
        print(f"Retry progress: {index}/{len(failures)}")

    elapsed_hours = (time.time() - start_time) / 3600
    print(f"\nRetry completed in {elapsed_hours:.2f} hours")
    print(f"Results saved to: {retry_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

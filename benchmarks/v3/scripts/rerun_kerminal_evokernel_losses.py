#!/usr/bin/env python3
"""Rerun Kerminal A100 tasks from the EvoKernel overlap report."""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_KERMINAL_BIN_DIR = Path.home() / ".local" / "bin"
DEFAULT_COMPARISON = (
    "artifacts/kerminal_a100/evokernel_comparison/comparison.csv"
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _kerminal_home() -> Path:
    return Path(os.environ.get("KERMINAL_HOME", Path.home() / ".kerminal")).expanduser()


def _has_kerminal_config_auth() -> bool:
    home = _kerminal_home()
    env_file = home / ".env"
    if env_file.exists() and "KERMINAL_API_KEY=" in env_file.read_text(encoding="utf-8", errors="ignore"):
        return True

    config = home / "config.toml"
    if not config.exists():
        return False
    config_text = config.read_text(encoding="utf-8", errors="ignore")
    return "experimental_bearer_token" in config_text or "api_key" in config_text


def _configure_environment() -> None:
    api_key = os.environ.get("KERMINAL_API_KEY", "").strip()
    has_config_auth = _has_kerminal_config_auth()
    if not api_key and not has_config_auth:
        api_key = getpass.getpass("KERMINAL_API_KEY: ").strip()
    if not api_key and not has_config_auth:
        raise SystemExit("KERMINAL_API_KEY is required unless ~/.kerminal config already has auth.")
    if api_key:
        os.environ["KERMINAL_API_KEY"] = api_key

    if KERMINAL_SDK_SRC.exists():
        sdk_path = str(KERMINAL_SDK_SRC)
        pythonpath = os.environ.get("PYTHONPATH", "")
        parts = [p for p in pythonpath.split(os.pathsep) if p]
        if sdk_path not in parts:
            os.environ["PYTHONPATH"] = os.pathsep.join([sdk_path, *parts])
        if sdk_path not in sys.path:
            sys.path.insert(0, sdk_path)

    if DEFAULT_KERMINAL_BIN_DIR.exists():
        path_parts = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
        bin_dir = str(DEFAULT_KERMINAL_BIN_DIR)
        if bin_dir not in path_parts:
            os.environ["PATH"] = os.pathsep.join([bin_dir, *path_parts])


def _resolve(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return REPO_ROOT / path


def _load_targets(
    comparison_path: Path,
    *,
    problem_filter: str | None = None,
    all_samples: bool = False,
) -> list[dict[str, Any]]:
    with comparison_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    targets = rows if all_samples else [row for row in rows if row.get("winner") != "Kerminal"]
    if problem_filter:
        targets = [row for row in targets if problem_filter in row.get("kerminal_problem", "")]
    return targets


def _problem_path_by_name(hardware: str) -> dict[str, tuple[int, Path]]:
    from src.hardware import get_target

    target = get_target(hardware)
    return {path.name: (level, path) for level, path in target.find_problems(REPO_ROOT)}


def _task_id(model_name: str, gpu_sku: str, problem: str) -> str:
    return f"{model_name}_{gpu_sku}_{problem}".replace("/", "-").replace(" ", "_")


def _output_dir(raw: str | None, *, all_samples: bool) -> Path:
    if raw:
        return Path(raw)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scope = "all_samples" if all_samples else "evokernel_losses"
    return Path("outputs/batch_eval") / f"rerun_{scope}_a100_turns5_1h_{timestamp}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", default=DEFAULT_COMPARISON)
    parser.add_argument("--hardware", default="a100_local")
    parser.add_argument("--model", default="kerminal/default")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--max-time-seconds", type=int, default=3600)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--problem", default=None, help="Only rerun problems whose filename contains this")
    parser.add_argument("--all-samples", action="store_true", help="Rerun every row in the comparison CSV, not only EvoKernel wins/no-speed rows")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    comparison_path = _resolve(args.comparison)
    targets = _load_targets(
        comparison_path,
        problem_filter=args.problem,
        all_samples=args.all_samples,
    )
    problem_map = _problem_path_by_name(args.hardware)
    missing = [row["kerminal_problem"] for row in targets if row["kerminal_problem"] not in problem_map]
    if missing:
        raise SystemExit(f"Problems not found for {args.hardware}: {', '.join(missing)}")

    print(f"Comparison: {comparison_path}")
    print(f"Hardware: {args.hardware}")
    print(f"Model: {args.model}")
    print(f"Max turns: {args.max_turns}")
    print(f"Max time seconds: {args.max_time_seconds}")
    print("Selection:", "all samples" if args.all_samples else "EvoKernel wins/no-speed only")
    print(f"Selected reruns: {len(targets)}")
    for row in targets:
        print(
            f"  L{row['level']} {row['kerminal_problem']} | "
            f"K={row.get('kerminal_speedup') or '--'} E={row.get('evokernel_speedup') or '--'} "
            f"winner={row.get('winner') or 'no-speed'}"
        )

    if args.dry_run or not targets:
        return 0

    _configure_environment()

    from src.batch import run_single_eval
    from src.hardware import get_target
    from src.models import get_model_config

    target = get_target(args.hardware)
    model_config = get_model_config(args.model)
    if model_config is None:
        raise SystemExit(f"Unknown model: {args.model}")
    output_dir = _output_dir(args.output_dir, all_samples=args.all_samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    manifest_path = output_dir / "rerun_manifest.json"
    manifest = {
        "comparison": str(comparison_path),
        "hardware": args.hardware,
        "model": args.model,
        "max_turns": args.max_turns,
        "max_time_seconds": args.max_time_seconds,
        "selection": "all_samples" if args.all_samples else "evokernel_losses",
        "selected_count": len(targets),
        "selected_problems": [row["kerminal_problem"] for row in targets],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nRerun directory: {output_dir}")
    start = time.time()
    for index, row in enumerate(targets, 1):
        problem = row["kerminal_problem"]
        level, problem_path = problem_map[problem]
        task_id = _task_id(model_config.name, target.gpu_sku, problem)
        turn_dir = output_dir / "turns" / task_id
        result = run_single_eval(
            args.hardware,
            args.model,
            level,
            problem_path,
            max_turns=args.max_turns,
            turn_artifact_dir=turn_dir,
            judge_model_key=args.judge_model,
            max_time_seconds=args.max_time_seconds,
        )
        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(result), default=str, sort_keys=True) + "\n")
        print(f"Rerun progress: {index}/{len(targets)}")

    elapsed_hours = (time.time() - start) / 3600
    print(f"\nRerun completed in {elapsed_hours:.2f} hours")
    print(f"Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

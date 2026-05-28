#!/usr/bin/env python3
"""Package Kerminal KernelBench operator review artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


SUMMARY_COLUMNS = [
    "level",
    "problem",
    "source_run",
    "compiled",
    "correct",
    "submitted",
    "speedup",
    "ref_ms",
    "sol_ms",
    "error",
    "baseline_type",
    "op_type",
    "precision_used",
    "ref_kernels",
    "sol_kernels",
    "elapsed_seconds",
    "baseline_path",
    "solution_path",
    "result_path",
    "solution_hash_result",
    "solution_hash_extracted",
    "solution_hash_match",
    "source_turn_dir",
    "solution_extract_error",
]


def _parse_run_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("runs must be LABEL=outputs/batch_eval/run_*")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("run label cannot be empty")
    return label, Path(path)


def _load_results(run_dir: Path, label: str, run_index: int) -> list[dict[str, Any]]:
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        raise SystemExit(f"No results.jsonl found: {results_path}")

    rows: list[dict[str, Any]] = []
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_run"] = label
            row["_source_run_dir"] = _relpath(run_dir)
            row["_source_index"] = run_index
            rows.append(row)
    return rows


def _clean_result(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {k: v for k, v in row.items() if not k.startswith("_")}


def _select_attempt(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [row for row in attempts if row.get("correct")]
    if correct:
        return max(
            correct,
            key=lambda row: (
                float(row.get("speedup") or -1.0),
                int(row.get("_source_index", -1)),
            ),
        )
    return max(attempts, key=lambda row: int(row.get("_source_index", -1)))


def _task_id(row: dict[str, Any]) -> str:
    return f"{row['model']}_{row['gpu']}_{row['problem']}".replace("/", "-").replace(" ", "_")


def _operator_dir_name(level: int, problem: str) -> str:
    stem = Path(problem).stem.rstrip("_")
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return f"L{level}_{stem}"


def _relpath(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _extract_solution_from_events(events_path: Path) -> tuple[str | None, int, str | None]:
    if not events_path.exists():
        return None, 0, f"missing {events_path.name}"

    content: str | None = None
    diff_count = 0
    try:
        with events_path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") != "turn_diff":
                    continue
                raw = event.get("raw") or {}
                diff = raw.get("unified_diff") or ""
                patched = _solution_from_unified_diff(diff, content)
                if patched is not None:
                    content = patched
                    diff_count += 1
    except OSError as exc:
        return None, diff_count, str(exc)

    if content is None:
        return None, diff_count, "no solution.py turn_diff found"
    return content, diff_count, None


def _solution_from_unified_diff(diff: str, previous: str | None) -> str | None:
    blocks = _split_diff_blocks(diff)
    for block in blocks:
        if not _is_solution_block(block):
            continue
        if any(line.startswith("--- /dev/null") for line in block):
            return _extract_new_file(block)
        if previous is not None:
            return _apply_update_patch(previous, block)
        return _extract_from_context_and_additions(block)
    return None


def _split_diff_blocks(diff: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _is_solution_block(block: list[str]) -> bool:
    header = "\n".join(block[:8])
    return "solution.py" in header


def _extract_new_file(block: list[str]) -> str:
    lines: list[str] = []
    for line in block:
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("diff ") or line.startswith("new file") or line.startswith("index "):
            continue
        if line.startswith("@@") or line.startswith("\\"):
            continue
        if line.startswith("+"):
            lines.append(line[1:])
    return "\n".join(lines) + "\n"


def _extract_from_context_and_additions(block: list[str]) -> str:
    lines: list[str] = []
    in_hunk = False
    for line in block:
        if line.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk or line.startswith("\\"):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            lines.append(line[1:])
        elif line.startswith(" "):
            lines.append(line[1:])
    return "\n".join(lines) + "\n"


def _apply_update_patch(previous: str, block: list[str]) -> str:
    old = previous.splitlines()
    new: list[str] = []
    old_index = 0
    in_hunk = False

    for line in block:
        if line.startswith("@@"):
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
            if not match:
                continue
            start = int(match.group(1)) - 1
            new.extend(old[old_index:start])
            old_index = start
            in_hunk = True
            continue

        if not in_hunk or line.startswith("\\"):
            continue
        if line.startswith(" "):
            if old_index < len(old):
                new.append(old[old_index])
            else:
                new.append(line[1:])
            old_index += 1
        elif line.startswith("-"):
            old_index += 1
        elif line.startswith("+") and not line.startswith("+++"):
            new.append(line[1:])

    new.extend(old[old_index:])
    return "\n".join(new) + "\n"


def _copy_optional_logs(source_turn_dir: Path, operator_dir: Path) -> None:
    if not source_turn_dir.exists():
        return
    for log_path in sorted(source_turn_dir.glob("turn_*_*.log")):
        shutil.copy2(log_path, operator_dir / log_path.name)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def _bundle_rel(path: Path, bundle_dir: Path) -> str:
    try:
        return str(path.relative_to(bundle_dir))
    except ValueError:
        return _relpath(path)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _geomean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def _fmt_metric(value: float | None) -> str:
    return f"{value:.6f}x" if value is not None else "N/A"


def _write_readme(
    path: Path,
    *,
    hardware_label: str,
    bundle_name: str,
    run_args: list[tuple[str, Path]],
    summary_rows: list[dict[str, Any]],
) -> None:
    total = len(summary_rows)
    compiled = sum(1 for row in summary_rows if row["compiled"])
    correct = sum(1 for row in summary_rows if row["correct"])
    speedups = [
        float(row["speedup"])
        for row in summary_rows
        if row["correct"] and row.get("speedup") not in (None, "")
    ]

    by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_level[int(row["level"])].append(row)

    lines = [
        f"# Kerminal {hardware_label} KernelBench Operator Review Bundle",
        "",
        f"This directory contains baseline operators, Kerminal-generated operator implementations, and benchmark metrics for the {hardware_label} KernelBench v3 run.",
        "",
        "## Runs",
        "",
    ]
    for label, run_dir in run_args:
        lines.append(f"- {label}: `{_relpath(run_dir)}`")

    lines.extend(
        [
            "",
            f"The combined table selects the best correct retry for each operator; if no attempt is correct, it keeps the latest retry result for that operator.",
            "",
            "## Summary",
            "",
            f"- Bundle: `{bundle_name}`",
            f"- Total operators: {total}",
            f"- Compiled: {compiled}/{total}",
            f"- Correct: {correct}/{total}",
            f"- Average speedup over correct operators: {_fmt_metric(_mean(speedups))}",
            f"- Geomean speedup over correct operators: {_fmt_metric(_geomean(speedups))}",
            "",
            "## Files",
            "",
            "- `summary.csv`: one row per operator with correctness, speedup, baseline path, and solution path.",
            "- `combined_results.jsonl`: selected raw result rows, one per operator.",
            "- `operators/<operator>/baseline.py`: original KernelBench problem definition when available.",
            "- `operators/<operator>/solution.py`: Kerminal implementation reconstructed from Kerminal event diffs when available.",
            "- `operators/<operator>/result.json`: selected result plus source run metadata and extraction status.",
            "- `operators/<operator>/turn_*_compile.log` and `turn_*_self_check.log`: validation logs when present.",
            "",
            "## By Level",
            "",
            "| Level | Correct / Total | Avg Speedup | Geomean Speedup |",
            "|---:|---:|---:|---:|",
        ]
    )

    for level in sorted(by_level):
        level_rows = by_level[level]
        level_speedups = [
            float(row["speedup"])
            for row in level_rows
            if row["correct"] and row.get("speedup") not in (None, "")
        ]
        lines.append(
            f"| L{level} | {sum(1 for row in level_rows if row['correct'])} / {len(level_rows)} | "
            f"{_fmt_metric(_mean(level_speedups))} | {_fmt_metric(_geomean(level_speedups))} |"
        )

    failed = [row for row in summary_rows if not row["correct"]]
    if failed:
        lines.extend(
            [
                "",
                "## Remaining Failures",
                "",
                "| Problem | Source | Compiled | Submitted | Error |",
                "|---|---|---:|---:|---|",
            ]
        )
        for row in failed:
            error = str(row.get("error") or "").replace("|", "\\|")
            lines.append(
                f"| `{row['problem']}` | `{row['source_run']}` | {row['compiled']} | {row['submitted']} | {error} |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_root_readme(path: Path, hardware_label: str, review_dir: Path, archive_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"# Kerminal {hardware_label} Artifacts",
                "",
                f"This directory contains archived results for the Kerminal {hardware_label} KernelBench v3 evaluation.",
                "",
                f"- `operator_review/{review_dir.name}/` contains per-operator review materials.",
                f"- `operator_review/{archive_name}` is the compressed review bundle.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _make_archive(bundle_dir: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_dir.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", required=True, help="Hardware target, e.g. a100_local")
    parser.add_argument("--hardware-label", default=None, help="Display label, e.g. A100")
    parser.add_argument("--bundle-name", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run", action="append", type=_parse_run_arg, required=True)
    parser.add_argument("--archive-name", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from src.hardware import get_target

    target = get_target(args.hardware)
    hardware_label = args.hardware_label or target.gpu_sku
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    bundle_dir = output_root / args.bundle_name
    archive_name = args.archive_name or f"kerminal_{target.gpu_sku.lower()}_operator_review.tar.gz"
    archive_path = output_root / archive_name

    if bundle_dir.exists():
        if not args.force:
            raise SystemExit(f"Output already exists, pass --force to replace: {bundle_dir}")
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    operators_dir = bundle_dir / "operators"
    operators_dir.mkdir()

    rows_by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    run_args: list[tuple[str, Path]] = []
    for index, (label, run_dir) in enumerate(args.run):
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        run_args.append((label, run_dir))
        for row in _load_results(run_dir, label, index):
            rows_by_problem[row["problem"]].append(row)

    problem_map = {path.name: (level, path) for level, path in target.find_problems(REPO_ROOT)}
    selected_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for problem, (level, problem_path) in problem_map.items():
        attempts = rows_by_problem.get(problem, [])
        if not attempts:
            continue
        selected = _select_attempt(attempts)
        selected_rows.append(_clean_result(selected) or {})

        op_dir_name = _operator_dir_name(level, problem)
        op_dir = operators_dir / op_dir_name
        op_dir.mkdir(parents=True)

        baseline_source: str | None = None
        baseline_path = ""
        if problem_path.exists():
            baseline_source = _relpath(problem_path)
            baseline_dest = op_dir / "baseline.py"
            shutil.copy2(problem_path, baseline_dest)
            baseline_path = _bundle_rel(baseline_dest, bundle_dir)

        source_run_dir = REPO_ROOT / selected["_source_run_dir"]
        source_turn_dir = source_run_dir / "turns" / _task_id(selected)
        solution_code, turn_diff_count, extract_error = _extract_solution_from_events(
            source_turn_dir / "kerminal_events.jsonl"
        )
        extracted_hash = _hash_text(solution_code) if solution_code is not None else None
        solution_path = ""
        if solution_code is not None:
            solution_dest = op_dir / "solution.py"
            solution_dest.write_text(solution_code, encoding="utf-8")
            solution_path = _bundle_rel(solution_dest, bundle_dir)

        _copy_optional_logs(source_turn_dir, op_dir)

        result_path = op_dir / "result.json"
        selected_clean = _clean_result(selected)
        result_hash = selected.get("solution_hash")
        hash_match = None
        if result_hash and extracted_hash:
            hash_match = result_hash == extracted_hash

        metadata = {
            "baseline_source": baseline_source,
            "extracted_solution_hash": extracted_hash,
            "attempt_results": {
                row["_source_run"]: _clean_result(row)
                for row in sorted(attempts, key=lambda item: int(item["_source_index"]))
            },
            "selected_result": selected_clean,
            "solution_extract_error": extract_error,
            "solution_hash_match": hash_match,
            "solution_turn_diff_count": turn_diff_count,
            "source_run": selected["_source_run"],
            "source_run_dir": selected["_source_run_dir"],
            "source_turn_dir": _relpath(source_turn_dir),
        }
        result_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        summary_rows.append(
            {
                "level": level,
                "problem": problem,
                "source_run": selected["_source_run"],
                "compiled": bool(selected.get("compiled")),
                "correct": bool(selected.get("correct")),
                "submitted": bool(selected.get("submitted")),
                "speedup": selected.get("speedup"),
                "ref_ms": selected.get("ref_ms"),
                "sol_ms": selected.get("sol_ms"),
                "error": selected.get("error") or "",
                "baseline_type": selected.get("baseline_type") or "",
                "op_type": selected.get("op_type") or "",
                "precision_used": selected.get("precision_used") or "",
                "ref_kernels": selected.get("ref_kernels"),
                "sol_kernels": selected.get("sol_kernels"),
                "elapsed_seconds": selected.get("elapsed_seconds"),
                "baseline_path": baseline_path,
                "solution_path": solution_path,
                "result_path": _bundle_rel(result_path, bundle_dir),
                "solution_hash_result": result_hash or "",
                "solution_hash_extracted": extracted_hash or "",
                "solution_hash_match": hash_match,
                "source_turn_dir": _relpath(source_turn_dir),
                "solution_extract_error": extract_error or "",
            }
        )

    _write_csv(bundle_dir / "summary.csv", summary_rows)
    (bundle_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (bundle_dir / "combined_results.jsonl").open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    _write_readme(
        bundle_dir / "README.md",
        hardware_label=hardware_label,
        bundle_name=args.bundle_name,
        run_args=run_args,
        summary_rows=summary_rows,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    _make_archive(bundle_dir, archive_path)
    _write_root_readme(output_root.parent / "README.md", hardware_label, bundle_dir, archive_name)

    print(f"Wrote bundle: {_relpath(bundle_dir)}")
    print(f"Wrote archive: {_relpath(archive_path)}")
    print(
        "Summary: "
        f"{sum(1 for row in summary_rows if row['correct'])}/{len(summary_rows)} correct, "
        f"{sum(1 for row in summary_rows if row['compiled'])}/{len(summary_rows)} compiled"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

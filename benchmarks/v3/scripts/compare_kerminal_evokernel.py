#!/usr/bin/env python3
"""Compare Kerminal A100 results against an EvoKernel leaderboard experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any


V3_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KERMINAL_SUMMARY = (
    "artifacts/kerminal_a100/operator_review/"
    "kerminal_a100_run_20260521_052335_with_retries_20260522/summary.csv"
)
DEFAULT_EVOKERNEL_OPS = "artifacts/evokernel_leaderboard/ops.csv"
DEFAULT_OUTPUT_DIR = "artifacts/kerminal_a100/evokernel_comparison"
DEFAULT_ARCHIVE_NAME = "kerminal_a100_vs_evokernel_kernelbench_ncu.tar.gz"

EXPECTED_KERMINAL_ROWS = 53
EXPECTED_EVOKERNEL_ROWS = 250
EXPECTED_INTERSECTION_ROWS = 32


COMPARISON_COLUMNS = [
    "match_key",
    "level",
    "category",
    "kerminal_problem",
    "evokernel_op_name",
    "kerminal_correct",
    "kerminal_compiled",
    "evokernel_valid",
    "evokernel_compiled",
    "kerminal_speedup",
    "evokernel_speedup",
    "kerminal_ref_ms",
    "kerminal_sol_ms",
    "evokernel_best_mean_ms",
    "winner",
    "kerminal_to_evokernel_speedup_ratio",
    "kerminal_error",
    "kerminal_source_run",
    "kerminal_baseline_type",
    "evokernel_detail_path",
]


PER_OPERATOR_COLUMNS = [
    "Level",
    "Problem",
    "Category",
    "Kerminal Correct",
    "Kerminal Speedup",
    "Kerminal Ref ms",
    "Kerminal Sol ms",
    "EvoKernel Valid",
    "EvoKernel Speedup",
    "EvoKernel Mean ms",
    "Winner",
    "K/E Speedup Ratio",
]


OP_CATEGORIES = {
    "attention": "Attention",
    "conv": "Conv",
    "elementwise": "Elementwise",
    "fused": "Fused",
    "gemm": "GEMM",
    "gemm_epilogue": "GEMM",
    "gemv": "GEMV",
    "layernorm": "Norm",
    "model": "Other",
    "moe_grouped_gemm": "GEMM",
    "reduction": "Reduction",
    "softmax": "Softmax",
}


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return V3_ROOT / path


def _relpath(path: Path) -> str:
    try:
        return str(path.relative_to(V3_ROOT))
    except ValueError:
        return str(path)


def _normalize_full_name(name: str) -> str:
    stem = Path(name).stem if name.endswith(".py") else name
    stem = stem.lower()
    return re.sub(r"[^a-z0-9]+", "", stem)


def _normalize_suffix_name(name: str) -> str:
    stem = Path(name).stem if name.endswith(".py") else name
    stem = re.sub(r"^\d+_", "", stem)
    stem = stem.lower()
    return re.sub(r"[^a-z0-9]+", "", stem)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _fmt_speedup(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{_fmt_float(value)}x"


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "--"
    return _fmt_float(value, digits=3)


def _geomean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _category(row: dict[str, Any]) -> str:
    op_type = str(row.get("op_type") or "").strip().lower()
    return OP_CATEGORIES.get(op_type, op_type.title() if op_type else "Other")


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _unique_index(rows: list[dict[str, str]], key_field: str) -> dict[str, dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[key_field]].append(row)
    return {key: values[0] for key, values in grouped.items() if len(values) == 1}


def _duplicate_keys(rows: list[dict[str, str]], key_field: str) -> set[str]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row[key_field]] += 1
    return {key for key, count in counts.items() if count > 1}


def _winner(k_speed: float | None, e_speed: float | None) -> str:
    if k_speed is None or e_speed is None:
        return ""
    if math.isclose(k_speed, e_speed, rel_tol=1e-9, abs_tol=1e-12):
        return "Tie"
    return "Kerminal" if k_speed > e_speed else "EvoKernel"


def _sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    level = int(row["level"])
    problem = str(row["kerminal_problem"])
    match = re.match(r"(\d+)_", problem)
    ordinal = int(match.group(1)) if match else 10_000
    return level, ordinal, problem


def _build_comparison(k_rows: list[dict[str, str]], e_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    for row in k_rows:
        row["match_full_key"] = _normalize_full_name(row["problem"])
        row["match_suffix_key"] = _normalize_suffix_name(row["problem"])
    for row in e_rows:
        row["match_full_key"] = _normalize_full_name(row["opName"])
        row["match_suffix_key"] = _normalize_suffix_name(row["opName"])

    k_by_full = _unique_index(k_rows, "match_full_key")
    e_by_full = _unique_index(e_rows, "match_full_key")
    k_by_suffix = _unique_index(k_rows, "match_suffix_key")
    e_by_suffix = _unique_index(e_rows, "match_suffix_key")
    k_duplicate_suffixes = _duplicate_keys(k_rows, "match_suffix_key")
    e_duplicate_suffixes = _duplicate_keys(e_rows, "match_suffix_key")

    pairs: dict[str, tuple[dict[str, str], dict[str, str], str]] = {}
    for full_key in sorted(set(k_by_full) & set(e_by_full)):
        k_row = k_by_full[full_key]
        e_row = e_by_full[full_key]
        pairs[k_row["match_suffix_key"]] = (k_row, e_row, "full")

    for suffix_key in sorted(set(k_by_suffix) & set(e_by_suffix)):
        if suffix_key in pairs:
            continue
        if suffix_key in k_duplicate_suffixes or suffix_key in e_duplicate_suffixes:
            raise SystemExit(f"Ambiguous suffix match without full-key match: {suffix_key}")
        pairs[suffix_key] = (k_by_suffix[suffix_key], e_by_suffix[suffix_key], "suffix")

    rows: list[dict[str, Any]] = []
    for key in sorted(pairs):
        k_row, e_row, match_method = pairs[key]
        k_speed = _float(k_row.get("speedup"))
        e_speed = _float(e_row.get("bestSpeedupVsEager"))
        win = _winner(k_speed, e_speed)
        ratio = k_speed / e_speed if k_speed is not None and e_speed not in (None, 0) else None
        rows.append(
            {
                "match_key": key,
                "match_method": match_method,
                "level": int(k_row["level"]),
                "category": _category(k_row),
                "kerminal_problem": k_row["problem"],
                "evokernel_op_name": e_row["opName"],
                "kerminal_correct": _bool(k_row.get("correct")),
                "kerminal_compiled": _bool(k_row.get("compiled")),
                "evokernel_valid": _bool(e_row.get("validAny")),
                "evokernel_compiled": _bool(e_row.get("compileAny")),
                "kerminal_speedup": k_speed,
                "evokernel_speedup": e_speed,
                "kerminal_ref_ms": _float(k_row.get("ref_ms")),
                "kerminal_sol_ms": _float(k_row.get("sol_ms")),
                "evokernel_best_mean_ms": _float(e_row.get("bestMeanMs")),
                "winner": win,
                "kerminal_to_evokernel_speedup_ratio": ratio,
                "kerminal_error": k_row.get("error") or "",
                "kerminal_source_run": k_row.get("source_run") or "",
                "kerminal_baseline_type": k_row.get("baseline_type") or "",
                "evokernel_detail_path": e_row.get("detailPath") or "",
            }
        )
    return sorted(rows, key=_sort_key)


def _numeric_speedups(rows: list[dict[str, Any]], field: str) -> list[float]:
    return [row[field] for row in rows if isinstance(row.get(field), float)]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    k_speeds = _numeric_speedups(rows, "kerminal_speedup")
    e_speeds = _numeric_speedups(rows, "evokernel_speedup")
    comparable = [
        row
        for row in rows
        if isinstance(row.get("kerminal_speedup"), float) and isinstance(row.get("evokernel_speedup"), float)
    ]
    return {
        "intersection_rows": len(rows),
        "kerminal_correct": sum(1 for row in rows if row["kerminal_correct"]),
        "kerminal_compiled": sum(1 for row in rows if row["kerminal_compiled"]),
        "evokernel_valid": sum(1 for row in rows if row["evokernel_valid"]),
        "evokernel_compiled": sum(1 for row in rows if row["evokernel_compiled"]),
        "kerminal_speedup_count": len(k_speeds),
        "evokernel_speedup_count": len(e_speeds),
        "kerminal_avg_speedup": _mean(k_speeds),
        "evokernel_avg_speedup": _mean(e_speeds),
        "kerminal_geomean_speedup": _geomean(k_speeds),
        "evokernel_geomean_speedup": _geomean(e_speeds),
        "kerminal_faster_than_1x": sum(1 for value in k_speeds if value > 1.0),
        "evokernel_faster_than_1x": sum(1 for value in e_speeds if value > 1.0),
        "head_to_head_comparable": len(comparable),
        "kerminal_wins": sum(1 for row in comparable if row["winner"] == "Kerminal"),
        "evokernel_wins": sum(1 for row in comparable if row["winner"] == "EvoKernel"),
        "ties": sum(1 for row in comparable if row["winner"] == "Tie"),
    }


def _by_level(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[int(row["level"])].append(row)

    level_rows: list[dict[str, Any]] = []
    for level in sorted(groups):
        group = groups[level]
        summary = _summarize(group)
        level_rows.append(
            {
                "Level": f"L{level}",
                "Ops": len(group),
                "Kerminal Correct": f"{summary['kerminal_correct']}/{len(group)}",
                "EvoKernel Valid": f"{summary['evokernel_valid']}/{len(group)}",
                "Kerminal >1x": f"{summary['kerminal_faster_than_1x']}/{summary['kerminal_speedup_count']}",
                "EvoKernel >1x": f"{summary['evokernel_faster_than_1x']}/{summary['evokernel_speedup_count']}",
                "Kerminal Geo": _fmt_speedup(summary["kerminal_geomean_speedup"]),
                "EvoKernel Geo": _fmt_speedup(summary["evokernel_geomean_speedup"]),
                "Kerminal Wins": summary["kerminal_wins"],
                "EvoKernel Wins": summary["evokernel_wins"],
                "Ties": summary["ties"],
            }
        )
    return level_rows


def _overall_rows(
    *,
    kerminal_rows: int,
    evokernel_rows: int,
    comparison_rows: list[dict[str, Any]],
    kerminal_hardware: str,
    evokernel_hardware: str,
) -> list[dict[str, str]]:
    summary = _summarize(comparison_rows)
    total = summary["intersection_rows"]
    return [
        {"Metric": "Comparison scope", "Value": "Exact-normalized KernelBench operator intersection"},
        {"Metric": "Kerminal hardware", "Value": kerminal_hardware},
        {"Metric": "EvoKernel hardware", "Value": evokernel_hardware},
        {"Metric": "Kerminal input rows", "Value": str(kerminal_rows)},
        {"Metric": "EvoKernel experiment rows", "Value": str(evokernel_rows)},
        {"Metric": "Intersection rows", "Value": str(total)},
        {"Metric": "Kerminal correct", "Value": f"{summary['kerminal_correct']}/{total}"},
        {"Metric": "EvoKernel valid", "Value": f"{summary['evokernel_valid']}/{total}"},
        {"Metric": "Kerminal geomean speedup", "Value": _fmt_speedup(summary["kerminal_geomean_speedup"])},
        {"Metric": "EvoKernel geomean speedup", "Value": _fmt_speedup(summary["evokernel_geomean_speedup"])},
        {"Metric": "Kerminal >1x", "Value": f"{summary['kerminal_faster_than_1x']}/{summary['kerminal_speedup_count']}"},
        {"Metric": "EvoKernel >1x", "Value": f"{summary['evokernel_faster_than_1x']}/{summary['evokernel_speedup_count']}"},
        {"Metric": "Head-to-head comparable", "Value": str(summary["head_to_head_comparable"])},
        {"Metric": "Kerminal higher relative speedup", "Value": str(summary["kerminal_wins"])},
        {"Metric": "EvoKernel higher relative speedup", "Value": str(summary["evokernel_wins"])},
        {"Metric": "Ties", "Value": str(summary["ties"])},
    ]


def _per_operator_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "Level": f"L{row['level']}",
            "Problem": Path(row["kerminal_problem"]).stem.rstrip("_").replace("_", " "),
            "Category": row["category"],
            "Kerminal Correct": "Y" if row["kerminal_correct"] else "N",
            "Kerminal Speedup": _fmt_speedup(row["kerminal_speedup"]),
            "Kerminal Ref ms": _fmt_ms(row["kerminal_ref_ms"]),
            "Kerminal Sol ms": _fmt_ms(row["kerminal_sol_ms"]),
            "EvoKernel Valid": "Y" if row["evokernel_valid"] else "N",
            "EvoKernel Speedup": _fmt_speedup(row["evokernel_speedup"]),
            "EvoKernel Mean ms": _fmt_ms(row["evokernel_best_mean_ms"]),
            "Winner": row["winner"] or "--",
            "K/E Speedup Ratio": _fmt_float(row["kerminal_to_evokernel_speedup_ratio"]),
        }
        for row in rows
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    fieldnames = columns or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in fieldnames})


def _markdown_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    if not rows:
        return ""
    fieldnames = columns or list(rows[0].keys())
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        values = [str(row.get(column, "")).replace("|", "\\|") for column in fieldnames]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _latex_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    if not rows:
        return ""
    fieldnames = columns or list(rows[0].keys())
    column_spec = "l" * len(fieldnames)
    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(_latex_escape(column) for column in fieldnames) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(row.get(column, "")) for column in fieldnames) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def _write_table_set(output_dir: Path, name: str, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    _write_csv(output_dir / f"{name}.csv", rows, columns)
    (output_dir / f"{name}.md").write_text(_markdown_table(rows, columns), encoding="utf-8")
    (output_dir / f"{name}.tex").write_text(_latex_table(rows, columns), encoding="utf-8")


def _write_readme(
    path: Path,
    *,
    overall_rows: list[dict[str, str]],
    by_level_rows: list[dict[str, Any]],
    per_operator_rows: list[dict[str, str]],
    evokernel_experiment: str,
) -> None:
    lines = [
        "# Kerminal A100 vs EvoKernel KernelBench + NCU",
        "",
        "This report compares the Kerminal A100 KernelBench-v3 run against the EvoKernel "
        f"`{evokernel_experiment}` leaderboard experiment over exact-normalized matching operators.",
        "",
        "**Hardware caveat:** Kerminal was run on `NVIDIA A100-SXM4-80GB`; the selected EvoKernel "
        "experiment was run on `NVIDIA A800-SXM4-80GB`. Treat this as a KernelBench task-overlap "
        "reference comparison using each system's own eager-relative speedup, not as a strict "
        "same-machine latency comparison.",
        "",
        "## Overall",
        "",
        _markdown_table(overall_rows),
        "",
        "## By Level",
        "",
        _markdown_table(by_level_rows),
        "",
        "## Per Operator",
        "",
        _markdown_table(per_operator_rows, PER_OPERATOR_COLUMNS),
        "",
        "## Files",
        "",
        "- `comparison.csv`: raw joined operator-level comparison.",
        "- `summary.json`: machine-readable summary, inputs, caveats, and aggregates.",
        "- `tables/`: CSV, Markdown, and LaTeX versions of the overall, by-level, and per-operator tables.",
        "- `kerminal_a100_vs_evokernel_kernelbench_ncu.tar.gz`: compressed copy of this report directory.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _make_archive(output_dir: Path, archive_path: Path) -> None:
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in sorted(output_dir.rglob("*")):
            if path == archive_path:
                continue
            tar.add(path, arcname=str(output_dir.name / path.relative_to(output_dir)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kerminal-summary", default=DEFAULT_KERMINAL_SUMMARY)
    parser.add_argument("--evokernel-ops", default=DEFAULT_EVOKERNEL_OPS)
    parser.add_argument("--experiment-id", default="kernelbench-cuda-ncu")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--archive-name", default=DEFAULT_ARCHIVE_NAME)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    kerminal_path = _resolve_path(args.kerminal_summary)
    evokernel_path = _resolve_path(args.evokernel_ops)
    output_dir = _resolve_path(args.output_dir)
    archive_path = output_dir / args.archive_name

    kerminal_rows = _load_csv(kerminal_path)
    evokernel_all_rows = _load_csv(evokernel_path)
    evokernel_rows = [row for row in evokernel_all_rows if row.get("experimentId") == args.experiment_id]
    if not evokernel_rows:
        raise SystemExit(f"No EvoKernel rows found for experiment: {args.experiment_id}")

    comparison_rows = _build_comparison(kerminal_rows, evokernel_rows)
    if len(kerminal_rows) != EXPECTED_KERMINAL_ROWS:
        raise SystemExit(f"Expected {EXPECTED_KERMINAL_ROWS} Kerminal rows, found {len(kerminal_rows)}")
    if len(evokernel_rows) != EXPECTED_EVOKERNEL_ROWS:
        raise SystemExit(f"Expected {EXPECTED_EVOKERNEL_ROWS} EvoKernel rows, found {len(evokernel_rows)}")
    if len(comparison_rows) != EXPECTED_INTERSECTION_ROWS:
        raise SystemExit(f"Expected {EXPECTED_INTERSECTION_ROWS} intersection rows, found {len(comparison_rows)}")

    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"Output already exists, pass --force to replace: {_relpath(output_dir)}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir()

    kerminal_hardware = "NVIDIA A100-SXM4-80GB"
    evokernel_hardware = evokernel_rows[0].get("hardware") or "NVIDIA A800-SXM4-80GB"
    overall_rows = _overall_rows(
        kerminal_rows=len(kerminal_rows),
        evokernel_rows=len(evokernel_rows),
        comparison_rows=comparison_rows,
        kerminal_hardware=kerminal_hardware,
        evokernel_hardware=evokernel_hardware,
    )
    by_level_rows = _by_level(comparison_rows)
    per_operator_rows = _per_operator_rows(comparison_rows)

    _write_csv(output_dir / "comparison.csv", comparison_rows, COMPARISON_COLUMNS)
    _write_table_set(tables_dir, "overall", overall_rows)
    _write_table_set(tables_dir, "by_level", by_level_rows)
    _write_table_set(tables_dir, "per_operator", per_operator_rows, PER_OPERATOR_COLUMNS)

    summary = {
        "comparison": _summarize(comparison_rows),
        "caveat": (
            "Kerminal rows were measured on NVIDIA A100-SXM4-80GB; EvoKernel "
            f"{args.experiment_id} rows were measured on {evokernel_hardware}. "
            "Compare eager-relative speedups, not absolute latency."
        ),
        "evokernel_experiment": {
            "id": args.experiment_id,
            "title": evokernel_rows[0].get("experimentTitle") or "",
            "hardware": evokernel_hardware,
            "rows": len(evokernel_rows),
        },
        "inputs": {
            "kerminal_summary": _relpath(kerminal_path),
            "evokernel_ops": _relpath(evokernel_path),
        },
        "kerminal": {
            "hardware": kerminal_hardware,
            "rows": len(kerminal_rows),
        },
        "matching": {
            "method": "strip .py, remove leading numeric prefix, lowercase, remove non-alphanumerics",
            "intersection_rows": len(comparison_rows),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_readme(
        output_dir / "README.md",
        overall_rows=overall_rows,
        by_level_rows=by_level_rows,
        per_operator_rows=per_operator_rows,
        evokernel_experiment=args.experiment_id,
    )
    _make_archive(output_dir, archive_path)

    comparison = summary["comparison"]
    print(f"Wrote comparison: {_relpath(output_dir)}")
    print(f"Wrote archive: {_relpath(archive_path)}")
    print(
        "Summary: "
        f"{comparison['intersection_rows']} overlap rows, "
        f"Kerminal {comparison['kerminal_correct']}/{comparison['intersection_rows']} correct, "
        f"EvoKernel {comparison['evokernel_valid']}/{comparison['intersection_rows']} valid, "
        f"wins {comparison['kerminal_wins']} Kerminal / {comparison['evokernel_wins']} EvoKernel"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

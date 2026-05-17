# Kerminal H100 KernelBench Operator Review Bundle

This directory contains baseline operators, Kerminal-generated operator implementations, and benchmark metrics for the H100 KernelBench v3 run.

## Runs

- Full run: `outputs/batch_eval/run_20260516_110448`
- L1 30-minute retry: `outputs/batch_eval/retry_run_20260516_110448_20260517_085420`
- The combined table uses retry results for the six retried L1 failures and full-run results for all other operators.

## Summary

- Total operators: 54
- Compiled: 49/54
- Correct: 47/54
- Average speedup over correct operators: 2.127008x
- Geomean speedup over correct operators: 1.602895x

## Files

- `summary.csv`: one row per operator with correctness, speedup, baseline path, and solution path.
- `combined_results.jsonl`: selected raw result rows, one per operator.
- `operators/<operator>/baseline.py`: original KernelBench baseline/problem definition.
- `operators/<operator>/solution.py`: Kerminal implementation reconstructed from Kerminal event diffs when available.
- `operators/<operator>/result.json`: selected result plus source run metadata and extraction status.
- `operators/<operator>/turn_*_compile.log` and `turn_*_self_check.log`: validation logs when present.

## By Level

| Level | Correct / Total | Avg Speedup | Geomean Speedup |
|---:|---:|---:|---:|
| L1 | 26 / 28 | 1.725498x | 1.424020x |
| L2 | 15 / 15 | 1.702036x | 1.504517x |
| L3 | 3 / 3 | 6.425779x | 4.624943x |
| L4 | 3 / 8 | 3.432849x | 2.126262x |

## Retried L1 Operators

| Problem | Correct | Speedup | Error |
|---|---:|---:|---|
| `2_Standard_matrix_multiplication_.py` | True | 0.762850x |  |
| `3_Batched_matrix_multiplication.py` | True | 0.702295x |  |
| `40_LayerNorm.py` | True | 2.183683x |  |
| `47_Sum_reduction_over_a_dimension.py` | True | 1.215707x |  |
| `4_Matrix_vector_multiplication_.py` | False |  | Non-deterministic output (possible race condition) |
| `8_Matmul_with_irregular_shapes_.py` | False |  | timeout_exceeded |

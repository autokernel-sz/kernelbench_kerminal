# Kerminal H100 Artifacts

This directory contains the archived results for the Kerminal H100 KernelBench v3 evaluation.

- `merged_v3_kerminal/results_with_kerminal_h100.csv` merges the public v3 results with the Kerminal H100 combined run.
- `merged_v3_kerminal/h100_model_summary.csv` summarizes H100 model-level results.
- `merged_v3_kerminal/paper_tables/` contains CSV, Markdown, and LaTeX tables.
- `operator_review/` contains per-operator review materials, including baseline implementations, reconstructed Kerminal `solution.py` files, and result metadata.

The Kerminal combined result uses:

- Full run: `outputs/batch_eval/run_20260516_110448`
- L1 retry: `outputs/batch_eval/retry_run_20260516_110448_20260517_085420`

The raw `outputs/batch_eval` logs are not archived here because they include large interaction logs. The review bundle keeps the reproducible per-operator baseline, generated solution, and benchmark result metadata.

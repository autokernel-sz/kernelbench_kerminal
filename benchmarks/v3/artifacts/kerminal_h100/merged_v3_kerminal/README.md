# V3 Results + Kerminal H100 Merge

- Base CSV: `/workspace/kernelbench_kerminal/public/data/v3/results.csv`
- Kerminal review bundle: `/workspace/kernelbench_kerminal/benchmarks/v3/outputs/operator_review/kerminal_h100_run_20260516_110448_with_l1_retry_20260517_085420`
- Merged CSV: `/workspace/kernelbench_kerminal/benchmarks/v3/outputs/merged_v3_kerminal/results_with_kerminal_h100.csv`
- H100 summary: `/workspace/kernelbench_kerminal/benchmarks/v3/outputs/merged_v3_kerminal/h100_model_summary.csv`

## H100 Ranking

| Rank | Model | Correct / Total | Compiled | Avg Speedup | Geomean Speedup |
|---:|---|---:|---:|---:|---:|
| 1 | Kerminal Default | 47 / 54 | 49 | 2.127x | 1.603x |
| 2 | GPT-5.4 | 42 / 54 | 45 | 0.968x | 0.654x |
| 3 | Gemini 3 Flash Preview | 41 / 54 | 46 | 1.048x | 0.778x |
| 4 | GPT-5.3 | 40 / 54 | 47 | 0.871x | 0.453x |
| 5 | Claude Opus 4.6 | 37 / 54 | 44 | 1.599x | 1.179x |
| 6 | GLM-5 | 32 / 54 | 42 | 1.049x | 0.762x |
| 7 | Kimi K2.5 | 27 / 54 | 34 | 1.227x | 0.801x |
| 8 | Qwen3.5 397B A17B | 22 / 54 | 34 | 0.735x | 0.536x |
| 9 | Claude Sonnet 4.6 | 19 / 54 | 20 | 1.057x | 0.801x |
| 10 | Qwen: Qwen3.5-122B-A10B | 17 / 54 | 33 | 0.843x | 0.487x |
| 11 | MiniMax M2.7 | 14 / 54 | 23 | 0.771x | 0.414x |
| 12 | Gemini 3.1 Pro Preview | 13 / 54 | 15 | 1.362x | 1.135x |
| 13 | MiniMax M2.5 | 9 / 54 | 21 | 1.573x | 1.204x |
| 14 | Qwen: Qwen3.5-35B-A3B | 0 / 54 | 2 | N/A | N/A |

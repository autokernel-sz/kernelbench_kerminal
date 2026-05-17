# Paper-Style Tables for V3 + Kerminal H100

Source: `outputs/merged_v3_kerminal/results_with_kerminal_h100.csv`

Generated tables are available as `.csv`, `.md`, and `.tex` files. The LaTeX tables use a booktabs-style layout (`\toprule`, `\midrule`, `\bottomrule`).

## Table 1: Overall H100 Results

| Rank | Model | Pass | Pass Rate | Compiled | >1x | Avg Speedup | Geo Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Kerminal Default | 47/54 | 87.0% | 49/54 | 35/47 | 2.13x | 1.60x |
| 2 | GPT-5.4 | 42/54 | 77.8% | 45/54 | 17/42 | 0.97x | 0.65x |
| 3 | Gemini 3 Flash Preview | 41/54 | 75.9% | 46/54 | 18/41 | 1.05x | 0.78x |
| 4 | GPT-5.3 | 40/54 | 74.1% | 47/54 | 11/40 | 0.87x | 0.45x |
| 5 | Claude Opus 4.6 | 37/54 | 68.5% | 44/54 | 20/37 | 1.60x | 1.18x |
| 6 | GLM-5 | 32/54 | 59.3% | 42/54 | 14/32 | 1.05x | 0.76x |
| 7 | Kimi K2.5 | 27/54 | 50.0% | 34/54 | 9/27 | 1.23x | 0.80x |
| 8 | Qwen3.5 397B A17B | 22/54 | 40.7% | 34/54 | 5/22 | 0.74x | 0.54x |
| 9 | Claude Sonnet 4.6 | 19/54 | 35.2% | 20/54 | 11/19 | 1.06x | 0.80x |
| 10 | Qwen: Qwen3.5-122B-A10B | 17/54 | 31.5% | 33/54 | 5/17 | 0.84x | 0.49x |
| 11 | MiniMax M2.7 | 14/54 | 25.9% | 23/54 | 5/14 | 0.77x | 0.41x |
| 12 | Gemini 3.1 Pro Preview | 13/54 | 24.1% | 15/54 | 5/13 | 1.36x | 1.14x |
| 13 | MiniMax M2.5 | 9/54 | 16.7% | 21/54 | 5/9 | 1.57x | 1.20x |
| 14 | Qwen: Qwen3.5-35B-A3B | 0/54 | 0.0% | 2/54 | 0/0 | -- | -- |

## Table 2: Pass Count by Level

| Model | L1 Pass | L2 Pass | L3 Pass | L4 Pass | All Pass | Geo Speedup |
| --- | --- | --- | --- | --- | --- | --- |
| Kerminal Default | 26/28 | 15/15 | 3/3 | 3/8 | 47/54 | 1.60x |
| GPT-5.4 | 23/28 | 11/15 | 3/3 | 5/8 | 42/54 | 0.65x |
| Gemini 3 Flash Preview | 24/28 | 10/15 | 3/3 | 4/8 | 41/54 | 0.78x |
| GPT-5.3 | 26/28 | 9/15 | 2/3 | 3/8 | 40/54 | 0.45x |
| Claude Opus 4.6 | 23/28 | 11/15 | 1/3 | 2/8 | 37/54 | 1.18x |
| GLM-5 | 22/28 | 5/15 | 1/3 | 4/8 | 32/54 | 0.76x |
| Kimi K2.5 | 15/28 | 6/15 | 3/3 | 3/8 | 27/54 | 0.80x |
| Qwen3.5 397B A17B | 17/28 | 4/15 | 1/3 | 0/8 | 22/54 | 0.54x |
| Claude Sonnet 4.6 | 12/28 | 5/15 | 0/3 | 2/8 | 19/54 | 0.80x |
| Qwen: Qwen3.5-122B-A10B | 11/28 | 3/15 | 0/3 | 3/8 | 17/54 | 0.49x |
| MiniMax M2.7 | 10/28 | 3/15 | 0/3 | 1/8 | 14/54 | 0.41x |
| Gemini 3.1 Pro Preview | 8/28 | 4/15 | 0/3 | 1/8 | 13/54 | 1.14x |
| MiniMax M2.5 | 3/28 | 3/15 | 2/3 | 1/8 | 9/54 | 1.20x |
| Qwen: Qwen3.5-35B-A3B | 0/28 | 0/15 | 0/3 | 0/8 | 0/54 | -- |

## Table 3: Geomean Speedup by Level

| Model | L1 Geo | L2 Geo | L3 Geo | L4 Geo | All Geo |
| --- | --- | --- | --- | --- | --- |
| Kerminal Default | 1.42x | 1.50x | 4.62x | 2.13x | 1.60x |
| GPT-5.4 | 0.70x | 0.78x | 0.85x | 0.27x | 0.65x |
| Gemini 3 Flash Preview | 0.66x | 0.99x | 1.49x | 0.72x | 0.78x |
| GPT-5.3 | 0.53x | 0.28x | 1.10x | 0.28x | 0.45x |
| Claude Opus 4.6 | 1.01x | 1.58x | 10.83x | 0.48x | 1.18x |
| GLM-5 | 0.70x | 1.50x | 1.32x | 0.45x | 0.76x |
| Kimi K2.5 | 0.91x | 0.88x | 0.79x | 0.36x | 0.80x |
| Qwen3.5 397B A17B | 0.61x | 0.25x | 1.35x | -- | 0.54x |
| Claude Sonnet 4.6 | 0.74x | 1.05x | -- | 0.69x | 0.80x |
| Qwen: Qwen3.5-122B-A10B | 0.42x | 0.41x | -- | 1.02x | 0.49x |
| MiniMax M2.7 | 0.35x | 0.53x | -- | 1.00x | 0.41x |
| Gemini 3.1 Pro Preview | 1.08x | 1.24x | -- | 1.20x | 1.14x |
| MiniMax M2.5 | 1.57x | 0.98x | 1.21x | 0.99x | 1.20x |
| Qwen: Qwen3.5-35B-A3B | -- | -- | -- | -- | -- |

## Table 6: Kerminal L1 Retry Outcomes

| Problem | Correct | Speedup | Ref ms | Sol ms | Error |
| --- | --- | --- | --- | --- | --- |
| Standard matrix multiplication | Y | 0.76x | 0.405 | 0.530 |  |
| Batched matrix multiplication | Y | 0.70x | 0.110 | 0.157 |  |
| LayerNorm | Y | 2.18x | 0.697 | 0.319 |  |
| Sum reduction over a dimension | Y | 1.22x | 0.019 | 0.015 |  |
| Matrix vector multiplication | N | -- | -- | -- |  |
| Matmul with irregular shapes | N | -- | -- | -- |  |

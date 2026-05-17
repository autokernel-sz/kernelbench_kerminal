| Problem | Correct | Speedup | Ref ms | Sol ms | Error |
| --- | --- | --- | --- | --- | --- |
| Standard matrix multiplication | Y | 0.76x | 0.405 | 0.530 |  |
| Batched matrix multiplication | Y | 0.70x | 0.110 | 0.157 |  |
| LayerNorm | Y | 2.18x | 0.697 | 0.319 |  |
| Sum reduction over a dimension | Y | 1.22x | 0.019 | 0.015 |  |
| Matrix vector multiplication | N | -- | -- | -- |  |
| Matmul with irregular shapes | N | -- | -- | -- |  |

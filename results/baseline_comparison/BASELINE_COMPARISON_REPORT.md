# Baseline Comparison Report

Generated on: /workspace/code

## Summary at K=5

| Model | Precision | Recall | NDCG | MAP | Avg Time (s) |
|---|---:|---:|---:|---:|---:|
| User-based CF | 0.000 | 0.000 | 0.000 | 0.000 | 0.008 |
| Item-based CF | 0.153 | 0.001 | 0.163 | 0.083 | 0.002 |
| Content-based Filter | 0.000 | 0.000 | 0.000 | 0.000 | 0.215 |
| SVD | 0.220 | 0.002 | 0.220 | 0.096 | 0.002 |
| NMF | 0.233 | 0.002 | 0.234 | 0.102 | 0.002 |
| Hybrid | 0.220 | 0.002 | 0.220 | 0.096 | 0.002 |
| Path Signature | 0.127 | 0.001 | 0.110 | 0.066 | 0.003 |

See `significance_matrix.csv` for pairwise adjusted p-values and `effect_sizes.csv` for effect sizes (r).
# Ablation Study Summary Report

Generated on: 2026-01-12 19:41:23

## 1. Path Signature Order Analysis

- Order 1: Precision@5 = 0.4667
- Order 2: Precision@5 = 0.5048
- Order 3: Precision@5 = 0.5048

**Best performing order: 2 (Precision@5 = 0.5048)**

## 2. Temperature Scaling Analysis

- Temperature 0.1: Precision@5 = 0.5048
- Temperature 0.5: Precision@5 = 0.5048
- Temperature 1.0: Precision@5 = 0.5048
- Temperature 2.0: Precision@5 = 0.5048
- Temperature 5.0: Precision@5 = 0.5048

**Best performing temperature: 0.1 (Precision@5 = 0.5048)**

## 3. Feature Combination Analysis

- mfccs: Precision@5 = 0.5048
- chroma: Precision@5 = 0.5048
- spectral_centroid: Precision@5 = 0.5048
- spectral_bandwidth: Precision@5 = 0.5048
- zero_crossing_rate: Precision@5 = 0.5048

**Best feature combination: mfccs (Precision@5 = 0.5048)**

## 4. Similarity Metric Analysis

- cosine: Precision@5 = 0.1525
- euclidean: Precision@5 = 0.1525
- manhattan: Precision@5 = 0.1412

**Best similarity metric: cosine (Precision@5 = 0.1525)**

## 5. Component Contribution Analysis

- path_signatures_only: Precision@5 = 0.1525
- temperature_scaling_only: Precision@5 = 0.1525

## Key Findings

**Best overall configuration: signature_orders: 2 (Precision@5 = 0.5048)**

### Recommendations:
1. Use the optimal path signature order identified above
2. Apply the best temperature scaling value
3. Include the most effective feature combinations
4. Choose the best performing similarity metric
5. Consider the contribution of each component
# Comprehensive Model Performance Summary

This report consolidates performance metrics from all model evaluation reports.

## 1. Overall Model Performance Ranking

Models ranked by average F1-Score across all test scenarios:

|   Rank | Model            |   Avg F1-Score |   Avg Precision |   Avg Recall |   Avg Accuracy |   Best F1 |   Worst F1 |   Tests |
|-------:|:-----------------|---------------:|----------------:|-------------:|---------------:|----------:|-----------:|--------:|
|      1 | XGBOOST          |         0.6131 |          0.625  |       0.6028 |         0.6339 |    1      |     0.1467 |      24 |
|      2 | RANDOM_FOREST    |         0.5998 |          0.6056 |       0.5951 |         0.6352 |    1      |     0.0646 |      24 |
|      3 | PCA_IFOREST      |         0.5957 |          0.672  |       0.585  |         0.689  |    0.9132 |     0.2774 |      72 |
|      4 | KNN              |         0.5588 |          0.6661 |       0.5313 |         0.6762 |    1      |     0.0906 |      24 |
|      5 | LOF              |         0.5294 |          0.7312 |       0.53   |         0.6809 |    0.9888 |     0.088  |      24 |
|      6 | ISOLATION_FOREST |         0.4471 |          0.7381 |       0.424  |         0.6819 |    0.9687 |     0.0801 |      24 |
|      7 | MLP              |         0.3646 |          0.4562 |       0.3937 |         0.5987 |    0.9897 |     0      |      24 |
|      8 | AUTOENCODER      |         0.1346 |          0.7204 |       0.0743 |         0.5227 |    0.1905 |     0.0929 |      24 |

## 2. Performance by Model Type

### Supervised Models

| Model         |   Avg F1-Score |   Avg Accuracy |   Tests |
|:--------------|---------------:|---------------:|--------:|
| XGBOOST       |         0.6131 |         0.6339 |      24 |
| RANDOM_FOREST |         0.5998 |         0.6352 |      24 |
| KNN           |         0.5588 |         0.6762 |      24 |
| MLP           |         0.3646 |         0.5987 |      24 |

### Unsupervised Models

| Model            |   Avg F1-Score |   Avg Accuracy |   Tests |
|:-----------------|---------------:|---------------:|--------:|
| PCA_IFOREST      |         0.5957 |         0.689  |      72 |
| LOF              |         0.5294 |         0.6809 |      24 |
| ISOLATION_FOREST |         0.4471 |         0.6819 |      24 |
| AUTOENCODER      |         0.1346 |         0.5227 |      24 |

## 3. Performance by Frequency (FM vs SMA)

| Model            |   FM F1-Score |   SMA F1-Score |   FM Accuracy |   SMA Accuracy |   FM Tests |   SMA Tests |
|:-----------------|--------------:|---------------:|--------------:|---------------:|-----------:|------------:|
| PCA_IFOREST      |        0.535  |         0.6563 |        0.6525 |         0.7255 |         36 |          36 |
| RANDOM_FOREST    |        0.5462 |         0.6533 |        0.5697 |         0.7006 |         12 |          12 |
| LOF              |        0.4102 |         0.6486 |        0.6777 |         0.6842 |         12 |          12 |
| XGBOOST          |        0.5811 |         0.6451 |        0.5913 |         0.6765 |         12 |          12 |
| KNN              |        0.5521 |         0.5655 |        0.6363 |         0.716  |         12 |          12 |
| ISOLATION_FOREST |        0.4312 |         0.463  |        0.6632 |         0.7006 |         12 |          12 |
| MLP              |        0.4519 |         0.2773 |        0.6328 |         0.5647 |         12 |          12 |
| AUTOENCODER      |        0.1343 |         0.1349 |        0.5235 |         0.5219 |         12 |          12 |

## 4. Performance by Noise Type

| Noise Type   |   Avg F1-Score |   Best F1-Score |   Worst F1-Score |   Models Tested |
|:-------------|---------------:|----------------:|-----------------:|----------------:|
| RUIDO        |         0.8522 |          1      |           0.1834 |              16 |
| BLOCKING     |         0.6167 |          0.991  |           0.1364 |              16 |
| DROPOUT      |         0.3129 |          0.5886 |           0.0952 |              16 |
| SPURIA       |         0.1397 |          0.4391 |           0      |              16 |

## 5. Key Insights and Recommendations

### Performance Highlights

- **Best Overall Model**: XGBOOST (F1-Score: 0.6131)
- **Worst Overall Model**: AUTOENCODER (F1-Score: 0.1346)
- **Performance Gap**: 0.4785 F1-Score difference

- **Frequency Performance**: SMA (0.5055) vs FM (0.4552) average F1-Score
  - SMA frequency shows 11.0% better performance

### Recommendations

1. **For Production Use**: Consider {best_model.upper()} for best overall performance
2. **For Robustness**: Random Forest and XGBoost show consistent performance across scenarios
3. **For Specific Noise Types**: 
   - RUIDO: All models perform excellently (F1 ≈ 1.0)
   - SPURIA: Most challenging noise type, requires careful model selection
   - DROPOUT: Moderate difficulty, ensemble methods recommended
   - BLOCKING: Generally well-handled by most models
4. **Frequency Considerations**: SMA frequency generally shows better performance than FM

## 6. Detailed Model Performance Tables

### XGBOOST Model

| Frequency   | Noise Type   |   F1-Score |   Precision |   Recall |   Accuracy |   Tests |
|:------------|:-------------|-----------:|------------:|---------:|-----------:|--------:|
| FM          | BLOCKING     |     0.7388 |      0.7419 |   0.7359 |     0.743  |       3 |
| FM          | DROPOUT      |     0.3933 |      0.4109 |   0.3774 |     0.4068 |       3 |
| FM          | RUIDO        |     1      |      1      |   1      |     1      |       3 |
| FM          | SPURIA       |     0.1922 |      0.2025 |   0.1832 |     0.2155 |       3 |
| SMA         | BLOCKING     |     0.9815 |      1      |   0.9649 |     0.9825 |       3 |
| SMA         | DROPOUT      |     0.4565 |      0.4766 |   0.4386 |     0.4737 |       3 |
| SMA         | RUIDO        |     0.9956 |      1      |   0.9912 |     0.9956 |       3 |
| SMA         | SPURIA       |     0.1467 |      0.1677 |   0.1316 |     0.2544 |       3 |

### RANDOM_FOREST Model

| Frequency   | Noise Type   |   F1-Score |   Precision |   Recall |   Accuracy |   Tests |
|:------------|:-------------|-----------:|------------:|---------:|-----------:|--------:|
| FM          | BLOCKING     |     0.7164 |      0.7189 |   0.7139 |     0.7197 |       3 |
| FM          | DROPOUT      |     0.3864 |      0.3912 |   0.3819 |     0.4143 |       3 |
| FM          | RUIDO        |     1      |      1      |   1      |     1      |       3 |
| FM          | SPURIA       |     0.0819 |      0.0877 |   0.077  |     0.1447 |       3 |
| SMA         | BLOCKING     |     0.991  |      1      |   0.9825 |     0.9912 |       3 |
| SMA         | DROPOUT      |     0.5577 |      0.5632 |   0.5526 |     0.5658 |       3 |
| SMA         | RUIDO        |     1      |      1      |   1      |     1      |       3 |
| SMA         | SPURIA       |     0.0646 |      0.0839 |   0.0526 |     0.2456 |       3 |

### PCA_IFOREST Model

| Frequency   | Noise Type   |   F1-Score |   Precision |   Recall |   Accuracy |   Tests |
|:------------|:-------------|-----------:|------------:|---------:|-----------:|--------:|
| FM          | BLOCKING     |     0.6181 |      0.705  |   0.5849 |     0.6935 |       9 |
| FM          | DROPOUT      |     0.3288 |      0.5383 |   0.238  |     0.5194 |       9 |
| FM          | RUIDO        |     0.9113 |      0.8371 |   1      |     0.9026 |       9 |
| FM          | SPURIA       |     0.2816 |      0.4846 |   0.1993 |     0.4944 |       9 |
| SMA         | BLOCKING     |     0.8662 |      0.8106 |   0.9328 |     0.8582 |       9 |
| SMA         | DROPOUT      |     0.5686 |      0.664  |   0.5263 |     0.6477 |       9 |
| SMA         | RUIDO        |     0.9132 |      0.8418 |   1      |     0.9035 |       9 |
| SMA         | SPURIA       |     0.2774 |      0.4947 |   0.1988 |     0.4927 |       9 |

### KNN Model

| Frequency   | Noise Type   |   F1-Score |   Precision |   Recall |   Accuracy |   Tests |
|:------------|:-------------|-----------:|------------:|---------:|-----------:|--------:|
| FM          | BLOCKING     |     0.7623 |      0.7959 |   0.7327 |     0.7799 |       3 |
| FM          | DROPOUT      |     0.3056 |      0.4988 |   0.2324 |     0.4418 |       3 |
| FM          | RUIDO        |     1      |      1      |   1      |     1      |       3 |
| FM          | SPURIA       |     0.1403 |      0.1944 |   0.1275 |     0.3236 |       3 |
| SMA         | BLOCKING     |     0.9589 |      0.9714 |   0.9474 |     0.9605 |       3 |
| SMA         | DROPOUT      |     0.2127 |      0.7333 |   0.1404 |     0.5    |       3 |
| SMA         | RUIDO        |     1      |      1      |   1      |     1      |       3 |
| SMA         | SPURIA       |     0.0906 |      0.1346 |   0.0702 |     0.4035 |       3 |

### LOF Model

| Frequency   | Noise Type   |   F1-Score |   Precision |   Recall |   Accuracy |   Tests |
|:------------|:-------------|-----------:|------------:|---------:|-----------:|--------:|
| FM          | BLOCKING     |     0.4555 |      0.8505 |   0.4065 |     0.6906 |       3 |
| FM          | DROPOUT      |     0.1083 |      0.7803 |   0.0583 |     0.5207 |       3 |
| FM          | RUIDO        |     0.9888 |      0.9779 |   1      |     0.9887 |       3 |
| FM          | SPURIA       |     0.088  |      0.6424 |   0.0472 |     0.5107 |       3 |
| SMA         | BLOCKING     |     0.6921 |      0.654  |   0.7719 |     0.7105 |       3 |
| SMA         | DROPOUT      |     0.5886 |      0.6224 |   0.5877 |     0.6403 |       3 |
| SMA         | RUIDO        |     0.8746 |      0.7783 |   1      |     0.8552 |       3 |
| SMA         | SPURIA       |     0.4391 |      0.5442 |   0.3684 |     0.5307 |       3 |

### ISOLATION_FOREST Model

| Frequency   | Noise Type   |   F1-Score |   Precision |   Recall |   Accuracy |   Tests |
|:------------|:-------------|-----------:|------------:|---------:|-----------:|--------:|
| FM          | BLOCKING     |     0.4806 |      0.727  |   0.4343 |     0.6829 |       3 |
| FM          | DROPOUT      |     0.1475 |      0.522  |   0.0861 |     0.5032 |       3 |
| FM          | RUIDO        |     0.9687 |      0.9393 |   1      |     0.9676 |       3 |
| FM          | SPURIA       |     0.1279 |      0.4907 |   0.0738 |     0.499  |       3 |
| SMA         | BLOCKING     |     0.7173 |      0.8401 |   0.7018 |     0.8289 |       3 |
| SMA         | DROPOUT      |     0.1    |      1      |   0.0526 |     0.5263 |       3 |
| SMA         | RUIDO        |     0.9544 |      0.9135 |   1      |     0.9518 |       3 |
| SMA         | SPURIA       |     0.0801 |      0.4722 |   0.0438 |     0.4956 |       3 |

### MLP Model

| Frequency   | Noise Type   |   F1-Score |   Precision |   Recall |   Accuracy |   Tests |
|:------------|:-------------|-----------:|------------:|---------:|-----------:|--------:|
| FM          | BLOCKING     |     0.3873 |      0.4804 |   0.4188 |     0.5505 |       3 |
| FM          | DROPOUT      |     0.4099 |      0.659  |   0.556  |     0.49   |       3 |
| FM          | RUIDO        |     0.9897 |      1      |   0.9799 |     0.99   |       3 |
| FM          | SPURIA       |     0.0207 |      0.1771 |   0.011  |     0.5006 |       3 |
| SMA         | BLOCKING     |     0.2222 |      0.1667 |   0.3333 |     0.5    |       3 |
| SMA         | DROPOUT      |     0.2222 |      0.1667 |   0.3333 |     0.5    |       3 |
| SMA         | RUIDO        |     0.6649 |      1      |   0.5175 |     0.7588 |       3 |
| SMA         | SPURIA       |     0      |      0      |   0      |     0.5    |       3 |

### AUTOENCODER Model

| Frequency   | Noise Type   |   F1-Score |   Precision |   Recall |   Accuracy |   Tests |
|:------------|:-------------|-----------:|------------:|---------:|-----------:|--------:|
| FM          | BLOCKING     |     0.1364 |      0.7436 |   0.0751 |     0.5246 |       3 |
| FM          | DROPOUT      |     0.1246 |      0.6795 |   0.0686 |     0.5181 |       3 |
| FM          | RUIDO        |     0.1834 |      1      |   0.101  |     0.5505 |       3 |
| FM          | SPURIA       |     0.0929 |      0.5064 |   0.0511 |     0.5007 |       3 |
| SMA         | BLOCKING     |     0.1429 |      0.75   |   0.079  |     0.5263 |       3 |
| SMA         | DROPOUT      |     0.0952 |      0.5    |   0.0526 |     0.5    |       3 |
| SMA         | RUIDO        |     0.1905 |      1      |   0.1053 |     0.5526 |       3 |
| SMA         | SPURIA       |     0.1111 |      0.5833 |   0.0614 |     0.5088 |       3 |


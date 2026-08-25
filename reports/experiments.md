# Experiments

## 1. Model Comparison

Three models were evaluated on the validation set.

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Logistic Regression | 0.912281 | 0.684211 | 0.781955 |
| KNN | 0.925373 | 0.815789 | 0.867133 |
| Decision Tree | 0.707317 | 0.763158 | 0.734177 |

KNN achieved the best overall validation performance.

Decision Tree achieved lower Precision and F1-score than KNN, so it was not selected as the final model.


## 2. Cross Validation

Stratified 5-fold cross-validation was used.

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Logistic Regression | 0.863288 | 0.605649 | 0.709644 |
| KNN | 0.915727 | 0.764632 | 0.831843 |
| Decision Tree | 0.758303 | 0.743123 | 0.749516 |

KNN achieved the strongest overall cross-validation performance.


## 3. Scaling Experiment

KNN was tested with and without feature scaling.

| Version | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Without Scaling | 1.000000 | 0.013158 | 0.025974 |
| With Scaling | 0.925373 | 0.815789 | 0.867133 |

Scaling had a major impact on KNN, especially on Recall and F1-score.

Decision Tree was not scaled because tree-based models are not sensitive to feature scale.


## 4. Hyperparameter Experiment

KNN was tested with different values of `k` using 5-fold cross-validation.

| k | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 1 | 0.888228 | 0.790982 | 0.835852 |
| 5 | 0.915727 | 0.764632 | 0.831843 |
| 20 | 0.853280 | 0.751404 | 0.797434 |

Although `k=1` achieved a slightly higher F1-score, `k=5` was selected because it provided higher Precision and similar overall performance.


## 5. Threshold Experiment

The KNN model with `k=5` was tested with different thresholds.

| Threshold | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.3 | 0.890411 | 0.855263 | 0.872483 |
| 0.5 | 0.925373 | 0.815789 | 0.867133 |
| 0.7 | 0.982759 | 0.750000 | 0.850746 |

Threshold `0.3` was selected because it achieved the highest Recall and F1-score.


## 6. Final Result

Final configuration:

- Model: `KNN`
- `k = 5`
- Scaling: `StandardScaler`
- Threshold: `0.3`

Final test results:

| Metric | Result |
|---|---:|
| Accuracy | 0.9994713283755683 |
| Precision | 0.9113924050632911 |
| Recall | 0.7578947368421053 |
| F1-score | 0.8275862068965517 |

Confusion Matrix:

```text
[[56644     7]
 [   23    72]]
```

The final model correctly detected `72` fraudulent transactions, missed `23`, and generated only `7` false positives.
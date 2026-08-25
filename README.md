# Credit Card Fraud Detection Pipeline

## 1. Problem Description

This project builds a machine learning pipeline to detect fraudulent credit card transactions.

- `0`: Legitimate
- `1`: Fraud

The original dataset contains `284807` transactions and `30` input features (`Time`, `V1` to `V28`, and `Amount`). Only about `0.17%` of transactions are fraudulent, so the dataset is highly imbalanced.

Because of this imbalance, `Precision`, `Recall`, `F1-score`, and Confusion Matrix are considered in addition to Accuracy.


## 2. Data Analysis

- Missing values: `0`
- Duplicate rows: `1081`
- Legitimate duplicates: `1062`
- Fraudulent duplicates: `19`

After removing duplicates:

- Total transactions: `283726`
- Legitimate: `283253`
- Fraudulent: `473`

A stratified `80/20` train-test split was used.


## 3. Initial Hypothesis

- Logistic Regression should provide a useful baseline.
- KNN should be highly affected by feature scaling.
- Decision Tree may have a higher risk of overfitting.
- Recall is important because missing fraud is costly.
- Accuracy alone may be misleading due to class imbalance.


## 4. Model Comparison

### Validation Results

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.999361 | 0.912281 | 0.684211 | 0.781955 |
| KNN | 0.999581 | 0.925373 | 0.815789 | 0.867133 |
| Decision Tree | 0.999075 | 0.707317 | 0.763158 | 0.734177 |

### 5-Fold Cross Validation

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Logistic Regression | 0.863288 | 0.605649 | 0.709644 |
| KNN | 0.915727 | 0.764632 | 0.831843 |
| Decision Tree | 0.758303 | 0.743123 | 0.749516 |

KNN achieved the strongest overall performance.


## 5. Scaling Experiment

| Version | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Without Scaling | 1.000000 | 0.013158 | 0.025974 |
| With Scaling | 0.925373 | 0.815789 | 0.867133 |

Scaling greatly improved KNN performance.

Decision Tree was not scaled because tree-based models are not sensitive to feature scale.


## 6. Hyperparameter Experiment

KNN was tested using 5-fold cross-validation.

| k | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 1 | 0.888228 | 0.790982 | 0.835852 |
| 5 | 0.915727 | 0.764632 | 0.831843 |
| 20 | 0.853280 | 0.751404 | 0.797434 |

Although `k=1` achieved a slightly higher F1-score, `k=5` was selected because it provided higher Precision, similar overall performance, and more useful probability values for threshold analysis.


## 7. Final Model Selection

Final configuration:

- Model: `KNN`
- `k = 5`
- Scaling: `StandardScaler`
- Threshold: `0.3`

### Threshold Results

| Threshold | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0.3 | 0.890411 | 0.855263 | 0.872483 |
| 0.5 | 0.925373 | 0.815789 | 0.867133 |
| 0.7 | 0.982759 | 0.750000 | 0.850746 |

Threshold `0.3` provided the highest Recall and F1-score.

### Final Test Results

- Accuracy: `0.9994713283755683`
- Precision: `0.9113924050632911`
- Recall: `0.7578947368421053`
- F1-score: `0.8275862068965517`
- Confusion Matrix: `[[56644, 7], [23, 72]]`


## 8. Running Instructions

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the dataset from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Then place `creditcard.csv` at:

```text
data/creditcard.csv
```

Train the model:

```bash
python src/train.py
```

Run prediction:

```bash
python src/predict.py
```


## 9. Reflection

### Question 1

**Why is Accuracy misleading?**

Because the dataset is highly imbalanced, a model can achieve high Accuracy while detecting very little fraud.

### Question 2

**What is the trade-off between detecting more fraud and generating more false alarms?**

Lowering the threshold increases Recall and detects more fraud, but may also increase False Positives.

### Question 3

**What would you improve with one additional week?**

I would test more models and hyperparameters, investigate class imbalance techniques, and perform more detailed threshold optimization.
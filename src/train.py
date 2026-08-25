#%% import libraries

import io
import joblib
import pandas as pd
from contextlib import redirect_stdout
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


#%% prepare data

with redirect_stdout(io.StringIO()):
    from src.data_prep import prepare_data
X_train, X_test, y_train, y_test, _, _, _ = prepare_data()
print(f"train: {X_train.shape}, test: {X_test.shape}")


#%% 

X_train_inner, X_val, y_train_inner, y_val = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42)
print(f"inner train: {X_train_inner.shape}, validation: {X_val.shape}")


#%% 

validation_scaler = StandardScaler()
X_train_inner_scaled = validation_scaler.fit_transform(X_train_inner)
X_val_scaled = validation_scaler.transform(X_val)


#%% logistic regression

logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train_inner_scaled, y_train_inner)
y_pred_logistic = logistic_model.predict(X_val_scaled)


#%% logistic regression evaluation

accuracy_logistic = accuracy_score(y_val, y_pred_logistic)
precision_logistic = precision_score(y_val, y_pred_logistic)
recall_logistic = recall_score(y_val, y_pred_logistic)
f1_logistic = f1_score(y_val, y_pred_logistic)
confusion_logistic = confusion_matrix(y_val, y_pred_logistic)

print(f"accuracy: {accuracy_logistic}")
print(f"precision: {precision_logistic}")
print(f"recall: {recall_logistic}")
print(f"f1 score: {f1_logistic}")
print("\nconfusion matrix:")
print(confusion_logistic)


#%% knn

knn_model = KNeighborsClassifier()
knn_model.fit(X_train_inner_scaled, y_train_inner)
y_pred_knn = knn_model.predict(X_val_scaled)


#%% knn evaluation

accuracy_knn = accuracy_score(y_val, y_pred_knn)
precision_knn = precision_score(y_val, y_pred_knn)
recall_knn = recall_score(y_val, y_pred_knn)
f1_knn = f1_score(y_val, y_pred_knn)
confusion_knn = confusion_matrix(y_val, y_pred_knn)

print(f"accuracy: {accuracy_knn}")
print(f"precision: {precision_knn}")
print(f"recall: {recall_knn}")
print(f"f1 score: {f1_knn}")
print("\nconfusion matrix:")
print(confusion_knn)


#%% decision tree

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train_inner, y_train_inner)
y_pred_tree = tree_model.predict(X_val)


#%% decision tree evaluation

accuracy_tree = accuracy_score(y_val, y_pred_tree)
precision_tree = precision_score(y_val, y_pred_tree)
recall_tree = recall_score(y_val, y_pred_tree)
f1_tree = f1_score(y_val, y_pred_tree)
confusion_tree = confusion_matrix(y_val, y_pred_tree)

print(f"accuracy: {accuracy_tree}")
print(f"precision: {precision_tree}")
print(f"recall: {recall_tree}")
print(f"f1 score: {f1_tree}")
print("\nconfusion matrix:")
print(confusion_tree)


#%% model comparison

results = pd.DataFrame({
    "model": ["logistic regression", "knn", "decision tree"],
    "accuracy": [accuracy_logistic, accuracy_knn, accuracy_tree],
    "precision": [precision_logistic, precision_knn, precision_tree],
    "recall": [recall_logistic, recall_knn, recall_tree],
    "f1": [f1_logistic, f1_knn, f1_tree]})

print(results)


#%% cross validation

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"precision": "precision", "recall": "recall", "f1": "f1"}

logistic_cv_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000, random_state=42))])

knn_cv_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier())])

tree_cv_model = DecisionTreeClassifier(random_state=42)

logistic_cv = cross_validate(logistic_cv_model, X_train, y_train, cv=cv, scoring=scoring)
knn_cv = cross_validate(knn_cv_model, X_train, y_train, cv=cv, scoring=scoring)
tree_cv = cross_validate(tree_cv_model, X_train, y_train, cv=cv, scoring=scoring)


#%% cross validation results

cv_results = pd.DataFrame({
    "model": ["logistic regression", "knn", "decision tree"],
    "precision": [
        logistic_cv["test_precision"].mean(),
        knn_cv["test_precision"].mean(),
        tree_cv["test_precision"].mean()],
    "recall": [
        logistic_cv["test_recall"].mean(),
        knn_cv["test_recall"].mean(),
        tree_cv["test_recall"].mean()],
    "f1": [
        logistic_cv["test_f1"].mean(),
        knn_cv["test_f1"].mean(),
        tree_cv["test_f1"].mean()]})

print(cv_results)


#%% scaling experiment

knn_unscaled = KNeighborsClassifier()
knn_unscaled.fit(X_train_inner, y_train_inner)
y_pred_knn_unscaled = knn_unscaled.predict(X_val)


#%% scaling experiment results

scaling_results = pd.DataFrame({
    "model": ["knn without scaling", "knn with scaling"],
    "precision": [precision_score(y_val, y_pred_knn_unscaled), precision_knn],
    "recall": [recall_score(y_val, y_pred_knn_unscaled), recall_knn],
    "f1": [f1_score(y_val, y_pred_knn_unscaled), f1_knn]})

print(scaling_results)


#%% knn hyperparameter experiment

k_values = [1, 5, 20]
knn_tuning_results = []

for k in k_values:
    knn_tuning_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=k))])
    scores = cross_validate(knn_tuning_model, X_train, y_train, cv=cv, scoring=scoring)
    knn_tuning_results.append({
        "k": k,
        "precision": scores["test_precision"].mean(),
        "recall": scores["test_recall"].mean(),
        "f1": scores["test_f1"].mean()})
knn_tuning_results = pd.DataFrame(knn_tuning_results)

print(knn_tuning_results)


#%% 

best_k = 5

#%% threshold model

threshold_model = KNeighborsClassifier(n_neighbors=best_k)
threshold_model.fit(X_train_inner_scaled, y_train_inner)
y_prob_val = threshold_model.predict_proba(X_val_scaled)[:, 1]


#%% threshold experiment

thresholds = [0.3, 0.5, 0.7]
threshold_results = []

for threshold in thresholds:
    y_pred_threshold = (y_prob_val >= threshold).astype(int)

    threshold_results.append({
        "threshold": threshold,
        "precision": precision_score(y_val, y_pred_threshold),
        "recall": recall_score(y_val, y_pred_threshold),
        "f1": f1_score(y_val, y_pred_threshold)})
threshold_results = pd.DataFrame(threshold_results)

print(threshold_results)


#%% best threshold

final_threshold = float(
    threshold_results.loc[threshold_results["f1"].idxmax(), "threshold"])

print("final model: knn")
print(f"k: {best_k}")
print(f"threshold: {final_threshold}")


#%% train final model

final_scaler = StandardScaler()
X_train_final_scaled = final_scaler.fit_transform(X_train)
X_test_final_scaled = final_scaler.transform(X_test)

final_model = KNeighborsClassifier(n_neighbors=best_k)
final_model.fit(X_train_final_scaled, y_train)


#%% final model evaluation

y_prob_final = final_model.predict_proba(X_test_final_scaled)[:, 1]
y_pred_final = (y_prob_final >= final_threshold).astype(int)

final_accuracy = accuracy_score(y_test, y_pred_final)
final_precision = precision_score(y_test, y_pred_final)
final_recall = recall_score(y_test, y_pred_final)
final_f1 = f1_score(y_test, y_pred_final)
final_confusion = confusion_matrix(y_test, y_pred_final)

print(f"accuracy: {final_accuracy}")
print(f"precision: {final_precision}")
print(f"recall: {final_recall}")
print(f"f1 score: {final_f1}")
print("\nconfusion matrix:")
print(final_confusion)


#%% save final model

joblib.dump(final_model, "models/model.pkl")
joblib.dump(final_scaler, "models/scaler.pkl")

print("model and scaler saved")
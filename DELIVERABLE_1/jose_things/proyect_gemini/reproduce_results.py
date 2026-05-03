import pandas as pd
import sklearn
import joblib
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from utils import get_features_from_df, get_advanced_features

# 1. Load Data
test_df = pd.read_csv("test_df.csv")
y_test = test_df["is_duplicate"].values

# 2. Load Models
models_dir = "models"
logistic_simple = joblib.load(os.path.join(models_dir, "logistic_simple.joblib"))
logistic_advanced = joblib.load(os.path.join(models_dir, "logistic_advanced.joblib"))
count_vectorizer = joblib.load(os.path.join(models_dir, "count_vectorizer.joblib"))

# 3. Evaluate Simple Model
X_te_simple = get_features_from_df(test_df, count_vectorizer)
y_pred_simple = logistic_simple.predict(X_te_simple)
y_prob_simple = logistic_simple.predict_proba(X_te_simple)[:, 1]

roc_simple = roc_auc_score(y_test, y_prob_simple)
prec_simple = precision_score(y_test, y_pred_simple)
rec_simple = recall_score(y_test, y_pred_simple)

print("Simple Model Results:")
print(f"ROC AUC: {roc_simple:.4f}")
print(f"Precision: {prec_simple:.4f}")
print(f"Recall: {rec_simple:.4f}")

# 4. Evaluate Advanced Model
X_te_advanced = get_advanced_features(test_df, count_vectorizer)
y_pred_advanced = logistic_advanced.predict(X_te_advanced)
y_prob_advanced = logistic_advanced.predict_proba(X_te_advanced)[:, 1]

roc_adv = roc_auc_score(y_test, y_prob_advanced)
prec_adv = precision_score(y_test, y_pred_advanced)
rec_adv = recall_score(y_test, y_pred_advanced)

print("\nAdvanced Model Results:")
print(f"ROC AUC: {roc_adv:.4f}")
print(f"Precision: {prec_adv:.4f}")
print(f"Recall: {rec_adv:.4f}")

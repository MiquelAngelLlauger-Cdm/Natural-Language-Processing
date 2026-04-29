import pandas as pd
import sklearn
import joblib
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sentence_transformers import SentenceTransformer
from utils import get_features_from_df, get_sbert_features

test_df = pd.read_csv("test_df.csv")
y_test = test_df["is_duplicate"].values

models_dir = "models"
logistic_simple = joblib.load(os.path.join(models_dir, "logistic_simple.joblib"))
logistic_advanced = joblib.load(os.path.join(models_dir, "logistic_advanced.joblib"))
count_vectorizer = joblib.load(os.path.join(models_dir, "count_vectorizer.joblib"))

X_te_simple = get_features_from_df(test_df, count_vectorizer)
y_pred_simple = logistic_simple.predict(X_te_simple)
y_prob_simple = logistic_simple.predict_proba(X_te_simple)[:, 1]

print(f"Simple Model ROC AUC: {roc_auc_score(y_test, y_prob_simple):.4f}")
print(f"Simple Model Precision: {precision_score(y_test, y_pred_simple):.4f}")
print(f"Simple Model Recall: {recall_score(y_test, y_pred_simple):.4f}")

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
X_te_advanced = get_sbert_features(test_df, sbert_model)
y_pred_advanced = logistic_advanced.predict(X_te_advanced)
y_prob_advanced = logistic_advanced.predict_proba(X_te_advanced)[:, 1]

print(f"Advanced Model ROC AUC: {roc_auc_score(y_test, y_prob_advanced):.4f}")
print(f"Advanced Model Precision: {precision_score(y_test, y_pred_advanced):.4f}")
print(f"Advanced Model Recall: {recall_score(y_test, y_pred_advanced):.4f}")


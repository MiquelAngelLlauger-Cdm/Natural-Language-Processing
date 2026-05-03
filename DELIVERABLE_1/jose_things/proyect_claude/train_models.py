"""
Equivalent of train_models.ipynb — run this to populate the models/ folder.
Usage: python train_models.py
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from utils import extract_simple_features, extract_improved_features

# ── 1. Load data ──────────────────────────────────────────────────────────────
candidate_paths = [
    os.path.expanduser('~/Datasets/QuoraQuestionPairs/quora_data.csv'),
    '../nlp_deliv1_materials/quora_data.csv',
    './quora_data.csv',
]
data_path = next((p for p in candidate_paths if os.path.exists(p)), None)
if data_path is None:
    raise FileNotFoundError('quora_data.csv not found.')

print(f'Loading data from: {data_path}')
quora_df = pd.read_csv(data_path)

# ── 2. Split ──────────────────────────────────────────────────────────────────
A_df, test_df   = train_test_split(quora_df,  test_size=0.05, random_state=123)
train_df, val_df = train_test_split(A_df,     test_size=0.05, random_state=123)
print(f'train={train_df.shape}  val={val_df.shape}  test={test_df.shape}')

train_df.to_csv('train_df.csv', index=False)
val_df.to_csv('val_df.csv',     index=False)
test_df.to_csv('test_df.csv',   index=False)

y_train = train_df['is_duplicate'].values
y_val   = val_df['is_duplicate'].values

all_train_questions = np.concatenate([
    train_df['question1'].fillna('').astype(str).values,
    train_df['question2'].fillna('').astype(str).values,
])

# ── 3. Simple model ───────────────────────────────────────────────────────────
print('\n[Simple model] Fitting TF-IDF...')
tfidf_simple = TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                               max_features=50_000, sublinear_tf=True)
tfidf_simple.fit(all_train_questions)

X_tr_s = extract_simple_features(train_df, tfidf_simple)
X_va_s = extract_simple_features(val_df,   tfidf_simple)

simple_lr = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=123)
simple_lr.fit(X_tr_s, y_train)

auc_s  = roc_auc_score(y_val, simple_lr.predict_proba(X_va_s)[:, 1])
prec_s = precision_score(y_val, simple_lr.predict(X_va_s))
rec_s  = recall_score(y_val, simple_lr.predict(X_va_s))
print(f'  Val  ROC AUC={auc_s:.4f}  Prec={prec_s:.4f}  Rec={rec_s:.4f}')

# ── 4. Improved model ─────────────────────────────────────────────────────────
print('\n[Improved model] Fitting TF-IDF vectorizers...')
tfidf_word = TfidfVectorizer(ngram_range=(1, 2), min_df=3,
                             max_features=50_000, sublinear_tf=True)
tfidf_word.fit(all_train_questions)

tfidf_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3),
                             min_df=5, max_features=50_000, sublinear_tf=True)
tfidf_char.fit(all_train_questions)

print('  Extracting training features (may take 3-5 minutes)...')
X_tr_i = extract_improved_features(train_df, tfidf_word, tfidf_char)
print('  Extracting validation features...')
X_va_i = extract_improved_features(val_df, tfidf_word, tfidf_char)
print(f'  Feature matrix shape: {X_tr_i.shape}')

improved_gbm = HistGradientBoostingClassifier(
    max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=123)
improved_gbm.fit(X_tr_i, y_train)

auc_i  = roc_auc_score(y_val, improved_gbm.predict_proba(X_va_i)[:, 1])
prec_i = precision_score(y_val, improved_gbm.predict(X_va_i))
rec_i  = recall_score(y_val, improved_gbm.predict(X_va_i))
print(f'  Val  ROC AUC={auc_i:.4f}  Prec={prec_i:.4f}  Rec={rec_i:.4f}')

# ── 5. Save ───────────────────────────────────────────────────────────────────
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

artifacts = {
    'tfidf_simple.joblib':  tfidf_simple,
    'simple_lr.joblib':     simple_lr,
    'tfidf_word.joblib':    tfidf_word,
    'tfidf_char.joblib':    tfidf_char,
    'improved_gbm.joblib':  improved_gbm,
}

all_exist = all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in artifacts)
if all_exist:
    print('\nAll model files already exist — skipping save.')
else:
    print('\nSaving models...')
    for fname, obj in artifacts.items():
        joblib.dump(obj, os.path.join(MODELS_DIR, fname))
        print(f'  Saved {fname}')
    print('Done.')

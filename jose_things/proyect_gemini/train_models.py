import pandas as pd
import sklearn
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from utils import cast_list_as_strings, get_features_from_df, get_advanced_features

# 1. Load Data
data_path = "../nlp_deliv1_materials/quora_data.csv"
quora_df = pd.read_csv(data_path)

# 2. Split Data (as specified in the guide)
A_df, test_df = train_test_split(quora_df, test_size=0.05, random_state=123)
train_df, val_df = train_test_split(A_df, test_size=0.05, random_state=123)

print('train_df.shape=', train_df.shape)
print('val_df.shape=', val_df.shape)
print('test_df.shape=', test_df.shape)

# 3. Fit CountVectorizer
q1_train = cast_list_as_strings(list(train_df["question1"]))
q2_train = cast_list_as_strings(list(train_df["question2"]))
all_questions = q1_train + q2_train

count_vectorizer = CountVectorizer(ngram_range=(1,1))
count_vectorizer.fit(all_questions)

# 4. Train Simple Model
X_tr_simple = get_features_from_df(train_df, count_vectorizer)
y_train = train_df["is_duplicate"].values

logistic_simple = LogisticRegression(solver="liblinear", random_state=123)
logistic_simple.fit(X_tr_simple, y_train)

# 5. Train Improved Model
X_tr_advanced = get_advanced_features(train_df, count_vectorizer)
logistic_advanced = LogisticRegression(solver="liblinear", random_state=123)
logistic_advanced.fit(X_tr_advanced, y_train)

# 6. Save Models
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

joblib.dump(logistic_simple, os.path.join(models_dir, "logistic_simple.joblib"))
joblib.dump(logistic_advanced, os.path.join(models_dir, "logistic_advanced.joblib"))
joblib.dump(count_vectorizer, os.path.join(models_dir, "count_vectorizer.joblib"))

# Store dfs for evaluation later
train_df.to_csv("train_df.csv", index=False)
val_df.to_csv("val_df.csv", index=False)
test_df.to_csv("test_df.csv", index=False)

print("Models trained and saved successfully.")

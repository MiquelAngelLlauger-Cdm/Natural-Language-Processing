import json

# Update train_models.ipynb
with open('train_models.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = cell['source']
        for i, line in enumerate(source):
            if "CountVectorizer + Jaccard Similarity" in line:
                source[i] = "2. An improved model using SentenceBERT embeddings + Logistic Regression."
    
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "from utils import cast_list_as_strings, get_features_from_df, get_advanced_features" in line:
                source.insert(i, "from sentence_transformers import SentenceTransformer\n")
                source[i+1] = "from utils import cast_list_as_strings, get_features_from_df, get_sbert_features"
            
            if "X_tr_advanced = get_advanced_features(train_df, count_vectorizer)" in line:
                source.insert(i, "sbert_model = SentenceTransformer('all-MiniLM-L6-v2')\n")
                source[i+1] = "X_tr_advanced = get_sbert_features(train_df, sbert_model)\n"
            if "logistic_advanced = LogisticRegression(solver=\"liblinear\", random_state=123)" in line:
                source[i] = "logistic_advanced = LogisticRegression(solver=\"liblinear\", random_state=123, max_iter=500)\n"

with open('train_models.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

# Update reproduce_results.ipynb
with open('reproduce_results.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if "from utils import get_features_from_df, get_advanced_features" in line:
                source.insert(i, "from sentence_transformers import SentenceTransformer\n")
                source[i+1] = "from utils import get_features_from_df, get_sbert_features"
            
            if "X_te_advanced = get_advanced_features(test_df, count_vectorizer)" in line:
                source.insert(i, "sbert_model = SentenceTransformer('all-MiniLM-L6-v2')\n")
                source[i+1] = "X_te_advanced = get_sbert_features(test_df, sbert_model)\n"

with open('reproduce_results.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebooks updated successfully.")

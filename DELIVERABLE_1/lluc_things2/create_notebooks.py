import nbformat as nbf

def create_train_models_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Train Models\nThis notebook loads the data, extracts features using `utils.py`, and trains the simple and improved models."),
        nbf.v4.new_code_cell("import os\nimport pandas as pd\nimport sklearn.model_selection\nfrom sklearn.linear_model import LogisticRegression\nimport joblib\nfrom utils import extract_simple_features, extract_improved_features"),
        nbf.v4.new_markdown_cell("## Data Loading and Splitting"),
        nbf.v4.new_code_cell("quora_df = pd.read_csv(\"./quora_data.csv\")\n\nA_df, test_df = sklearn.model_selection.train_test_split(quora_df, test_size=0.05, random_state=123)\ntrain_df, val_df = sklearn.model_selection.train_test_split(A_df, test_size=0.05, random_state=123)\n\nprint('train_df.shape=', train_df.shape)\nprint('val_df.shape=', val_df.shape)\nprint('test_df.shape=', test_df.shape)"),
        nbf.v4.new_markdown_cell("## Simple Model\nUsing simple word overlap as feature."),
        nbf.v4.new_code_cell("X_train_simple = extract_simple_features(train_df)\nX_val_simple = extract_simple_features(val_df)\ny_train = train_df['is_duplicate'].values\ny_val = val_df['is_duplicate'].values\n\nmodel_simple = LogisticRegression()\nmodel_simple.fit(X_train_simple, y_train)\nprint(f\"Simple Model Accuracy on Val: {model_simple.score(X_val_simple, y_val):.4f}\")"),
        nbf.v4.new_markdown_cell("## Improved Model\nUsing word overlap, Jaccard similarity, and length differences."),
        nbf.v4.new_code_cell("X_train_improved = extract_improved_features(train_df)\nX_val_improved = extract_improved_features(val_df)\n\nmodel_improved = LogisticRegression()\nmodel_improved.fit(X_train_improved, y_train)\nprint(f\"Improved Model Accuracy on Val: {model_improved.score(X_val_improved, y_val):.4f}\")"),
        nbf.v4.new_markdown_cell("## Save Models\nSave the trained models to the `models` folder."),
        nbf.v4.new_code_cell("os.makedirs('models', exist_ok=True)\nif not os.path.exists('models/model_simple.pkl'):\n    joblib.dump(model_simple, 'models/model_simple.pkl')\nif not os.path.exists('models/model_improved.pkl'):\n    joblib.dump(model_improved, 'models/model_improved.pkl')\nprint('Models saved successfully.')")
    ]
    with open('train_models.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_reproduce_results_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Reproduce Results\nThis notebook loads the saved models and reports ROC AUC, Precision, and Recall on the train, validation, and test sets."),
        nbf.v4.new_code_cell("import os\nimport pandas as pd\nimport sklearn.model_selection\nfrom sklearn.metrics import roc_auc_score, precision_score, recall_score\nimport joblib\nfrom utils import extract_simple_features, extract_improved_features"),
        nbf.v4.new_markdown_cell("## Data Loading and Splitting"),
        nbf.v4.new_code_cell("quora_df = pd.read_csv(\"./quora_data.csv\")\nA_df, test_df = sklearn.model_selection.train_test_split(quora_df, test_size=0.05, random_state=123)\ntrain_df, val_df = sklearn.model_selection.train_test_split(A_df, test_size=0.05, random_state=123)"),
        nbf.v4.new_markdown_cell("## Feature Extraction"),
        nbf.v4.new_code_cell("X_train_simp = extract_simple_features(train_df)\nX_val_simp = extract_simple_features(val_df)\nX_test_simp = extract_simple_features(test_df)\n\nX_train_imp = extract_improved_features(train_df)\nX_val_imp = extract_improved_features(val_df)\nX_test_imp = extract_improved_features(test_df)\n\ny_train = train_df['is_duplicate'].values\ny_val = val_df['is_duplicate'].values\ny_test = test_df['is_duplicate'].values"),
        nbf.v4.new_markdown_cell("## Evaluation Function"),
        nbf.v4.new_code_cell("def evaluate_model(model, X, y):\n    preds = model.predict(X)\n    probs = model.predict_proba(X)[:, 1]\n    return {\n        'ROC_AUC': roc_auc_score(y, probs),\n        'Precision': precision_score(y, preds),\n        'Recall': recall_score(y, preds)\n    }"),
        nbf.v4.new_markdown_cell("## Load Models & Evaluate"),
        nbf.v4.new_code_cell("model_simple = joblib.load('models/model_simple.pkl')\nmodel_improved = joblib.load('models/model_improved.pkl')\n\nresults = {\n    'Simple_Train': evaluate_model(model_simple, X_train_simp, y_train),\n    'Simple_Val': evaluate_model(model_simple, X_val_simp, y_val),\n    'Simple_Test': evaluate_model(model_simple, X_test_simp, y_test),\n    'Improved_Train': evaluate_model(model_improved, X_train_imp, y_train),\n    'Improved_Val': evaluate_model(model_improved, X_val_imp, y_val),\n    'Improved_Test': evaluate_model(model_improved, X_test_imp, y_test)\n}\n\nresults_df = pd.DataFrame(results).T\ndisplay(results_df)")
    ]
    with open('reproduce_results.ipynb', 'w') as f:
        nbf.write(nb, f)

def create_utils_notebooks():
    # utils_Lluc
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Lluc's Utilities\nExplanation of the advanced feature extraction."),
        nbf.v4.new_code_cell("from utils import extract_improved_features\nimport pandas as pd\ndf = pd.DataFrame({'question1': ['How are you?'], 'question2': ['How are you doing?']})\ndisplay(extract_improved_features(df))")
    ]
    with open('utils_Lluc.ipynb', 'w') as f: nbf.write(nb, f)

    # utils_Jose
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Jose's Utilities\nExplanation of string distances."),
        nbf.v4.new_code_cell("from utils import compute_jaccard_similarity\nprint('Jaccard:', compute_jaccard_similarity('How are you?', 'How are you doing?'))")
    ]
    with open('utils_Jose.ipynb', 'w') as f: nbf.write(nb, f)

    # utils_Miki
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Miki's Utilities\nExplanation of data cleaning."),
        nbf.v4.new_code_cell("from utils import clean_text\nprint('Cleaned:', clean_text('How are you doing?!'))")
    ]
    with open('utils_Miki.ipynb', 'w') as f: nbf.write(nb, f)

if __name__ == '__main__':
    create_train_models_notebook()
    create_reproduce_results_notebook()
    create_utils_notebooks()

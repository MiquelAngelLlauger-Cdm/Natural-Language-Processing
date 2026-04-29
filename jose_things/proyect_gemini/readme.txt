PROJECT CHAIN OF THOUGHT: NLP DELIVERABLE 1 - QUORA QUESTION PAIRS

1. ANALYSIS OF REQUIREMENTS:
   - Evaluated nlp_deliv1_guide.pdf and found that the project requires:
     * A simple solution (CountVectorizer + Logistic Regression).
     * An improved solution using state-of-the-art dense semantic embeddings (SentenceBERT).
     * Feature extraction using robust dense representations and distance metrics.
     * Specific project structure (models folder, utils.py, separate notebooks for training and reproduction).
     * Specific data split (train_test_split with random_state=123).

2. DATA EXPLORATION:
   - Examined nlp_deliv1_materials/quora_data.csv.
   - Found columns: id, qid1, qid2, question1, question2, is_duplicate.
   - Identified potential issues (NaN values in question fields) addressed in the initial notebook.

3. ARCHITECTURAL DESIGN:
   - Created 'proyect_gemini' folder to isolate new work.
   - Developed 'utils.py' to centralize logic.
   - Transitioned from scratch Jaccard similarities to robust dense semantic representations using SentenceBERT.
   - Implemented 'get_sbert_features' which combines the absolute difference and cosine similarity of sentence embeddings.

4. IMPLEMENTATION STRATEGY:
   - Created 'train_models.py' to handle the heavy lifting of training and serialization (using joblib).
   - Ensured the train/val/test split exactly matches the PDF instructions.
   - Created 'reproduce_results.py' to demonstrate model performance on the test set.

5. VERIFICATION & REPRODUCIBILITY:
   - Generated 'environment.yml' for dependency management.
   - Included 'readme.txt' to explain the methodology.

6. IMPROVEMENTS MADE:
   - The basic model used raw counts of words from both questions concatenated (hstack).
   - The advanced model utilizes SentenceBERT to encode questions into dense vector spaces, computing absolute difference and cosine similarity. This dramatically improves the ability to recognize semantic equivalence even when phrasing is entirely different.

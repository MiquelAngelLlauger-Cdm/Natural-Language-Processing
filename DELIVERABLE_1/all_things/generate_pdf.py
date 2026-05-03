from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(80)
        self.cell(30, 10, 'Quora Challenge NLP Deliverable', 0, 0, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

pdf = PDF()
pdf.add_page()
pdf.set_font('Arial', '', 12)

# Team
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Team Members & Work Distribution:', 0, 1)
pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, 'We have built a collaborative pipeline where each member contributed specific advanced feature extraction methodologies to improve our Baseline. The work was distributed as follows:\n- Miki: Focused on the baseline data cleaning and advanced TF-IDF cosine similarities on character n-grams.\n- Jose: Investigated and coded advanced string distance theory, specifically using normalized Levenshtein Edit Distance.\n- Lluc: Extracted starting-word features to capture non-duplicate edge cases, and led the implementation of the final Random Forest and Decision Tree models.')
pdf.ln(5)

# Baseline Solution
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Baseline Model:', 0, 1)
pdf.set_font('Arial', '', 12)
text_simple = (
    "Our simple baseline solution relies on a Logistic Regression model trained on a single feature: basic Word Overlap. "
    "Limitations include that it ignores the order of words, synonyms, and context, often failing on questions that have "
    "the exact same words but completely different meanings (e.g., 'How do I read a book?' vs 'I read a book, how?'). "
    "It also does not properly weight the importance of rare words versus common stopwords."
)
pdf.multi_cell(0, 10, text_simple)
pdf.ln(5)

# Final Solution
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Final Combined Solution:', 0, 1)
pdf.set_font('Arial', '', 12)
text_final = (
    "To build a powerful final model, we combined three advanced techniques:\n\n"
    "1. TF-IDF Cosine Similarity (Miki): By fitting a TF-IDF vectorizer on character n-grams, we capture structural similarities and correctly weight rare sub-strings higher than common noise.\n\n"
    "2. Levenshtein Edit Distance (Jose): Standard overlap is rigid. We utilized the Levenshtein distance normalized by string length to robustly handle misspellings and subtle word variations. Crucially, as requested by the deliverable guidelines, we implemented this dynamic programming algorithm entirely from scratch along with all our tokenization strategies.\n\n"
    "3. First-Word & WH-word Matching (Lluc): After discussing with the professor, Daniel, we observed that questions with high word overlap but different starting WH-words (e.g., 'Who is...' vs 'Where is...') are rarely duplicates. We added binary features for matching first words and WH-word flags.\n\n"
    "Finally, we trained a Random Forest Classifier on this combined feature space, which successfully learned the non-linear interactions between these diverse metrics and significantly outperformed the baseline."
)
pdf.multi_cell(0, 10, text_final)

pdf.output('main.pdf', 'F')

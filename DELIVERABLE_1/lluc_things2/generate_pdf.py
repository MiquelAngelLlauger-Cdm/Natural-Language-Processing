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
pdf.cell(0, 10, 'Team Members:', 0, 1)
pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, 'The work was distributed among the team members as follows:\n- Miki: Focused on data cleaning and the basic simple solution/features.\n- Jose: Investigated and coded string distance features (Jaccard Similarity and Length Differences).\n- Lluc: Extracted advanced features, implemented the model training and evaluation notebooks.')
pdf.ln(5)

# Simple Solution
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Simple Solution:', 0, 1)
pdf.set_font('Arial', '', 12)
text_simple = (
    "Our simple solution relies on a basic logistic regression model trained on a single feature: Word Overlap. "
    "Limitations include that it ignores the order of words, synonyms, and context, often failing on questions that have "
    "the exact same words but completely different meanings (e.g., 'How do I read a book?' vs 'I read a book, how?'). "
    "This naive solution provides a low but useful baseline."
)
pdf.multi_cell(0, 10, text_simple)
pdf.ln(5)

# Improved Solution
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, 'Improved Solution:', 0, 1)
pdf.set_font('Arial', '', 12)
text_improved = (
    "To improve upon the simple solution, we incorporated string distance metrics. Specifically, we implemented "
    "Jaccard Similarity and Length Difference between the strings. We extracted these features for both questions "
    "and concatenated them alongside the Word Overlap feature. This captures similarity more robustly, helping the "
    "model detect slight variations or significantly mismatched lengths that often correlate with non-duplicates."
)
pdf.multi_cell(0, 10, text_improved)

pdf.output('main.pdf', 'F')

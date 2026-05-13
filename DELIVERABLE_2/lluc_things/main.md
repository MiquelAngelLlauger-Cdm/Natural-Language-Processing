# NLP Deliverable 2: Named Entity Recognition
**Team Members:** Miki, Jose, Lluc

## 1. Introduction
The objective of this project is to apply machine learning techniques to a Named Entity Recognition (NER) task. We compare a classical machine learning approach (CRF) with a modern deep learning approach (BiLSTM).

## 2. Exploratory Data Analysis
The dataset consists of sentences where each word is tagged with entity labels (e.g., B-per, I-per, B-geo, etc.) using the IOB format.
- **Training set size:** [Add number] sentences.
- **Test set size:** [Add number] sentences.
- **Entity distribution:** [Briefly describe the balance of entities].

## 3. Methodology
### 3.1 CRF (Conditional Random Fields)
We implemented a CRF model using `sklearn-crfsuite`. Features used include:
- Word identity (lowercase).
- Suffixes (last 2 and 3 characters).
- Shape features (isupper, istitle, isdigit).
- Context features (previous and next word properties).

### 3.2 Deep Learning (BiLSTM)
We implemented a Bidirectional LSTM using PyTorch. 
- **Embeddings:** Word embeddings learned from scratch.
- **Architecture:** BiLSTM layer followed by a Linear layer for classification.
- **Loss:** CrossEntropyLoss ignoring padding.

## 4. Experimental Setup
- **Evaluation Metrics:** Accuracy (excluding 'O' tags), F1-score (weighted), and Confusion Matrix.
- **Hyperparameters:** [Describe CRF c1/c2 and BiLSTM learning rate/epochs].

## 5. Results
### 5.1 Test Set Performance
| Model | Accuracy (non-O) | F1-score |
|-------|------------------|----------|
| CRF   | [Result]         | [Result] |
| BiLSTM| [Result]         | [Result] |

### 5.2 Confusion Matrix
[Insert images of confusion matrices here]

### 5.3 TINY TEST Accuracy
The models were evaluated on the provided TINY TEST dataset to check generalization on specific edge cases.

## 6. Conclusions
[Summarize which model performed better and why. Discuss the impact of features vs. automatic feature extraction.]

## 7. Team Member Contributions
- **Miki:** [Describe tasks]
- **Jose:** [Describe tasks]
- **Lluc:** [Describe tasks]

## 8. Use of AI
AI was used as a coding assistant to help with boilerplate code, feature extraction logic, and structuring the deliverable.

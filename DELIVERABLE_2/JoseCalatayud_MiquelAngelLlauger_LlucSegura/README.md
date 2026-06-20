# NER - Deliverable 2

Team: Jose Calatayud, Miquel Àngel Llauger, Lluc Segura

This package follows the required submission layout. `train_models.ipynb` contains the
complete EDA and training pipeline for three CRFs, BiLSTM, BiLSTM-CRF, BiLSTM-GloVe,
BiLSTM-GloVe-CRF, and DistilBERT. `reproduce_results.ipynb` performs evaluation only and
loads artifacts from `fitted_models/`.

## Reproduction

```bash
conda env create -f environment.yml
conda activate nlp_d2
jupyter nbconvert --to notebook --execute --inplace reproduce_results.ipynb
```

For retraining, execute `train_models.ipynb` from top to bottom. GPU execution is strongly
recommended for the neural models. GloVe training may download pretrained vectors.

The `source_notebooks/` directory preserves the original, individually executable stages.
The included DistilBERT directory contains its configuration and tokenizer; if weight files
are unavailable, the reproduction notebook skips that model and evaluates the remaining
supplied artifacts.

# NER — Deliverable 2 (jose_things)

Named Entity Recognition (NER) sobre el dataset `nlp_d2_data/`, comparando el enfoque
**clásico** (CRF con features hechas a mano) con el enfoque **moderno** de deep learning
(BiLSTM desde cero y DistilBERT fine-tuned). Todo el trabajo está organizado en notebooks
autocontenidos y numerados, pensados para ejecutarse de arriba a abajo.

---

## 1. El plan que hemos seguido

### Paso 0 — Entorno
Entorno conda limpio (`nlp_d2`, Python 3.10) con versiones compatibles
(`numpy<2`, `pandas`, `scikit-learn`, `sklearn-crfsuite`, `torch`, `transformers`,
`datasets`, `seqeval`, `jupyter`). Exportado a `environment.yml` para reproducibilidad.

### Paso 1 — `01_EDA.ipynb`
Análisis exploratorio: nº de frases/tokens, longitudes de frase, distribución de tags
(con y sin `O`), desbalance de clases, palabras fuera de vocabulario (OOV) y ejemplos de
entidades frecuentes por tipo.

### Paso 2 — `02_Model_CRF.ipynb` (ML clásico)
Tres CRF (`sklearn-crfsuite`) de complejidad de features creciente, para mostrar el
efecto de la ingeniería de features:
- **CRF v1 (baseline):** features mínimas (`bias`, `lower`, `isupper`, `istitle`, `isdigit`).
- **CRF v2 (mejorado):** + sufijos/prefijos, *word shape* y contexto ±1.
- **CRF v3 (fabuloso):** + contexto ±2, afijos multi-longitud, *word shape* normalizado,
  flags ortográficos y regularización ajustada (c1/c2).

Se guardan `crf_v1/v2/v3.pkl` en `fitted_models/`.

### Paso 3 — `03_Model_DL_BiLSTM.ipynb` (DL #1)
BiLSTM en PyTorch entrenado **desde cero** (embeddings aprendidos de los datos, sin
features manuales). Auto-detecta `cuda/mps/cpu`. Guarda `bilstm_model.pth` +
`dl_mappings.pkl`.

Con entrenamiento largo en Colab sobreajusta (train F1 ≈ 0.99 vs test F1 ≈ 0.52): el
test tiene palabras no vistas en train (OOV) y el modelo predice cada tag de forma
independiente. Dos notebooks lo mejoran **sin salir del mundo RNN**:

#### `03_1_BiLSTM_CRF.ipynb` — BiLSTM + capa CRF
Añade una **CRF lineal** encima del BiLSTM: en vez de un softmax independiente por
palabra, puntúa la **secuencia completa** y aprende **scores de transición** entre tags
(penaliza saltos imposibles como `O → I-per`); en inferencia decodifica con **Viterbi**.
Incluye **split de validación + early stopping** y weight decay contra el sobreajuste.
Guarda `bilstm_crf_model.pth` + `bilstm_crf_mappings.pkl`.

#### `03_2_BiLSTM_GloVe.ipynb` — BiLSTM + embeddings GloVe
Reemplaza la tabla de embeddings aleatoria por **GloVe pre-entrenado**
(`glove-wiki-gigaword-100`). Ataca el OOV de dos formas: (1) mejores vectores de partida
para generalizar y (2) **extiende el vocabulario con el de GloVe**, de modo que una
palabra de test no vista en train recibe su vector real en lugar de `<UNK>` (en el
smoke-test, **99.5%** del vocabulario quedó inicializado desde GloVe). Lookup con case +
fallback a minúsculas, early stopping. Guarda `bilstm_glove_model.pth` +
`bilstm_glove_mappings.pkl`.

#### `03_3_BiLSTM_GloVe_CRF.ipynb` — combo (mejor modelo no-transformer)
Une las dos mejoras en una sola arquitectura `Embedding(GloVe) → BiLSTM → Linear → CRF`
con early stopping. GloVe ataca el OOV, el CRF da consistencia estructural y el BiLSTM el
contexto. Es el estándar pre-transformer y el mejor de la familia `03.x` (en el smoke-test
CPU ya supera a 03_1 y 03_2). Guarda `bilstm_glove_crf_model.pth` +
`bilstm_glove_crf_mappings.pkl`.

> **Nota:** los números locales de los notebooks `03_1/03_2/03_3` salen bajos porque corren
> en modo CPU-rápido (subset + pocas épocas, aún convergiendo); su rendimiento real se ve
> al re-ejecutarlos en **Colab GPU** (`FULL=True`, datos completos, 30 épocas).

### Paso 4 — `04_Model_DL_Transformer.ipynb` (DL #2)
Fine-tuning de **DistilBERT** (`distilbert-base-cased`) para token classification, con
alineación de etiquetas a sub-tokens (etiqueta en el primer sub-token, `-100` en el
resto). Auto-detecta GPU: full en GPU, *fallback* a subset/1 época en CPU. Guarda
`fitted_models/distilbert_ner/` + `distilbert_labels.pkl`.

### Paso 5 — `05_reproduce_results.ipynb`
Notebook de reproducción exigido por la guía. **No entrena nada**: carga los 5 modelos
desde `fitted_models/` y reporta, para train y test:
- Accuracy considerando solo tokens cuya etiqueta real ≠ `O`.
- Matriz de confusión (heatmap) del mejor modelo.
- F-score ponderado (`sklearn.metrics.f1_score`).
- Predicciones del **TINY TEST** en formato `w1/t1 w2/t2 ...` + su accuracy.
- Tabla comparativa de los 5 modelos.

---

## 2. Resultados actuales

| modelo | train_acc_nonO | test_acc_nonO | train_f1 | test_f1 | tiny_acc |
|---|---|---|---|---|---|
| CRF v1 | 0.864 | 0.477 | 0.876 | 0.477 | 0.76 |
| CRF v2 | 0.931 | 0.705 | 0.934 | 0.727 | 0.88 |
| **CRF v3** | 0.963 | **0.732** | 0.965 | **0.753** | 0.76 |
| BiLSTM | 0.834 | 0.460 | 0.847 | 0.477 | 0.76 |
| DistilBERT* | 0.759 | 0.585 | 0.762 | 0.590 | 0.88 |

\* DistilBERT entrenado en **modo CPU-fallback** (subset de 3.000 frases, 1 época). Aun
así ya iguala al mejor en tiny-test. Es el modelo con más margen de mejora.

---

## 3. Cómo reproducir

```bash
conda env create -f environment.yml
conda activate nlp_d2
# Entrenar (genera los artefactos en fitted_models/):
jupyter nbconvert --to notebook --execute --inplace 02_Model_CRF.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_Model_DL_BiLSTM.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_Model_DL_Transformer.ipynb
# Evaluar (carga todo desde disco):
jupyter nbconvert --to notebook --execute --inplace 05_reproduce_results.ipynb
```

Todos los notebooks detectan Google Colab y ajustan rutas / instalan dependencias
automáticamente.

---

## 4. Puntos de mejora para resultados excelentes

### 4.1 DistilBERT — la mayor palanca (prioridad alta)
Ahora mismo está infra-entrenado (subset + 1 época en CPU). Es casi seguro que con
entrenamiento completo supera al CRF v3 y pasa a ser el mejor modelo.
- **Entrenar full en Colab GPU**: dataset completo, 3–4 épocas, `batch=16`, `lr≈3e-5`,
  `weight_decay=0.01`, *warmup* y *linear scheduler*.
- Probar `bert-base-cased` o `roberta-base` (más capacidad que DistilBERT).
- Métricas a **nivel de entidad** con `seqeval` (precision/recall/F1 por tipo), no solo
  a nivel de token — es el estándar en NER y suele pedirse.

### 4.2 CRF
- Búsqueda de hiperparámetros `c1/c2` con `RandomizedSearchCV` + `flat_f1_score`.
- Features de **gazetteer** (listas de países, organizaciones, nombres) y *clusters* de
  Brown / embeddings discretizados.
- Word shape más rico (patrón colapsado, p. ej. `Xx+` → `Xx`) y features de prefijo/
  sufijo de longitud 4.
- Probar el **structured perceptron** que menciona la guía como comparativa.

### 4.3 BiLSTM
- **Embeddings pre-entrenados** (GloVe / fastText) en lugar de aprenderlos desde cero —
  ataca directamente el problema de OOV visto en el EDA.
- Añadir una capa **CRF encima del BiLSTM** (BiLSTM-CRF) para modelar transiciones de
  etiquetas válidas (evita secuencias imposibles como `O I-per`).
- **Char-level CNN/LSTM** para capturar morfología (mayúsculas, sufijos) de palabras no
  vistas.
- Entrenar más épocas con *early stopping* sobre un split de validación; añadir
  `pack_padded_sequence` y máscara de pérdida correcta sobre el padding.

### 4.4 Datos y evaluación
- **Desbalance de clases**: pérdida ponderada por clase o *focal loss*; las clases raras
  (`nat`, `eve`, `art`) tienen muy pocos ejemplos y arrastran el F1.
- Reportar **F1 macro** además del ponderado (el ponderado oculta el mal rendimiento en
  clases raras).
- Validar consistencia BIO (post-procesado que corrija transiciones inválidas).
- Análisis de errores cualitativo en el tiny-test (casos como `Apple` empresa vs. fruta,
  `London` ciudad vs. apellido) para la sección de conclusiones del informe.

### 4.5 Reproducibilidad / entrega
- Fijar **semillas** (`torch`, `numpy`, `random`) para resultados deterministas.
- Extraer las funciones repetidas (`load_data`, métricas, features CRF) a `utils/utils.py`
  cuando se monte la estructura final del zip.

---

## 5. Pendiente para la entrega final
- `main.pdf`: informe (portada, índice, intro, EDA, metodología, setup, resultados,
  conclusiones, contribuciones, uso de IA).
- Estructura del zip que pide la guía: `data/`, `train_models.ipynb`,
  `reproduce_results.ipynb`, `fitted_models/`, `utils/utils.py`, `environment.yml`.
- (Opcional pero recomendado) entrenar DistilBERT completo en Colab GPU.

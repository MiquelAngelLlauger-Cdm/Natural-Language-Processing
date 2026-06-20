# Guion del informe (`main.pdf`) — NER Deliverable 2

Documento de trabajo para redactar el informe. Sigue la estructura que pide la guía y, en
la parte de Metodología/Resultados, articula los modelos como una **progresión** en la que
cada uno corrige la debilidad del anterior:

`CRF → BiLSTM → BiLSTM+GloVe → BiLSTM+CRF → GloVe+BiLSTM+CRF → DistilBERT`

---

## Estructura del documento (secciones obligatorias)

1. **Portada** — título, asignatura, nombres del equipo.
2. **Índice.**
3. **Introducción** — qué es NER y para qué sirve; las dos familias que comparamos (ML
   clásico con features vs. DL que aprende representaciones); avance de conclusiones.
4. **Análisis exploratorio (EDA)** — del notebook `01_EDA`.
5. **Metodología** — descripción de cada técnica (la progresión de abajo).
6. **Setup experimental** — datos, splits, métricas, configuración de entrenamiento.
7. **Resultados** — tabla comparativa + matrices de confusión + tiny-test.
8. **Conclusiones.**
9. **Contribuciones del equipo.**
10. **Uso de IA.**

---

## EDA — puntos a destacar (sección 4)

- 38.366 frases train / 38.367 test / 13 tiny-test; ~840k tokens por split.
- 17 etiquetas BIO: `O` + B/I de `geo, gpe, org, per, tim, art, eve, nat`.
- **Desbalance severo**: `O` domina (~83% de tokens); clases raras (`nat`, `eve`, `art`)
  con pocos cientos de ejemplos → motiva evaluar **solo sobre tokens ≠ O** y mirar F1.
- **OOV**: porcentaje de palabras de test no vistas en train → argumento central que
  explica el hueco train/test de los modelos sin embeddings pre-entrenados.
- Longitudes de frase (histograma) → justifica `max_len` del padding.

---

## Hilo argumental (Metodología + Resultados)

Idea: cada modelo añade UNA mejora y deja una debilidad clara que motiva el siguiente.

### 1. CRF (clásico) — `02`
**Qué es:** modelo probabilístico de secuencias sobre **features hechas a mano**
(word shape, afijos, mayúsculas, contexto ±1/±2). Tres versiones de complejidad creciente.

- **Fortalezas:**
  - Modela dependencias entre etiquetas (transiciones) → secuencias BIO coherentes.
  - Muy fuerte con pocos datos; entrenamiento rápido, sin GPU.
  - Interpretable: se pueden inspeccionar pesos de features y transiciones.
  - El salto v1→v3 demuestra el **valor de la ingeniería de features** (test F1 0.477 → 0.753).
- **Debilidades:**
  - Depende totalmente de features diseñadas a mano → trabajo experto, poco transferible.
  - Las features son **discretas y locales**; no captura semántica (sinónimos, similitud).
  - Palabras OOV solo se cubren si las features genéricas (shape/afijos) lo permiten.
- **Motiva →** ¿y si el modelo **aprende** las representaciones en vez de diseñarlas a mano?

### 2. BiLSTM desde cero (DL #1) — `03`
**Qué es:** red recurrente bidireccional que aprende embeddings y contexto sin features.

- **Fortalezas:**
  - Cero ingeniería de features; aprende representaciones automáticamente.
  - El BiLSTM ve contexto **izquierdo y derecho** de cada palabra.
- **Debilidades:**
  - **Sobreajuste fuerte** (train F1 ≈ 0.99 vs test ≈ 0.52) con entrenamiento largo.
  - **OOV**: embeddings aprendidos solo del train → toda palabra de test no vista cae en
    `<UNK>` y pierde información (causa principal del hueco train/test).
  - Decide cada etiqueta de forma **independiente** → puede producir secuencias BIO
    inválidas (`O I-per`, `B-geo I-org`).
- **Motiva →** dos problemas separados: (a) OOV/representaciones y (b) consistencia de
  secuencia. Atacamos cada uno.

### 3. BiLSTM + GloVe (DL) — `03_2`  *(ataca el OOV)*
**Qué es:** mismo BiLSTM pero con **embeddings GloVe pre-entrenados** y **vocabulario
extendido con el de GloVe**.

- **Fortalezas:**
  - **Transfer learning**: vectores aprendidos de miles de millones de tokens → palabras
    similares ya empiezan cerca; generaliza con menos ejemplos.
  - **Cobertura de OOV**: palabras de test no vistas en train reciben su vector real en
    lugar de `<UNK>` (en el smoke-test, 99.5% del vocab inicializado desde GloVe).
- **Debilidades:**
  - Sigue decidiendo etiquetas de forma **independiente** (no resuelve las secuencias BIO
    inválidas).
  - GloVe es **no contextual**: un mismo token tiene el mismo vector en cualquier frase
    (`Apple` empresa vs. fruta) — limitación que solo el transformer resuelve del todo.
- **Motiva →** falta dar **estructura** a la salida.

### 4. BiLSTM + CRF (DL) — `03_1`  *(ataca la consistencia de secuencia)*
**Qué es:** BiLSTM con una **capa CRF** encima; el CRF aprende una matriz de transiciones
y decodifica la secuencia óptima con **Viterbi**. Además: validación + early stopping.

- **Fortalezas:**
  - Garantiza **secuencias BIO globalmente consistentes** (penaliza transiciones ilegales).
  - Combina representaciones aprendidas (BiLSTM) con dependencias entre etiquetas (CRF) —
    lo mejor del DL y del CRF clásico.
  - Early stopping + weight decay → menos sobreajuste que el BiLSTM puro.
- **Debilidades:**
  - **No resuelve el OOV**: sigue usando embeddings desde cero → palabras nuevas → `<UNK>`.
  - Más lento y complejo de entrenar (cálculo de la función de partición).
- **Motiva →** combinar las dos mejoras (GloVe + CRF) en un solo modelo.

### 5. GloVe + BiLSTM + CRF (DL, combo) — `03_3`  *(mejor no-transformer)*
**Qué es:** une todo: `Embedding(GloVe) → BiLSTM → Linear → CRF` con early stopping.

- **Fortalezas:**
  - **Cubre las dos debilidades a la vez**: OOV (GloVe) + consistencia de secuencia (CRF).
  - Es el **estándar pre-transformer** para NER; mejor de la familia `03.x`
    (en smoke-test CPU ya supera a 03_1 y 03_2).
- **Debilidades:**
  - Embeddings **no contextuales** (GloVe) → no desambigua según el contexto de la frase.
  - Modelo más pesado; varios componentes que ajustar.
- **Motiva →** el último paso: representaciones **contextuales** profundas.

### 6. DistilBERT fine-tuned (DL Transformer) — `04`  *(estado del arte)*
**Qué es:** transformer pre-entrenado en grandes corpus, **fine-tuned** para token
classification, con alineación de etiquetas a sub-tokens.

- **Fortalezas:**
  - **Embeddings contextuales**: el vector de una palabra depende de toda la frase →
    desambigua (`Apple` empresa vs. fruta).
  - Transfer learning masivo → suele ser el mejor modelo, sobre todo en entidades raras.
  - Tokenización sub-word → maneja palabras desconocidas por sus piezas (mitiga OOV sin
    listas externas).
- **Debilidades:**
  - **Caro**: requiere GPU, más memoria y tiempo; modelo grande (~250 MB).
  - Menos interpretable; sensible a hiperparámetros.
  - En este proyecto, los números reportados son de un entrenamiento parcial (CPU); el
    potencial completo se ve entrenando en Colab GPU.
- **Cierre →** confirma la tesis del informe: pasar de features manuales a representaciones
  aprendidas, y de locales a contextuales, mejora progresivamente el NER.

---

## Tabla de síntesis (fortaleza/debilidad por modelo)

| Modelo | Fortaleza clave | Debilidad clave | Qué mejora respecto al anterior |
|---|---|---|---|
| CRF | features + transiciones, rápido, interpretable | features manuales, sin semántica | — (línea base clásica) |
| BiLSTM | aprende representaciones, contexto bi-direccional | OOV + sobreajuste + tags independientes | quita la ingeniería de features |
| BiLSTM+GloVe | transfer learning, **cubre OOV** | tags independientes, embeddings no contextuales | resuelve OOV |
| BiLSTM+CRF | **secuencias BIO consistentes** | sigue con OOV | resuelve consistencia |
| GloVe+BiLSTM+CRF | OOV **y** consistencia juntas | embeddings no contextuales | combina ambas mejoras |
| DistilBERT | **embeddings contextuales**, SOTA | coste de cómputo, menos interpretable | añade contexto profundo |

---

## Setup experimental (sección 6) — a rellenar con números finales

- **Datos/splits:** train 38.366 / test 38.367; validación = 10% del train (para early
  stopping en los DL). Tiny-test = 13 frases.
- **Métricas:** accuracy sobre tokens ≠ O; F1 ponderado (excluyendo O); matrices de
  confusión; accuracy del tiny-test. (Recomendado añadir F1 macro y F1 a nivel de entidad
  con `seqeval`.)
- **CRF:** `lbfgs`, `all_possible_transitions=True`, c1/c2 por versión.
- **BiLSTM/variantes:** emb 100/200/300, hidden 128, dropout 0.5, Adam lr 1e-3,
  weight_decay 1e-4, early stopping (paciencia 3), hasta 30 épocas en GPU.
- **DistilBERT:** `distilbert-base-cased`, lr 3e-5, batch 16, 3 épocas, max_len 64.

> **Importante para el informe:** rellenar la tabla de Resultados con los números
> definitivos obtenidos en **Colab GPU** (los locales son de modo CPU-rápido y subestiman
> a los modelos DL).

---

## Conclusiones (esqueleto)

- La ingeniería de features (CRF v1→v3) ya da un modelo clásico muy competitivo.
- En DL, las dos palancas grandes son **representaciones pre-entrenadas (OOV)** y
  **modelado de la estructura de etiquetas (CRF)**; juntas (03_3) dan el mejor
  no-transformer.
- Las representaciones **contextuales** (DistilBERT) son el siguiente salto, a coste de
  cómputo.
- Limitaciones: desbalance de clases (clases raras), tamaño de los datos, y métricas a
  nivel de token vs. entidad.

# Motor de Búsqueda de Ofertas Laborales — EPN

Sistema de Recuperación de Información que compara modelos clásicos y semánticos sobre un corpus de **47 322 ofertas laborales** orientadas a egresados de la Escuela Politécnica Nacional.

**Proyecto 1er Bimestre · Recuperación de Información · Prof. Iván Carrera**
**Integrantes:**

- Mateo Sebastián Cumbal Guasgua

- Daniel Ismael Flores Espín

- Alicia Dayana Pereira Tuqueres

---

## Modelos implementados

| # | Modelo | Tipo |
|---|--------|------|
| 1 | Jaccard | Binario / conjuntos |
| 2 | Coseno TF-IDF | Vectorial |
| 3 | BM25 | Probabilístico |
| 4 | Embeddings semánticos | Dense retrieval (ChromaDB) |

---

## Estructura del proyecto

```
motor_busqueda_RI/
├── main.py                  # CLI interactiva (punto de entrada)
├── requirements.txt         # Dependencias Python
├── src/
│   ├── preprocessing.py     # Limpieza y construcción del corpus
│   ├── indexer.py           # Construcción del índice invertido
│   ├── models.py            # Jaccard, Coseno TF-IDF, BM25
│   ├── embeddings.py        # Recuperación semántica con ChromaDB
│   └── evaluation.py        # Métricas: Precision@K, Recall@K, MAP
├── notebooks/
│   ├── informe_tecnico.ipynb   # Informe técnico del proyecto
│   ├── prueba_indexador.ipynb
│   ├── test_clasic_models.ipynb
│   ├── reporte_experimentos.ipynb
│   └── stats_new_index.ipynb
└── data/
    ├── raw/                 # CSVs originales por carrera (no incluidos en git)
    └── processed/
        ├── corpus_limpio.csv       # Corpus unificado y limpio
        ├── indice_invertido.pkl    # Índice invertido serializado
        └── chroma_db/             # Base de datos vectorial (no incluida en git)
```

---

## Requisitos

- Python 3.12
- Las dependencias están en `requirements.txt`

---

## Instalación

```bash
# 1. Crear y activar el entorno virtual
python -m venv env
.\env\Scripts\activate        # Windows
# source env/bin/activate     # Linux / macOS

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar recursos de NLTK (solo la primera vez)
python -c "import nltk; nltk.download('stopwords')"
```

---

## Pasos de configuración (ejecutar una sola vez)

Los artefactos de datos (`data/raw/`, `chroma_db/`) no están incluidos en el repositorio. Sigue estos pasos en orden para generarlos.

### Paso 1 — Construir el corpus limpio

Coloca los CSVs de ofertas laborales en `data/raw/` (una subcarpeta por carrera, cada una con su archivo `*_Merged.csv`). Luego ejecuta:

```bash
python src/preprocessing.py
```

Genera `data/processed/corpus_limpio.csv` (~47 k documentos únicos).

### Paso 2 — Construir el índice invertido

```bash
python src/indexer.py
```

Genera `data/processed/indice_invertido.pkl` (~24 MB).

### Paso 3 — Construir la base de datos vectorial

```bash
python src/embeddings.py
```

Codifica los ~47 k documentos con `paraphrase-multilingual-MiniLM-L12-v2` y los almacena en ChromaDB. **Este paso tarda varios minutos** dependiendo del hardware.

---

## Uso

### CLI interactiva

```bash
python main.py
```

Muestra un menú para seleccionar el modelo y escribir consultas de texto libre. Devuelve los 10 resultados más relevantes con score y título del puesto.

```
============================================================
   SISTEMA DE RECUPERACIÓN DE OFERTAS LABORALES — EPN
============================================================
  1.  Modelo Binario     (Jaccard)
  2.  Modelo Vectorial   (Coseno TF-IDF)
  3.  Modelo BM25
  4.  Recuperación Semántica (Embeddings)
  5.  Salir
```

### Evaluación de métricas

```bash
cd src
python evaluation.py
```

Ejecuta 24 consultas de prueba (una por carrera de la EPN) sobre los 4 modelos y reporta **Precision@10**, **Recall@10** y **MAP**.

---

## Descripción técnica de los modelos

### Jaccard
Trata la consulta y el documento como conjuntos de términos (vectores binarios):

```
J(Q, D) = |Q ∩ D| / |Q ∪ D|
```

### Coseno TF-IDF
Pondera cada término por su frecuencia en el documento (TF) y su rareza en el corpus (IDF). Calcula la similitud como producto punto normalizado:

```
TF(t,d)  = freq(t,d) / |d|
IDF(t)   = log(N / df(t))
score    = (q · d) / (‖q‖ · ‖d‖)
```

Las normas de los documentos se precomputan al inicializar el modelo.

### BM25
Modelo probabilístico con saturación del TF y normalización por longitud. Parámetros estándar: `k1 = 1.2`, `b = 0.75`:

```
score(q,d) = Σ IDF(t) · [freq(t,d)·(k1+1)] / [freq(t,d) + k1·(1 - b + b·|d|/avgdl)]
IDF(t)     = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
```

### Embeddings semánticos
Codifica documentos y consultas con `paraphrase-multilingual-MiniLM-L12-v2` (soporte nativo para español). Los vectores se almacenan en ChromaDB y la búsqueda se realiza por similitud en el espacio vectorial:

```
score = 1 / (1 + distancia_L2)
```

---

## Pipeline de preprocesamiento

Todo el texto (documentos y consultas) pasa por el mismo pipeline antes de ser indexado o buscado:

1. **Normalización** (`preprocessing.py`): minúsculas, eliminación de tildes (NFD Unicode), remoción de caracteres especiales.
2. **Tokenización NLP** (`indexer.py`): stemming con `SnowballStemmer("spanish")`, filtrado de stopwords en español, descarte de tokens de longitud ≤ 1.

El campo indexado concatena `job_title + description_final + careers_required` de cada oferta.

---

## Evaluación

Los *qrels* se construyen automáticamente combinando dos fuentes:
- **Etiqueta exacta**: documentos cuya columna `all_careers` contiene la carrera de la consulta.
- **Título del puesto**: documentos cuyo `job_title` contiene términos asociados a la carrera (en español e inglés).

Esto reduce el sesgo de las métricas hacia modelos léxicos y da una evaluación más justa al modelo semántico.
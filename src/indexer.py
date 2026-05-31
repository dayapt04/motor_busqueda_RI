# src/indexer.py
# Construcción del índice invertido a partir del corpus limpio.
# Uso: python src/indexer.py

import os
import pickle
from collections import defaultdict
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download("stopwords", quiet=True)

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_CORPUS = BASE_DIR / "data" / "processed" / "corpus_limpio.csv"
RUTA_INDICE = BASE_DIR / "data" / "processed" / "indice_invertido.pkl"


# ──────────────────────────────────────────────────────
#  Preprocesamiento NLP
# ──────────────────────────────────────────────────────


class ProcesadorNLP:
    """Tokeniza texto aplicando stopwords en español y stemming SnowballStemmer."""

    def __init__(self):
        self.stop_words = set(stopwords.words("spanish"))
        self.stemmer = SnowballStemmer("spanish")

    def tokenizar(self, texto: str) -> list:
        if not isinstance(texto, str):
            return []
        tokens = []
        for palabra in texto.split():
            if len(palabra) > 1 and palabra not in self.stop_words:
                tokens.append(self.stemmer.stem(palabra))
        return tokens


# ──────────────────────────────────────────────────────
#  Índice invertido
# ──────────────────────────────────────────────────────


class IndiceInvertido:
    """
    Construye y persiste el índice invertido.
    Estructura: indice[término][doc_id] = frecuencia
    """

    def __init__(self):
        self.nlp = ProcesadorNLP()
        self.indice = defaultdict(lambda: defaultdict(int))
        self.longitud_docs = {}
        self.total_documentos = 0

    def construir_desde_dataframe(self, df: pd.DataFrame) -> None:
        self.total_documentos = len(df)

        for _, fila in df.iterrows():
            doc_id = fila["job_id"]
            texto = (
                str(fila["job_title"])
                + " "
                + str(fila["description_final"])
                + " "
                + str(fila["careers_required"])
            )
            tokens = self.nlp.tokenizar(texto)

            self.longitud_docs[doc_id] = len(tokens)
            for token in tokens:
                self.indice[token][doc_id] += 1

        print(f"Vocabulario total: {len(self.indice):,} términos únicos.")

    def guardar(self, ruta: Path) -> None:
        os.makedirs(ruta.parent, exist_ok=True)
        datos = {
            "indice": dict(self.indice),
            "longitud_docs": self.longitud_docs,
            "total_documentos": self.total_documentos,
        }
        with open(ruta, "wb") as f:
            pickle.dump(datos, f)
        print(f"Índice guardado en {ruta}")

    def cargar(self, ruta: Path) -> None:
        with open(ruta, "rb") as f:
            datos = pickle.load(f)
        self.indice = datos["indice"]
        self.longitud_docs = datos["longitud_docs"]
        self.total_documentos = datos["total_documentos"]
        print(
            f"Índice cargado: {len(self.indice):,} términos | {self.total_documentos:,} documentos."
        )


# ──────────────────────────────────────────────────────
#  Script de construcción
# ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Cargando corpus...")
    df = pd.read_csv(RUTA_CORPUS)
    print(f"{len(df):,} documentos cargados.")

    indexador = IndiceInvertido()
    indexador.construir_desde_dataframe(df)
    indexador.guardar(RUTA_INDICE)

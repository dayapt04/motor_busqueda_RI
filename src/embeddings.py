# src/embeddings.py
# Recuperación semántica con embeddings y ChromaDB.
#
# Como módulo: expone BuscadorSemantico para ser importado por main.py
# Como script:  construye la base de datos vectorial (ejecutar una sola vez)
#
#   python src/embeddings.py

import os
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_CORPUS = BASE_DIR / "data" / "processed" / "corpus_limpio.csv"
RUTA_DB = BASE_DIR / "data" / "processed" / "chroma_db"
NOMBRE_COLECCION = "empleos"
MODELO_NOMBRE = "paraphrase-multilingual-MiniLM-L12-v2"


# ──────────────────────────────────────────────────────
#  Buscador semántico (importable desde main.py)
# ──────────────────────────────────────────────────────


class BuscadorSemantico:
    """
    Recuperación semántica usando embeddings + ChromaDB.
    Requiere que la BD vectorial ya esté generada (python src/embeddings.py).

        score = 1 / (1 + distancia_L2)   → mayor score = más similar
    """

    def __init__(self, ruta_db=None):
        print("Cargando motor semántico...")
        ruta = Path(ruta_db) if ruta_db else RUTA_DB
        self.modelo = SentenceTransformer(MODELO_NOMBRE)
        cliente = chromadb.PersistentClient(path=str(ruta))
        self.coleccion = cliente.get_collection(name=NOMBRE_COLECCION)

    def buscar(self, consulta: str, top_k: int = 10) -> list:
        embedding = self.modelo.encode([consulta]).tolist()
        respuesta = self.coleccion.query(query_embeddings=embedding, n_results=top_k)
        resultados = [
            (doc_id, 1 / (1 + dist))
            for doc_id, dist in zip(respuesta["ids"][0], respuesta["distances"][0])
        ]
        resultados.sort(key=lambda x: x[1], reverse=True)
        return resultados


# ──────────────────────────────────────────────────────
#  Script de construcción de la BD vectorial
# ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Cargando modelo de embeddings...")
    modelo = SentenceTransformer(MODELO_NOMBRE)

    print("Cargando corpus...")
    df = pd.read_csv(RUTA_CORPUS).fillna("")

    textos = (
        df["job_title"]
        + " - "
        + df["careers_required"]
        + ". "
        + df["description_final"]
    ).tolist()
    ids = df["job_id"].astype(str).tolist()

    print(f"Generando embeddings para {len(textos):,} documentos...")
    embeddings = modelo.encode(textos, show_progress_bar=True, batch_size=32).tolist()

    print("Inicializando ChromaDB...")
    os.makedirs(RUTA_DB, exist_ok=True)
    cliente = chromadb.PersistentClient(path=str(RUTA_DB))

    try:
        cliente.delete_collection(name=NOMBRE_COLECCION)
    except Exception:
        pass

    coleccion = cliente.create_collection(name=NOMBRE_COLECCION)

    print("Insertando vectores en ChromaDB...")
    batch = 5000
    for i in range(0, len(ids), batch):
        coleccion.add(
            embeddings=embeddings[i : i + batch],
            documents=textos[i : i + batch],
            ids=ids[i : i + batch],
        )
        print(f"  {min(i+batch, len(ids)):,} / {len(ids):,}")

    print("Base de datos vectorial lista.")
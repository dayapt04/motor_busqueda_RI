# src/models.py
# Modelos clásicos de recuperación: Jaccard, Coseno TF-IDF, BM25.
# Los tres comparten una clase base que carga el índice invertido y
# preprocesa consultas usando el mismo pipeline que indexer.py.

import pickle
import math
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from indexer import ProcesadorNLP
from preprocessing import limpiar_texto

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_INDICE_DEFAULT = BASE_DIR / "data" / "processed" / "indice_invertido.pkl"


def cargar_indice(ruta=None) -> dict:
    """
    Carga el índice invertido desde disco una sola vez.
    Pasar el resultado a los buscadores via datos_indice= para
    evitar leer el pickle tres veces.
    """
    ruta = Path(ruta) if ruta else RUTA_INDICE_DEFAULT
    print(f"Cargando índice desde {ruta}...")
    with open(ruta, "rb") as f:
        datos = pickle.load(f)
    print(
        f"Índice listo: {len(datos['indice']):,} términos | {datos['total_documentos']:,} documentos."
    )
    return datos


# ──────────────────────────────────────────────────────
#  Clase base
# ──────────────────────────────────────────────────────


class Buscador:
    """
    Carga el índice invertido y expone el preprocesamiento de consultas.
    Uso recomendado para cargar el índice una sola vez:

        datos = cargar_indice()
        jac   = BuscadorJaccard(datos_indice=datos)
        cos   = BuscadorCoseno(datos_indice=datos)
        bm25  = BuscadorBM25(datos_indice=datos)
    """

    def __init__(self, ruta_indice=None, datos_indice=None):
        self.nlp = ProcesadorNLP()
        if datos_indice is not None:
            self.indice = datos_indice["indice"]
            self.longitud_docs = datos_indice["longitud_docs"]
            self.total_documentos = datos_indice["total_documentos"]
        else:
            datos = cargar_indice(ruta_indice)
            self.indice = datos["indice"]
            self.longitud_docs = datos["longitud_docs"]
            self.total_documentos = datos["total_documentos"]

    def _preprocesar_consulta(self, consulta: str) -> list:
        """Mismo pipeline que indexer.py: limpiar_texto + tokenizar con stemming y stopwords."""
        return self.nlp.tokenizar(limpiar_texto(consulta))

    def _get_candidatos(self, tokens: list) -> set:
        """Retorna doc_ids que contienen al menos un token de la consulta."""
        candidatos = set()
        for token in tokens:
            if token in self.indice:
                candidatos.update(self.indice[token].keys())
        return candidatos

    def buscar(self, consulta: str, top_k: int = 10) -> list:
        raise NotImplementedError


# ──────────────────────────────────────────────────────
#  Modelo 1 — Jaccard (vectores binarios)
# ──────────────────────────────────────────────────────


class BuscadorJaccard(Buscador):
    """
    Similitud de Jaccard entre la consulta Q y el documento D como conjuntos de términos.

        J(Q, D) = |Q ∩ D| / |Q ∪ D|
    """

    def __init__(self, ruta_indice=None, datos_indice=None):
        super().__init__(ruta_indice, datos_indice)
        # Cantidad de términos únicos por documento — necesario para calcular |D| en la unión
        self._terminos_por_doc = self._construir_terminos_por_doc()

    def _construir_terminos_por_doc(self) -> dict:
        conteo = defaultdict(int)
        for postings in self.indice.values():
            for doc_id in postings:
                conteo[doc_id] += 1
        return dict(conteo)

    def buscar(self, consulta: str, top_k: int = 10) -> list:
        tokens = self._preprocesar_consulta(consulta)
        if not tokens:
            return []

        set_consulta = set(tokens)
        candidatos = self._get_candidatos(list(set_consulta))
        if not candidatos:
            return []

        resultados = []
        for doc_id in candidatos:
            interseccion = sum(
                1 for t in set_consulta if t in self.indice and doc_id in self.indice[t]
            )
            union = (
                len(set_consulta) + self._terminos_por_doc.get(doc_id, 0) - interseccion
            )
            resultados.append((doc_id, interseccion / union if union > 0 else 0.0))

        resultados.sort(key=lambda x: x[1], reverse=True)
        return resultados[:top_k]


# ──────────────────────────────────────────────────────
#  Modelo 2 — Coseno con TF-IDF
# ──────────────────────────────────────────────────────


class BuscadorCoseno(Buscador):
    """
    Similitud de Coseno con pesos TF-IDF.

        TF(t, d)     = freq(t, d) / len(d)
        IDF(t)       = log(N / df(t))
        TF-IDF(t, d) = TF(t, d) * IDF(t)
        score(q, d)  = dot(q, d) / (||q|| * ||d||)

    Las normas de los documentos se precomputan una sola vez al inicializar.
    """

    def __init__(self, ruta_indice=None, datos_indice=None):
        super().__init__(ruta_indice, datos_indice)
        self._cache_idf = {}
        self._normas_docs = self._precomputar_normas()

    def _idf(self, token: str) -> float:
        if token not in self._cache_idf:
            df = len(self.indice.get(token, {}))
            self._cache_idf[token] = (
                math.log(self.total_documentos / df) if df > 0 else 0.0
            )
        return self._cache_idf[token]

    def _precomputar_normas(self) -> dict:
        """Recorre las ~3.1M entradas del índice una vez para calcular ||d|| de cada documento."""
        print("Precomputando normas TF-IDF de los documentos...")
        normas_sq = defaultdict(float)
        for token, postings in self.indice.items():
            df = len(postings)
            if df == 0:
                continue
            idf = math.log(self.total_documentos / df)
            for doc_id, freq in postings.items():
                tf = freq / self.longitud_docs.get(doc_id, 1)
                normas_sq[doc_id] += (tf * idf) ** 2
        print("Normas listas.")
        return {doc_id: math.sqrt(val) for doc_id, val in normas_sq.items()}

    def buscar(self, consulta: str, top_k: int = 10) -> list:
        tokens = self._preprocesar_consulta(consulta)
        if not tokens:
            return []

        candidatos = self._get_candidatos(tokens)
        if not candidatos:
            return []

        # Vector TF-IDF de la consulta
        freq_q = defaultdict(int)
        for t in tokens:
            freq_q[t] += 1
        vector_q = {t: (f / len(tokens)) * self._idf(t) for t, f in freq_q.items()}
        norma_q = math.sqrt(sum(v**2 for v in vector_q.values()))
        if norma_q == 0:
            return []

        # Producto punto consulta · documento
        scores = defaultdict(float)
        for token, peso_q in vector_q.items():
            if token not in self.indice:
                continue
            idf = self._idf(token)
            for doc_id, freq in self.indice[token].items():
                tf = freq / self.longitud_docs.get(doc_id, 1)
                scores[doc_id] += peso_q * (tf * idf)

        resultados = []
        for doc_id, dot in scores.items():
            norma_d = self._normas_docs.get(doc_id, 0.0)
            denom = norma_q * norma_d
            resultados.append((doc_id, dot / denom if denom > 0 else 0.0))

        resultados.sort(key=lambda x: x[1], reverse=True)
        return resultados[:top_k]


# ──────────────────────────────────────────────────────
#  Modelo 3 — BM25
# ──────────────────────────────────────────────────────


class BuscadorBM25(Buscador):
    """
    Modelo BM25 (Best Match 25).

        IDF(t)      = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        score(q, d) = Σ IDF(t) * [freq(t,d) * (k1+1)] /
                          [freq(t,d) + k1 * (1 - b + b * |d|/avgdl)]

        k1 = 1.2  — saturación del TF
        b  = 0.75 — normalización por longitud del documento
    """

    def __init__(
        self, ruta_indice=None, datos_indice=None, k1: float = 1.2, b: float = 0.75
    ):
        super().__init__(ruta_indice, datos_indice)
        self.k1 = k1
        self.b = b
        self.avgdl = sum(self.longitud_docs.values()) / len(self.longitud_docs)
        self._idf = self._precomputar_idf()

    def _precomputar_idf(self) -> dict:
        N = self.total_documentos
        return {
            termino: math.log((N - len(postings) + 0.5) / (len(postings) + 0.5) + 1)
            for termino, postings in self.indice.items()
        }

    def buscar(self, consulta: str, top_k: int = 10) -> list:
        tokens = self._preprocesar_consulta(consulta)
        if not tokens:
            return []

        candidatos = self._get_candidatos(tokens)
        if not candidatos:
            return []

        scores = {}
        for token in set(tokens):
            if token not in self.indice:
                continue
            idf = self._idf[token]
            for doc_id, freq in self.indice[token].items():
                dl = self.longitud_docs.get(doc_id, 1)
                denom = freq + self.k1 * (1 - self.b + self.b * (dl / self.avgdl))
                scores[doc_id] = (
                    scores.get(doc_id, 0.0) + idf * (freq * (self.k1 + 1)) / denom
                )

        resultados = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return resultados[:top_k]


# ──────────────────────────────────────────────────────
#  Helper de presentación (notebooks / pruebas rápidas)
# ──────────────────────────────────────────────────────


def mostrar_resultados(resultados: list, modelo: str, corpus_df=None) -> None:
    print(f"\n{'─'*60}")
    print(f"  Modelo: {modelo}  |  {len(resultados)} resultado(s)")
    print(f"{'─'*60}")
    if not resultados:
        print("  Sin resultados.")
        return
    for i, (doc_id, score) in enumerate(resultados, 1):
        titulo = doc_id[:30] + "..."
        if corpus_df is not None:
            fila = corpus_df[corpus_df["job_id"] == doc_id]
            if not fila.empty:
                titulo = fila.iloc[0]["job_title"]
        print(f"  {i:>2}. score={score:.4f}  |  {titulo}")
    print()

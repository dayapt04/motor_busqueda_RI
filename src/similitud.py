# src/similitud.py

import pickle
import math
import sys
from pathlib import Path
from collections import defaultdict

# Asegura que Python encuentre los módulos del mismo paquete
sys.path.insert(0, str(Path(__file__).resolve().parent))
from indexar import ProcesadorNLP
from preprocessing import limpiar_texto

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_INDICE_DEFAULT = BASE_DIR / "data" / "processed" / "indice_invertido.pkl"


# ─────────────────────────────────────────────
#  Clase base compartida por los dos modelos
# ─────────────────────────────────────────────


class Buscador:
    """
    Carga el índice invertido y expone el preprocesamiento
    de consultas usando el mismo pipeline que Persona 1 y 2.
    """

    def __init__(self, ruta_indice=None):
        self.nlp = ProcesadorNLP()
        ruta = Path(ruta_indice) if ruta_indice else RUTA_INDICE_DEFAULT
        self._cargar_indice(ruta)

    def _cargar_indice(self, ruta: Path):
        print(f"Cargando índice desde {ruta}...")
        with open(ruta, "rb") as f:
            datos = pickle.load(f)
        self.indice = datos["indice"]  # dict[term -> dict[doc_id -> freq]]
        self.longitud_docs = datos["longitud_docs"]  # dict[doc_id -> total_tokens]
        self.total_documentos = datos["total_documentos"]  # int
        print(
            f"Índice listo: {len(self.indice):,} términos | {self.total_documentos:,} documentos."
        )

    def _preprocesar_consulta(self, consulta: str) -> list:
        """
        Aplica limpiar_texto (Persona 1) + tokenizar con stopwords y stemming (Persona 2).
        Garantiza que los tokens de la consulta sean comparables con los del índice.
        """
        texto_limpio = limpiar_texto(consulta)
        return self.nlp.tokenizar(texto_limpio)

    def _get_candidatos(self, tokens: list) -> set:
        """
        Devuelve el conjunto de doc_ids que contienen al menos
        un término de la consulta. Evita recorrer los 74 k documentos.
        """
        candidatos = set()
        for token in tokens:
            if token in self.indice:
                candidatos.update(self.indice[token].keys())
        return candidatos

    def buscar(self, consulta: str, top_k: int = 10) -> list:
        raise NotImplementedError("Implementar en la subclase.")


# ─────────────────────────────────────────────
#  Modelo 1 — Jaccard (vectores binarios)
# ─────────────────────────────────────────────


class BuscadorJaccard(Buscador):
    """
    Similitud de Jaccard entre la consulta Q y cada documento D.

        J(Q, D) = |Q ∩ D| / |Q ∪ D|

    Q y D se tratan como conjuntos de términos únicos (binario:
    no importa cuántas veces aparece cada término).
    """

    def __init__(self, ruta_indice=None):
        super().__init__(ruta_indice)
        # Precomputa cuántos términos únicos tiene cada documento.
        # Necesario para calcular |D| en la fórmula de la unión.
        self._doc_term_counts = self._construir_doc_term_counts()

    def _construir_doc_term_counts(self) -> dict:
        """
        Recorre el índice una sola vez y cuenta cuántos términos
        distintos aparecen en cada documento.
        """
        conteo = defaultdict(int)
        for postings in self.indice.values():
            for doc_id in postings:
                conteo[doc_id] += 1
        return dict(conteo)

    def buscar(self, consulta: str, top_k: int = 10) -> list:
        """
        Retorna una lista de (doc_id, score_jaccard) ordenada de mayor a menor,
        limitada a top_k resultados.
        """
        tokens = self._preprocesar_consulta(consulta)
        if not tokens:
            print(
                "[Jaccard] Consulta sin términos válidos después del preprocesamiento."
            )
            return []

        set_consulta = set(tokens)  # binario: sin duplicados
        candidatos = self._get_candidatos(list(set_consulta))
        if not candidatos:
            print("[Jaccard] Ningún documento contiene los términos de la consulta.")
            return []

        resultados = []
        for doc_id in candidatos:
            # |Q ∩ D|: términos de la consulta que aparecen en el documento
            interseccion = sum(
                1 for t in set_consulta if t in self.indice and doc_id in self.indice[t]
            )
            # |Q ∪ D| = |Q| + |D_unique| - |Q ∩ D|
            terminos_doc = self._doc_term_counts.get(doc_id, 0)
            union = len(set_consulta) + terminos_doc - interseccion

            score = interseccion / union if union > 0 else 0.0
            resultados.append((doc_id, score))

        resultados.sort(key=lambda x: x[1], reverse=True)
        return resultados[:top_k]


# ─────────────────────────────────────────────
#  Modelo 2 — Coseno con TF-IDF
# ─────────────────────────────────────────────


class BuscadorCoseno(Buscador):
    """
    Similitud de Coseno usando pesos TF-IDF.

        TF(t, d)     = freq(t, d) / total_tokens(d)
        IDF(t)       = log( N / df(t) )
        TF-IDF(t, d) = TF(t, d) × IDF(t)

        score(q, d)  = (q⃗ · d⃗) / ( ||q⃗|| × ||d⃗|| )

    Las normas de los documentos se precomputan al cargar el índice
    para no recalcularlas en cada consulta.
    """

    def __init__(self, ruta_indice=None):
        super().__init__(ruta_indice)
        self._idf_cache = {}
        self._doc_norms = self._precompute_doc_norms()

    def _idf(self, token: str) -> float:
        """IDF con caché para no recalcular el mismo token dos veces."""
        if token not in self._idf_cache:
            df = len(self.indice.get(token, {}))
            self._idf_cache[token] = (
                math.log(self.total_documentos / df) if df > 0 else 0.0
            )
        return self._idf_cache[token]

    def _precompute_doc_norms(self) -> dict:
        """
        Calcula ||d⃗||  para cada documento una sola vez al arrancar.
        Recorre las ~3.1 M entradas del índice: tarda unos segundos
        pero elimina ese costo en cada consulta posterior.
        """
        print("Precomputando normas TF-IDF de los documentos (operación única)...")
        normas_sq = defaultdict(float)  # acumula la suma de cuadrados

        for token, postings in self.indice.items():
            df = len(postings)
            if df == 0:
                continue
            idf = math.log(self.total_documentos / df)
            for doc_id, freq in postings.items():
                tf = freq / self.longitud_docs.get(doc_id, 1)
                normas_sq[doc_id] += (tf * idf) ** 2

        normas = {doc_id: math.sqrt(val) for doc_id, val in normas_sq.items()}
        print("Normas precomputadas.")
        return normas

    def buscar(self, consulta: str, top_k: int = 10) -> list:
        """
        Retorna una lista de (doc_id, score_coseno) ordenada de mayor a menor,
        limitada a top_k resultados.
        """
        tokens = self._preprocesar_consulta(consulta)
        if not tokens:
            print(
                "[Coseno] Consulta sin términos válidos después del preprocesamiento."
            )
            return []

        candidatos = self._get_candidatos(tokens)
        if not candidatos:
            print("[Coseno] Ningún documento contiene los términos de la consulta.")
            return []

        # ── Vector TF-IDF de la consulta ──────────────────────────────
        freq_q = defaultdict(int)
        for t in tokens:
            freq_q[t] += 1

        vector_q = {
            t: (freq / len(tokens)) * self._idf(t) for t, freq in freq_q.items()
        }
        norma_q = math.sqrt(sum(v**2 for v in vector_q.values()))
        if norma_q == 0:
            return []

        # ── Producto punto q⃗ · d⃗ para cada documento candidato ────────
        scores = defaultdict(float)
        for token, peso_q in vector_q.items():
            if token not in self.indice:
                continue
            idf = self._idf(token)
            for doc_id, freq in self.indice[token].items():
                tf = freq / self.longitud_docs.get(doc_id, 1)
                scores[doc_id] += peso_q * (tf * idf)

        # ── Similitud coseno = dot / (||q|| × ||d||) ──────────────────
        resultados = []
        for doc_id, dot in scores.items():
            norma_d = self._doc_norms.get(doc_id, 0.0)
            denom = norma_q * norma_d
            score = dot / denom if denom > 0 else 0.0
            resultados.append((doc_id, score))

        resultados.sort(key=lambda x: x[1], reverse=True)
        return resultados[:top_k]


# ─────────────────────────────────────────────
#  Helper de presentación
# ─────────────────────────────────────────────


def mostrar_resultados(resultados: list, modelo: str, corpus_df=None) -> None:
    """Imprime el ranking en consola de forma legible."""
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


# ─────────────────────────────────────────────
#  Ejecución directa para pruebas rápidas
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd

    df_corpus = pd.read_csv(BASE_DIR / "data" / "processed" / "corpus_limpio.csv")
    consulta = input("Consulta: ").strip()

    print("\n=== JACCARD ===")
    jac = BuscadorJaccard()
    res_jac = jac.buscar(consulta, top_k=10)
    mostrar_resultados(res_jac, "Jaccard", df_corpus)

    print("\n=== COSENO TF-IDF ===")
    cos = BuscadorCoseno()
    res_cos = cos.buscar(consulta, top_k=10)
    mostrar_resultados(res_cos, "Coseno TF-IDF", df_corpus)

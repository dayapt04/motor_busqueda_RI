#src/indexear.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from collections import defaultdict
import pickle
import os
from pathlib import Path

# recursos de NLTK necesarios para el procesamiento de texto
nltk.download('stopwords', quiet=True)


class ProcesadorNLP:
    """
    Clase encargada de limpiar el texto, quitar stopwords y aplicar stemming.
    """

    def __init__(self):
        self.stop_words = set(stopwords.words('spanish'))
        self.stemmer = SnowballStemmer('spanish')

    def tokenizar(self, texto):
        # validar que el texto no sea nulo
        if type(texto) != str:
            return []

        palabras = texto.split()

        tokens_finales = []

        for palabra in palabras:
            # Filtrar palabras muy cortas y stopwords
            if len(palabra) > 1 and palabra not in self.stop_words:
                # stemmer (ej. 'programadores' -> 'program')
                raiz = self.stemmer.stem(palabra)
                tokens_finales.append(raiz)

        return tokens_finales


class IndiceInvertido:
    """
    Construir índice invertido a partir del DataFrame con columnas 'job_id', 'job_title', 'description_final' y 'careers_required'.
    """

    def __init__(self):
        self.nlp = ProcesadorNLP()
        # Diccionario de diccionarios: indice['python']['doc_1'] = frecuencia
        self.indice = defaultdict(lambda: defaultdict(int))
        # longitud de cada documento
        self.longitud_docs = {}
        self.total_documentos = 0

    def construir_desde_dataframe(self, df):
        self.total_documentos = len(df)

        # iterar fila por fila del DataFrame
        for index, fila in df.iterrows():
            doc_id = fila['job_id']

            #  texto completo del documento
            texto_completo = str(fila['job_title']) + " " + \
                             str(fila['description_final']) + " " + \
                             str(fila['careers_required'])

            tokens = self.nlp.tokenizar(texto_completo)

            # cuántas palabras tiene el documento
            self.longitud_docs[doc_id] = len(tokens)

            for token in tokens:
                self.indice[token][doc_id] += 1

        print(f"Vocabulario total: {len(self.indice)} palabras únicas.")

    def guardar(self, ruta_archivo):
        """
        Guarda el índice en el disco duro usando pickle.
        """
        print(f"Guardando índice en {ruta_archivo}...")
        os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)

        # datos a guardar: el índice, la longitud de cada documento y el total de documentos
        datos = {
            'indice': dict(self.indice),
            'longitud_docs': self.longitud_docs,
            'total_documentos': self.total_documentos
        }

        with open(ruta_archivo, 'wb') as archivo:
            pickle.dump(datos, archivo)
        print("Índice guardado exitosamente.")

    def cargar(self, ruta_archivo):
        """
        Carga un índice previamente guardado en el disco duro.
        """
        print(f"Cargando índice desde {ruta_archivo}...")
        with open(ruta_archivo, 'rb') as archivo:
            datos = pickle.load(archivo)

        self.indice = datos['indice']
        self.longitud_docs = datos['longitud_docs']
        self.total_documentos = datos['total_documentos']
        print("Índice cargado y listo para buscar.")


if __name__ == '__main__':
    BASE_DIR = Path(__file__).resolve().parent.parent
    RUTA_CORPUS = BASE_DIR / 'data' / 'processed' / 'corpus_limpio.csv'
    RUTA_INDICE = BASE_DIR / 'data' / 'processed' / 'indice_invertido.pkl'

    print("Cargando corpus...")
    df_corpus = pd.read_csv(RUTA_CORPUS)

    indexador = IndiceInvertido()
    indexador.construir_desde_dataframe(df_corpus)

    indexador.guardar(RUTA_INDICE)

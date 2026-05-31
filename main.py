import sys
from pathlib import Path
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Asegurar que se puedan importar los módulos de la carpeta src
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

# Importamos las clases de tu compañero Mateo
from similitud import BuscadorJaccard, BuscadorCoseno, mostrar_resultados

class BuscadorSemantico:
    def __init__(self, ruta_db=None):
        print("Cargando motor semántico y modelo IA...")
        if ruta_db is None:
            ruta_db = BASE_DIR / 'data' / 'processed' / 'chroma_db'
            
        self.cliente = chromadb.PersistentClient(path=str(ruta_db))
        self.coleccion = self.cliente.get_collection(name="empleos")
        self.modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
    def buscar(self, consulta: str, top_k: int = 10) -> list:
        # Generar el vector de la consulta
        query_embedding = self.modelo.encode([consulta]).tolist()
        
        # Buscar en ChromaDB
        resultados_chroma = self.coleccion.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        ids_encontrados = resultados_chroma['ids'][0]
        distancias = resultados_chroma['distances'][0]
        
        resultados = []
        for doc_id, distancia in zip(ids_encontrados, distancias):
            # ChromaDB devuelve distancia L2 (menor es mejor). 
            # Lo convertimos a score (mayor es mejor) para que sea compatible con Jaccard/Coseno
            score = 1 / (1 + distancia) 
            resultados.append((doc_id, score))
            
        # Ordenamos de mayor a menor score, igual que hace Mateo
        resultados.sort(key=lambda x: x[1], reverse=True)
        return resultados

def iniciar_cli():
    print("\nInicializando Sistema de Recuperación de Información...")
    ruta_corpus = BASE_DIR / "data" / "processed" / "corpus_limpio.csv"
    df_corpus = pd.read_csv(ruta_corpus).fillna('')
    
    # Instanciamos los modelos
    jac = BuscadorJaccard()
    cos = BuscadorCoseno()
    sem = BuscadorSemantico()
    # bm25 = BuscadorBM25()  # Descomentar cuando Mateo termine el suyo

    while True:
        print("\n" + "="*60)
        print("   SISTEMA DE RECUPERACIÓN DE OFERTAS LABORALES (EPN)")
        print("="*60)
        print("1. Modelo Binario (Jaccard)")
        print("2. Modelo Vectorial (TF-IDF - Similitud Coseno)")
        print("3. Modelo Probabilístico (BM25) - En desarrollo")
        print("4. Recuperación Semántica (Embeddings)")
        print("5. Salir")
        
        opcion = input("\nSeleccione un modelo (1-5): ").strip()
        
        if opcion == '5':
            print("Saliendo del sistema...")
            break
            
        if opcion not in ['1', '2', '3', '4']:
            print("Opción inválida.")
            continue
            
        consulta = input("\nIngrese su consulta: ").strip()
        if not consulta:
            continue
            
        if opcion == '1':
            res = jac.buscar(consulta, top_k=10)
            mostrar_resultados(res, "Jaccard", df_corpus)
        elif opcion == '2':
            res = cos.buscar(consulta, top_k=10)
            mostrar_resultados(res, "Coseno TF-IDF", df_corpus)
        elif opcion == '3':
            print("\n[Módulo BM25 pendiente de integración]")
            # res = bm25.buscar(consulta, top_k=10)
            # mostrar_resultados(res, "BM25", df_corpus)
        elif opcion == '4':
            res = sem.buscar(consulta, top_k=10)
            mostrar_resultados(res, "Semántico (ChromaDB)", df_corpus)

if __name__ == "__main__":
    iniciar_cli()
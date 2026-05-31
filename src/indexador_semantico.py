import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_CORPUS = BASE_DIR / 'data' / 'processed' / 'corpus_limpio.csv'
RUTA_DB = BASE_DIR / 'data' / 'processed' / 'chroma_db'

print("Cargando modelo de embeddings (paraphrase-multilingual-MiniLM-L12-v2)...")
modelo = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

print("Cargando corpus limpio...")
df = pd.read_csv(RUTA_CORPUS).fillna('')

# Unimos título, carrera requerida y descripción
textos_para_embed = (df['job_title'] + " - " + 
                     df['careers_required'] + ". " + 
                     df['description_final']).tolist()

ids_documentos = df['job_id'].astype(str).tolist()

print(f"Generando embeddings para {len(textos_para_embed)} documentos...")
embeddings = modelo.encode(textos_para_embed, show_progress_bar=True, batch_size=32).tolist()

print("Inicializando ChromaDB...")
os.makedirs(RUTA_DB, exist_ok=True)
cliente = chromadb.PersistentClient(path=str(RUTA_DB))

try:
    cliente.delete_collection(name="empleos")
except:
    pass

coleccion = cliente.create_collection(name="empleos")

print("Insertando vectores en ChromaDB...")
batch_size = 5000
for i in range(0, len(ids_documentos), batch_size):
    coleccion.add(
        embeddings=embeddings[i:i+batch_size],
        documents=textos_para_embed[i:i+batch_size],
        ids=ids_documentos[i:i+batch_size]
    )
    print(f"Insertados {min(i+batch_size, len(ids_documentos))} / {len(ids_documentos)}")

print("¡Base de datos vectorial lista y guardada en disco!")
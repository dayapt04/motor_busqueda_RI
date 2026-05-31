import pandas as pd
from pathlib import Path
import unicodedata
import re

BASE_DIR = Path(__file__).resolve().parent.parent
DIR_ENTRADA = BASE_DIR / 'data' / 'raw'
ARCHIVO_SALIDA = BASE_DIR / 'data' / 'processed' / 'corpus_limpio.csv'

def limpiar_texto(texto: str) -> str:
    """
    Convierte el texto a minúsculas, elimina tildes (acentos diacríticos) 
    y caracteres especiales, manteniendo solo letras y números.
    Ideal para preparar el texto en español para el Índice Invertido.
    """
    if not isinstance(texto, str):
        return ""
        
    texto = texto.lower()
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^a-z0-9\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    
    return texto

def construir_corpus(input_dir: str, output_file: str) -> None:
    """
    Explora las carpetas crudas, concatena los CSVs de ofertas laborales, 
    limpia el texto y exporta el corpus unificado.
    """
    print(f"Buscando archivos en: {input_dir}")
    ruta_base = Path(input_dir)
    dfs = []

    for carpeta in ruta_base.iterdir():
        if carpeta.is_dir():
            archivo = carpeta / f"{carpeta.name}_Merged.csv"
            if archivo.exists():
                try:
                    df = pd.read_csv(archivo, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(archivo, encoding='latin-1')
                dfs.append(df)

    if not dfs:
        print("Error: No se encontraron archivos CSV para procesar.")
        return

    print(f"Concatenando {len(dfs)} archivos...")
    corpus = pd.concat(dfs, ignore_index=True)
    
    corpus = corpus[['job_id', 'job_title', 'description_final', 'careers_required']]
    corpus = corpus[corpus['description_final'].notna()].reset_index(drop=True)

    print("Aplicando limpieza de texto...")
    corpus['job_title']         = corpus['job_title'].fillna('').apply(limpiar_texto)
    corpus['description_final'] = corpus['description_final'].fillna('').apply(limpiar_texto)
    corpus['careers_required']  = corpus['careers_required'].fillna('').apply(limpiar_texto)

    # Agrupar todas las carreras por job_id antes de eliminar duplicados.
    # Una oferta puede aparecer en múltiples CSVs si es relevante para varias carreras.
    # Conservar esa información es esencial para construir qrels correctos.
    carreras_por_job = (
        corpus.groupby("job_id")["careers_required"]
        .apply(lambda x: list(x[x != ""].unique()))
        .to_dict()
    )
    corpus = corpus.drop_duplicates(subset="job_id", keep="first").reset_index(drop=True)
    corpus["all_careers"] = corpus["job_id"].map(carreras_por_job)
    print(f"Documentos únicos tras deduplicación: {len(corpus)}")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(output_file, index=False, encoding='utf-8')
    print(f"Corpus limpio guardado en: {output_file}")


if __name__ == '__main__':
    construir_corpus(DIR_ENTRADA, ARCHIVO_SALIDA)
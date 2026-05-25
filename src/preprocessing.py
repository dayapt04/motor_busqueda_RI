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
    # Normalización NFD separa el carácter de su acento
    texto = unicodedata.normalize('NFD', texto)
    # Filtra los caracteres que son marcas diacríticas (Mn)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    # Reemplaza todo lo que no sea letra minúscula o número por un espacio
    texto = re.sub(r'[^a-z0-9\s]', ' ', texto)
    # Elimina espacios múltiples
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

    # Iterar sobre las carpetas de carreras
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
    
    # Seleccionar columnas y eliminar nulos en la descripción
    corpus = corpus[['job_id', 'job_title', 'description_final', 'careers_required']]
    corpus = corpus[corpus['description_final'].notna()].reset_index(drop=True)

    print("Aplicando limpieza de texto (minúsculas, sin tildes, sin caracteres especiales)...")
    corpus['job_title'] = corpus['job_title'].fillna('').apply(limpiar_texto)
    corpus['description_final'] = corpus['description_final'].fillna('').apply(limpiar_texto)
    corpus['careers_required'] = corpus['careers_required'].fillna('').apply(limpiar_texto)

    # Exportar resultados
    # Asegurar que el directorio de salida exista
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    corpus.to_csv(output_file, index=False, encoding='utf-8')
    print(f"¡Éxito! Documentos totales: {len(corpus)}")
    print(f"Corpus limpio guardado en: {output_file}")


# Bloque principal para ejecución directa
if __name__ == '__main__':
    construir_corpus(DIR_ENTRADA, ARCHIVO_SALIDA)
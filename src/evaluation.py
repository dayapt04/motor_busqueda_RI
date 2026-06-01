# src/evaluation.py
import ast
import numpy as np
import pandas as pd
from pathlib import Path
 
from models import cargar_indice, BuscadorJaccard, BuscadorCoseno, BuscadorBM25
from embeddings import BuscadorSemantico
 
BASE_DIR    = Path(__file__).resolve().parent.parent
RUTA_CORPUS = BASE_DIR / "data" / "processed" / "corpus_limpio.csv"
 
# Claves coinciden exactamente con los valores de all_careers en el corpus
CONSULTAS = {
    "administracion de empresas"    : "gestion empresarial recursos humanos administrador finanzas",
    "agroindustria"                 : "agroindustria procesamiento alimentos cadena productiva agricola",
    "ciencia de datos"              : "cientifico de datos analisis estadistico visualizacion python sql",
    "computacion"                   : "programador computacion algoritmos estructuras de datos desarrollo",
    "economia"                      : "economista analisis macroeconomico mercado finanzas indicadores",
    "electricidad"                  : "ingeniero electrico redes electricas subestaciones energia",
    "electronica y automatizacion"  : "electronica automatizacion control industrial plc sensores",
    "fisica"                        : "fisica investigacion laboratorio simulacion modelado",
    "geologia"                      : "geologo exploracion mineria rocas yacimientos",
    "ingenieria ambiental"          : "ingeniero ambiental medio ambiente impacto ambiental sostenibilidad",
    "ingenieria civil"              : "ingeniero civil construccion infraestructura obras hidraulica",
    "ingenieria de la produccion"   : "produccion manufactura operaciones cadena de suministro calidad",
    "ingenieria quimica"            : "ingeniero quimico procesos quimicos planta industrial laboratorio",
    "inteligencia artificial"       : "inteligencia artificial deep learning redes neuronales nlp vision",
    "matematica"                    : "matematico modelado matematico estadistica investigacion docencia",
    "matematica aplicada"           : "matematica aplicada optimizacion simulacion numerica calculo",
    "materiales"                    : "ingenieria de materiales metalurgia ensayos mecanicos polimeros",
    "mecanica"                      : "ingeniero mecanico diseno mecanico manufactura turbinas mantenimiento",
    "mecatronica"                   : "mecatronica robotica sistemas embebidos automatizacion diseno",
    "petroleos"                     : "petroleo gas upstream downstream refineria yacimientos",
    "sistemas de informacion"       : "sistemas de informacion erp bases de datos infraestructura ti",
    "software"                      : "desarrollador software backend frontend arquitectura agile devops",
    "tecnologias de la informacion" : "tecnologias informacion soporte redes seguridad helpdesk",
    "telecomunicaciones"            : "telecomunicaciones redes fibra optica protocolos transmision",
}
 
 
def construir_qrels(corpus_df: pd.DataFrame) -> dict:
    """
    Construye qrels desde la columna all_careers.
    Para cada carrera, los relevantes son todos los job_ids que la tienen en all_careers.
    """
    print("Construyendo ground truth basado en metadatos de la EPN...")
    qrels = {carrera: set() for carrera in CONSULTAS}
 
    for _, fila in corpus_df.iterrows():
        try:
            carreras = ast.literal_eval(fila["all_careers"])
        except Exception:
            carreras = []
        for carrera in carreras:
            if carrera.lower().strip() in qrels:
                qrels[carrera.lower().strip()].add(str(fila["job_id"]))
 
    return qrels
 
 
def precision_at_k(recuperados: list, relevantes: set, k: int = 10) -> float:
    """Fracción de documentos relevantes entre los primeros k resultados."""
    if not recuperados:
        return 0.0
    return sum(1 for doc in recuperados[:k] if doc in relevantes) / k
 
 
def recall_at_k(recuperados: list, relevantes: set, k: int = 10) -> float:
    """Fracción de relevantes totales que aparecen en los primeros k resultados."""
    if not relevantes:
        return 0.0
    return sum(1 for doc in recuperados[:k] if doc in relevantes) / len(relevantes)
 
 
def average_precision(recuperados: list, relevantes: set) -> float:
    """
    Average Precision (AP) para una consulta.
    Promedia la precisión en cada posición donde se recupera un relevante.
    """
    if not relevantes or not recuperados:
        return 0.0
    relevantes_vistos = 0
    suma = 0.0
    for i, doc in enumerate(recuperados):
        if doc in relevantes:
            relevantes_vistos += 1
            suma += relevantes_vistos / (i + 1)
    return suma / len(relevantes)
 
 
def evaluar_sistema(top_k: int = 10) -> pd.DataFrame:
    """
    Ejecuta la evaluación completa sobre las 24 consultas y retorna
    un DataFrame con Precision@10, Recall@10 y MAP por modelo.
    """
    print("\nINICIANDO EVALUACIÓN AUTOMÁTICA DEL SISTEMA DE RI")
 
    df_corpus = pd.read_csv(RUTA_CORPUS).fillna("")
    df_corpus["job_id"] = df_corpus["job_id"].astype(str)
 
    qrels = construir_qrels(df_corpus)
    for carrera, rels in list(qrels.items())[:3]:
        print(f"  -> '{carrera}': {len(rels)} documentos relevantes")
 
    print("\nInicializando modelos...")
    indice  = cargar_indice()
    modelos = {
        "Jaccard"          : BuscadorJaccard(datos_indice=indice),
        "TF-IDF"           : BuscadorCoseno(datos_indice=indice),
        "BM25"             : BuscadorBM25(datos_indice=indice),
        "Embeddings Densos": BuscadorSemantico(),
    }
 
    metricas = {nombre: {"P@10": [], "R@10": [], "AP": []} for nombre in modelos}
 
    print("\nEvaluando 24 consultas...")
    for carrera, query in CONSULTAS.items():
        relevantes = qrels.get(carrera, set())
        if not relevantes:
            continue
        for nombre, modelo in modelos.items():
            recuperados = [str(r[0]) for r in modelo.buscar(query, top_k=50)]
            metricas[nombre]["P@10"].append(precision_at_k(recuperados, relevantes, k=top_k))
            metricas[nombre]["R@10"].append(recall_at_k(recuperados, relevantes, k=top_k))
            metricas[nombre]["AP"].append(average_precision(recuperados, relevantes))
 
    resultados = [
        {
            "Modelo"      : nombre,
            "Precision@10": round(float(np.mean(vals["P@10"])), 4),
            "Recall@10"   : round(float(np.mean(vals["R@10"])), 4),
            "MAP"         : round(float(np.mean(vals["AP"])),   4),
        }
        for nombre, vals in metricas.items()
    ]
 
    return pd.DataFrame(resultados)
 
 
if __name__ == "__main__":
    df = evaluar_sistema()
    print("\n\nRESULTADOS FINALES:")
    print(df.round(4).to_string(index=False))
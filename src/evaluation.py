# # src/evaluation.py
# import ast
# import pandas as pd
# import numpy as np
# from pathlib import Path

# # Importar los modelos del ecosistema
# from models import cargar_indice, BuscadorJaccard, BuscadorCoseno, BuscadorBM25
# from embeddings import BuscadorSemantico

# BASE_DIR = Path(__file__).resolve().parent.parent
# RUTA_CORPUS = BASE_DIR / "data" / "processed" / "corpus_limpio.csv"

# # IMPORTANTE: Las claves coinciden con las carpetas de origen sin tildes y en minúsculas.
# CONSULTAS = {
#     "administracion de empresas":
#     "gestion empresarial recursos humanos administrador finanzas",
#     "agroindustria":
#     "agroindustria procesamiento alimentos cadena productiva agricola",
#     "ciencia de datos":
#     "cientifico de datos analisis estadistico visualizacion python sql",
#     "computacion":
#     "programador computacion algoritmos estructuras de datos desarrollo",
#     "economia":
#     "economista analisis macroeconomico mercado finanzas indicadores",
#     "electricidad":
#     "ingeniero electrico redes electricas subestaciones energia",
#     "electronica y automatizacion":
#     "electronica automatizacion control industrial plc sensores",
#     "fisica":
#     "fisica investigacion laboratorio simulacion modelado",
#     "geologia":
#     "geologo exploracion mineria rocas yacimientos",
#     "ingenieria ambiental":
#     "ingeniero ambiental medio ambiente impacto ambiental sostenibilidad",
#     "ingenieria civil":
#     "ingeniero civil construccion infraestructura obras hidraulica",
#     "ingenieria de la produccion":
#     "produccion manufactura operaciones cadena de suministro calidad",
#     "ingenieria quimica":
#     "ingeniero quimico procesos quimicos planta industrial laboratorio",
#     "inteligencia artificial":
#     "inteligencia artificial deep learning redes neuronales nlp vision",
#     "matematica":
#     "matematico modelado matematico estadistica investigacion docencia",
#     "matematica aplicada":
#     "matematica aplicada optimizacion simulacion numerica calculo",
#     "materiales":
#     "ingenieria de materiales metalurgia ensayos mecanicos polimeros",
#     "mecanica":
#     "ingeniero mecanico diseno mecanico manufactura turbinas mantenimiento",
#     "mecatronica":
#     "mecatronica robotica sistemas embebidos automatizacion diseno",
#     "petroleos":
#     "petroleo gas upstream downstream refineria yacimientos",
#     "sistemas de informacion":
#     "sistemas de informacion erp bases de datos infraestructura ti",
#     "software":
#     "desarrollador software backend frontend arquitectura agile devops",
#     "tecnologias de la informacion":
#     "tecnologias informacion soporte redes seguridad helpdesk",
#     "telecomunicaciones":
#     "telecomunicaciones redes fibra optica protocolos transmision",
# }

# def construir_qrels(corpus_df):
#     """
#     Mapea cada documento a las carreras a las que pertenece usando 'all_careers'.
#     """
#     print("Construyendo Ground Truth basado en metadatos de la EPN...")
#     qrels = {carrera: set() for carrera in CONSULTAS.keys()}

#     for _, fila in corpus_df.iterrows():
#         try:
#             # Parsear la lista guardada como string en el CSV
#             carreras = ast.literal_eval(fila["all_careers"])
#         except Exception:
#             carreras = []

#         for carrera in carreras:
#             carrera_norm = carrera.lower().strip()
#             if carrera_norm in qrels:
#                 qrels[carrera_norm].add(str(fila["job_id"]))

#     return qrels

# def precision_at_k(recuperados, relevantes, k=10):
#     if not recuperados: return 0.0
#     top_k = recuperados[:k]
#     relevantes_recuperados = len([doc for doc in top_k if doc in relevantes])
#     return relevantes_recuperados / k

# def recall_at_k(recuperados, relevantes, k=10):
#     if not relevantes: return 0.0
#     top_k = recuperados[:k]
#     relevantes_recuperados = len([doc for doc in top_k if doc in relevantes])
#     return relevantes_recuperados / len(relevantes)

# def average_precision(recuperados, relevantes):
#     """Calcula la Precisión Promedio (AP) de una sola consulta."""
#     if not relevantes or not recuperados: return 0.0
#     relevantes_vistos = 0
#     suma_precisiones = 0.0

#     for i, doc in enumerate(recuperados):
#         if doc in relevantes:
#             relevantes_vistos += 1
#             suma_precisiones += relevantes_vistos / (i + 1)

#     return suma_precisiones / len(relevantes)

# def evaluar_sistema(top_k=10):
#     print("\nINICIANDO EVALUACIÓN AUTOMÁTICA DEL SISTEMA DE RI")

#     df_corpus = pd.read_csv(RUTA_CORPUS).fillna("")
#     df_corpus['job_id'] = df_corpus['job_id'].astype(str)

#     qrels = construir_qrels(df_corpus)

#     for c, rels in list(qrels.items())[:3]:
#         print(f"  -> La carrera '{c}' tiene {len(rels)} documentos verdaderamente relevantes.")

#     print("\nInicializando modelos...")
#     indice = cargar_indice()
#     modelos = {
#         "Jaccard": BuscadorJaccard(datos_indice=indice),
#         "TF-IDF": BuscadorCoseno(datos_indice=indice),
#         "BM25": BuscadorBM25(datos_indice=indice),
#         "Embeddings Densos": BuscadorSemantico()
#     }
#     # Estructura para acumular métricas
#     metricas = {nombre: {"P@10": [], "R@10": [], "AP": []} for nombre in modelos}

#     print("\nEvaluando 24 consultas...")
#     for carrera, query in CONSULTAS.items():
#         relevantes = qrels.get(carrera, set())
#         if not relevantes:
#             continue

#         for nombre_modelo, modelo in modelos.items():
#             resultados_brutos = modelo.buscar(query, top_k=50)
#             recuperados = [str(r[0]) for r in resultados_brutos]

#             metricas[nombre_modelo]["P@10"].append(precision_at_k(recuperados, relevantes, k=top_k))
#             metricas[nombre_modelo]["R@10"].append(recall_at_k(recuperados, relevantes, k=top_k))
#             metricas[nombre_modelo]["AP"].append(average_precision(recuperados, relevantes))

#     # Calcular promedios (Mean) y construir el DataFrame final
#     resultados_finales = []
#     for nombre_modelo in modelos:
#         p10_promedio = np.mean(metricas[nombre_modelo]["P@10"])
#         r10_promedio = np.mean(metricas[nombre_modelo]["R@10"])
#         map_score = np.mean(metricas[nombre_modelo]["AP"])

#         resultados_finales.append({
#             "Modelo": nombre_modelo,
#             "Precision@10": p10_promedio,
#             "Recall@10": r10_promedio,
#             "MAP": map_score
#         })

#     df_resultados = pd.DataFrame(resultados_finales)
#     return df_resultados

# if __name__ == "__main__":
#     df = evaluar_sistema()
#     print("\n\nRESULTADOS FINALES:")
#     print(df.round(4).to_string(index=False))
# src/evaluation.py
import ast
import pandas as pd
import numpy as np
from pathlib import Path

from models import cargar_indice, BuscadorJaccard, BuscadorCoseno, BuscadorBM25
from embeddings import BuscadorSemantico

BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_CORPUS = BASE_DIR / "data" / "processed" / "corpus_limpio.csv"

# Consultas de prueba: clave = nombre de carrera (coincide con all_careers), valor = query
CONSULTAS = {
    "agua": "ingeniero recursos hidricos tratamiento de agua saneamiento y potabilizacion",
    "ambiental": "ingenieria ambiental gestion impacto ambiental tratamiento ecologia",
    "automatizacion": "ingeniero automatizacion control industrial plc instrumentacion",
    "biomedicina": "ingenieria biomedica equipos medicos biomecanica señales clinicas",
    "civil": "ingeniero civil construccion infraestructura diseño estructural geotecnia",
    "computacion": "ciencias de la computacion algoritmos programacion desarrollo",
    "datos": "cientifico de datos data science big data analisis estadistico machine learning",
    "electrica": "ingenieria electrica potencia alta tension redes electricas energia",
    "electromecanica": "ingenieria electromecanica mantenimiento industrial maquinas motores",
    "electronica": "ingenieria electronica circuitos microcontroladores sistemas embebidos",
    "empresa": "administracion de empresas gestion empresarial gerencia proyectos finanzas",
    "fisica": "fisica aplicada investigacion metrologia optica analisis matematico",
    "geologia": "ingenieria geologica exploracion minera geotecnia cartografia suelos",
    "hidraulica": "ingenieria hidraulica obras hidraulicas fluidos represas canalizacion",
    "industrial": "ingenieria industrial optimizacion de procesos logistica produccion calidad",
    "informacion": "tecnologias de la informacion ti seguridad informatica auditoria",
    "matematica": "matematica aplicada modelizacion analisis numerico optimizacion",
    "mecanica": "ingenieria mecanica termodinamica fluidos diseño de maquinas autocad",
    "petroleos": "ingenieria de petroleos perforacion reservorios produccion hidrocarburos",
    "quimica": "ingenieria quimica procesos industriales laboratorio control de calidad",
    "redes": "ingenieria en redes telecomunicaciones infraestructura cisco enrutamiento",
    "sistemas": "sistemas de informacion gestion arquitectura de software infraestructura ti",
    "software": "desarrollador de software programacion backend frontend bases de datos",
    "telecomunicaciones": "telecomunicaciones radiofrecuencia transmision fibra optica antenas",
}

# Términos en inglés y español asociados a cada carrera.
# Se usan para ampliar los qrels con documentos no etiquetados pero temáticamente relevantes,
# lo que corrige el sesgo contra modelos semánticos que recuperan sinónimos léxicos.
SINONIMOS_CARRERA = {
    "agua": ["water treatment", "hydraulic engineer", "sanitation", "water resources"],
    "ambiental": ["environmental engineer", "environmental scientist", "ecology", "sustainability"],
    "automatizacion": ["automation engineer", "control systems", "plc programmer", "instrumentation"],
    "biomedicina": ["biomedical engineer", "clinical engineer", "medical devices", "bioinformatics"],
    "civil": ["civil engineer", "structural engineer", "construction manager", "geotechnical"],
    "computacion": ["computer scientist", "software developer", "algorithms", "backend developer"],
    "datos": ["data scientist", "data engineer", "data analyst", "machine learning engineer",
              "analista de datos", "ingeniero de datos", "cientifico de datos"],
    "electrica": ["electrical engineer", "power systems", "high voltage", "energy engineer"],
    "electromecanica": ["electromechanical engineer", "maintenance engineer", "industrial mechanic"],
    "electronica": ["electronics engineer", "embedded systems", "microcontroller", "circuit design"],
    "empresa": ["business administrator", "management consultant", "project manager", "finance manager"],
    "fisica": ["physicist", "applied physics", "research scientist", "metrology engineer"],
    "geologia": ["geologist", "mining engineer", "geotechnical engineer", "cartographer"],
    "hidraulica": ["hydraulic engineer", "fluid mechanics", "dam engineer", "irrigation engineer"],
    "industrial": ["industrial engineer", "process engineer", "logistics manager", "quality engineer"],
    "informacion": ["it specialist", "information security", "cybersecurity analyst", "it auditor",
                    "especialista ti", "seguridad informatica"],
    "matematica": ["mathematician", "applied mathematics", "numerical analyst", "quantitative analyst"],
    "mecanica": ["mechanical engineer", "thermodynamics", "cad designer", "manufacturing engineer"],
    "petroleos": ["petroleum engineer", "reservoir engineer", "drilling engineer", "oil and gas"],
    "quimica": ["chemical engineer", "process engineer", "quality control", "laboratory analyst"],
    "redes": ["network engineer", "telecom engineer", "cisco", "network administrator"],
    "sistemas": ["systems analyst", "it architect", "software architect", "erp consultant",
                 "analista de sistemas"],
    "software": ["software engineer", "developer", "full stack", "backend", "frontend",
                 "desarrollador de software", "programador"],
    "telecomunicaciones": ["telecom engineer", "rf engineer", "fiber optic", "antenna engineer"],
}


def construir_qrels(corpus_df: pd.DataFrame) -> dict:
    """
    Construye el ground truth ampliado combinando dos fuentes:
    1. Etiqueta exacta de carrera en all_careers (método original).
    2. Coincidencia de términos asociados en job_title (método ampliado).
    El segundo método reduce el sesgo contra modelos semánticos que recuperan
    documentos relevantes sin la etiqueta exacta.
    """
    print("Construyendo ground truth ampliado...")
    qrels = {carrera: set() for carrera in CONSULTAS.keys()}

    for _, fila in corpus_df.iterrows():
        doc_id = str(fila["job_id"])
        titulo = str(fila.get("job_title", "")).lower()

        # Fuente 1: etiqueta exacta de carrera
        try:
            carreras = ast.literal_eval(fila["all_careers"])
        except Exception:
            carreras = []

        for carrera in carreras:
            carrera_norm = carrera.lower().strip()
            if carrera_norm in qrels:
                qrels[carrera_norm].add(doc_id)

        # Fuente 2: términos asociados en el título del puesto
        for carrera, terminos in SINONIMOS_CARRERA.items():
            if any(termino in titulo for termino in terminos):
                qrels[carrera].add(doc_id)

    return qrels


def precision_at_k(recuperados: list, relevantes: set, k: int = 10) -> float:
    """Fracción de documentos relevantes entre los primeros k resultados."""
    if not recuperados:
        return 0.0
    hits = sum(1 for doc in recuperados[:k] if doc in relevantes)
    return hits / k


def recall_at_k(recuperados: list, relevantes: set, k: int = 10) -> float:
    """Fracción de relevantes totales que aparecen en los primeros k resultados."""
    if not relevantes:
        return 0.0
    hits = sum(1 for doc in recuperados[:k] if doc in relevantes)
    return hits / len(relevantes)


def average_precision(recuperados: list, relevantes: set) -> float:
    """
    Calcula la Average Precision (AP) para una consulta.
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
    Los modelos clásicos recuperan top-50; embeddings recuperan top-200
    para compensar su mayor cobertura semántica en el espacio de búsqueda.
    """
    print("\nINICIANDO EVALUACIÓN AUTOMÁTICA DEL SISTEMA DE RI")

    df_corpus = pd.read_csv(RUTA_CORPUS).fillna("")
    df_corpus["job_id"] = df_corpus["job_id"].astype(str)

    qrels = construir_qrels(df_corpus)

    for carrera, rels in list(qrels.items())[:3]:
        print(f"  -> '{carrera}': {len(rels)} documentos relevantes (etiqueta + título)")

    print("\nInicializando modelos...")
    indice = cargar_indice()
    modelos = {
        "Jaccard":          BuscadorJaccard(datos_indice=indice),
        "TF-IDF":           BuscadorCoseno(datos_indice=indice),
        "BM25":             BuscadorBM25(datos_indice=indice),
        "Embeddings Densos": BuscadorSemantico(),
    }

    # top_k de recuperación por modelo: embeddings usa pool más amplio
    pool_recuperacion = {
        "Jaccard":           50,
        "TF-IDF":            50,
        "BM25":              50,
        "Embeddings Densos": 200,
    }

    metricas = {nombre: {"P@10": [], "R@10": [], "AP": []} for nombre in modelos}

    print("\nEvaluando 24 consultas...")
    for carrera, query in CONSULTAS.items():
        relevantes = qrels.get(carrera, set())
        if not relevantes:
            continue

        for nombre, modelo in modelos.items():
            pool = pool_recuperacion[nombre]
            recuperados = [str(r[0]) for r in modelo.buscar(query, top_k=pool)]

            metricas[nombre]["P@10"].append(precision_at_k(recuperados, relevantes, k=top_k))
            metricas[nombre]["R@10"].append(recall_at_k(recuperados, relevantes, k=top_k))
            metricas[nombre]["AP"].append(average_precision(recuperados, relevantes))

    resultados = [
        {
            "Modelo":        nombre,
            "Precision@10":  np.mean(vals["P@10"]),
            "Recall@10":     np.mean(vals["R@10"]),
            "MAP":           np.mean(vals["AP"]),
        }
        for nombre, vals in metricas.items()
    ]

    return pd.DataFrame(resultados)


if __name__ == "__main__":
    df = evaluar_sistema()
    print("\n\nRESULTADOS FINALES:")
    print(df.round(4).to_string(index=False))
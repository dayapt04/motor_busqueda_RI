# main.py
# Interfaz de línea de comandos del sistema de recuperación de información.
# Uso: python main.py

import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from models import cargar_indice, BuscadorJaccard, BuscadorCoseno, BuscadorBM25
from embeddings import BuscadorSemantico

console = Console()


def mostrar_resultados(resultados: list, modelo: str, consulta: str, corpus_df) -> None:
    tabla = Table(
        title=f'🔍  "{consulta}"',
        box=box.DOUBLE_EDGE,
        show_header=True,
        header_style="bold cyan",
    )
    tabla.add_column("#", style="dim", width=4, justify="right")
    tabla.add_column("Título", width=48)
    tabla.add_column("Score", style="green", width=10, justify="right")
    tabla.caption = f"Modelo: {modelo}  |  {len(resultados)} resultado(s)"

    for i, (doc_id, score) in enumerate(resultados, 1):
        titulo = doc_id[:46] + "..."
        fila = corpus_df[corpus_df["job_id"] == doc_id]
        if not fila.empty:
            titulo = fila.iloc[0]["job_title"]
        tabla.add_row(str(i), titulo, f"{score:.4f}")

    console.print(tabla)


def iniciar_cli():
    console.print("\n[bold cyan]Inicializando sistema...[/bold cyan]")
    df_corpus = pd.read_csv(
        BASE_DIR / "data" / "processed" / "corpus_limpio.csv"
    ).fillna("")

    # Índice compartido — se carga una sola vez para los tres modelos clásicos
    datos = cargar_indice()
    jac = BuscadorJaccard(datos_indice=datos)
    cos = BuscadorCoseno(datos_indice=datos)
    bm25 = BuscadorBM25(datos_indice=datos)
    sem = BuscadorSemantico()

    modelos = {
        "1": (jac, "Jaccard"),
        "2": (cos, "Coseno TF-IDF"),
        "3": (bm25, "BM25"),
        "4": (sem, "Semántico (ChromaDB)"),
    }

    while True:
        console.print("\n[bold]" + "=" * 60 + "[/bold]")
        console.print(
            "[bold cyan]   SISTEMA DE RECUPERACIÓN DE OFERTAS LABORALES — EPN[/bold cyan]"
        )
        console.print("[bold]" + "=" * 60 + "[/bold]")
        console.print("  1.  Modelo Binario     [dim](Jaccard)[/dim]")
        console.print("  2.  Modelo Vectorial   [dim](Coseno TF-IDF)[/dim]")
        console.print("  3.  Modelo BM25")
        console.print("  4.  Recuperación Semántica [dim](Embeddings)[/dim]")
        console.print("  5.  Salir")

        opcion = input("\nSeleccione un modelo (1-5): ").strip()

        if opcion == "5":
            console.print("[bold red]Saliendo...[/bold red]")
            break

        if opcion not in modelos:
            console.print("[red]Opción inválida.[/red]")
            continue

        consulta = input("\nIngrese su consulta: ").strip()
        if not consulta:
            continue

        buscador, nombre = modelos[opcion]
        resultados = buscador.buscar(consulta, top_k=10)
        mostrar_resultados(resultados, nombre, consulta, df_corpus)


if __name__ == "__main__":
    iniciar_cli()

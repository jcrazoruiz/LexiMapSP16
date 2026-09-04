from __future__ import annotations

import argparse
import sys

from dataclasses import asdict
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SRC_DIRECTORY = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIRECTORY),
    )


from leximapsp16.wikipedia_heldout_extractor import (
    WikipediaHeldoutExtractor,
)

from leximapsp16.wikipedia_wikitext_cleaner import (
    WikipediaWikitextCleaner,
)


SNAPSHOT_DATE = "20260801"

SEGMENT_FILENAME = (
    "eswiki-20260801-pages-articles-"
    "multistream2.xml-p159401p693323.bz2"
)

SEGMENT_LABEL = "segment2"

DUMP_PATH = (
    PROJECT_ROOT
    / "data"
    / "heldout"
    / "wikipedia_es_20260801_segment2"
    / "dump"
    / SEGMENT_FILENAME
)


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Extrae una muestra held-out determinista "
            "de Wikipedia para LexiMapSp-16."
        )
    )

    parser.add_argument(
        "--skip",
        type=int,
        required=True,
        help=(
            "Cantidad de artículos válidos que se omitirán "
            "antes de iniciar la muestra."
        ),
    )

    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help=(
            "Cantidad de artículos válidos que se extraerán "
            "después del salto."
        ),
    )

    arguments = parser.parse_args()

    if arguments.skip < 0:
        parser.error(
            "--skip no puede ser negativo."
        )

    if arguments.count <= 0:
        parser.error(
            "--count debe ser mayor que cero."
        )

    return arguments


def build_dataset_name(
    skip_articles: int,
    target_articles: int,
) -> str:

    return (
        "wikipedia_es_"
        f"{SNAPSHOT_DATE}_"
        f"{SEGMENT_LABEL}_"
        f"skip{skip_articles}_"
        f"count{target_articles}"
    )


def format_elapsed(
    seconds: float,
) -> str:

    total = int(
        seconds
    )

    hours, remainder = divmod(
        total,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


def main() -> int:

    arguments = parse_arguments()

    skip_articles = arguments.skip
    target_articles = arguments.count

    dataset_name = (
        build_dataset_name(
            skip_articles=skip_articles,
            target_articles=target_articles,
        )
    )

    heldout_directory = (
        PROJECT_ROOT
        / "data"
        / "heldout"
        / dataset_name
    )

    article_directory = (
        heldout_directory
        / "articles"
    )

    manifest_path = (
        heldout_directory
        / "manifest.jsonl"
    )

    first_valid_article = (
        skip_articles + 1
    )

    last_valid_article = (
        skip_articles
        + target_articles
    )

    print()

    print(
        "LexiMapSp-16 - Extracción Wikipedia Held-Out"
    )

    print("=" * 88)

    print(
        "Rol experimental      : HELD_OUT"
    )

    print(
        "Snapshot               : "
        f"{SNAPSHOT_DATE}"
    )

    print(
        "Segmento               : "
        f"{SEGMENT_FILENAME}"
    )

    print(
        "Rango del segmento     : "
        "p159401 -> p693323"
    )

    print(
        "Artículos a omitir     : "
        f"{skip_articles:,}"
    )

    print(
        "Artículos a procesar   : "
        f"{target_articles:,}"
    )

    print(
        "Posiciones válidas     : "
        f"{first_valid_article:,}"
        " -> "
        f"{last_valid_article:,}"
    )

    print(
        "Dataset                : "
        f"{dataset_name}"
    )

    print(
        "Directorio artículos   : "
        f"{article_directory}"
    )

    print(
        "Manifiesto             : "
        f"{manifest_path}"
    )

    print("-" * 88)

    if not DUMP_PATH.exists():

        print(
            "ERROR: no existe el dump:"
        )

        print(
            DUMP_PATH
        )

        return 1

    extractor = (
        WikipediaHeldoutExtractor(
            cleaner=(
                WikipediaWikitextCleaner()
            ),
            output_directory=(
                article_directory
            ),
            manifest_path=(
                manifest_path
            ),
            snapshot_date=(
                SNAPSHOT_DATE
            ),
            target_articles=(
                target_articles
            ),
            skip_articles=(
                skip_articles
            ),
        )
    )

    timer_start = perf_counter()

    try:

        result = extractor.extract(
            DUMP_PATH
        )

    except Exception as exc:

        print()

        print(
            "ERROR DURANTE LA EXTRACCIÓN"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return 1

    elapsed = (
        perf_counter()
        - timer_start
    )

    print()

    print("=" * 88)

    print(
        "RESULTADO HELD-OUT"
    )

    print("-" * 88)

    for key, value in (
        asdict(
            result
        ).items()
    ):

        print(
            f"{key:<25}: "
            f"{value:,}"
        )

    print(
        "Tiempo total             : "
        f"{format_elapsed(elapsed)}"
    )

    print(
        "Artículos válidos usados : "
        f"{first_valid_article:,}"
        " -> "
        f"{last_valid_article:,}"
    )

    print(
        "Directorio artículos     : "
        f"{article_directory}"
    )

    print(
        "Manifiesto               : "
        f"{manifest_path}"
    )

    print("-" * 88)

    if (
        result.skipped_articles
        != skip_articles
    ):

        print(
            "RESULTADO FINAL: "
            "CANTIDAD OMITIDA INCORRECTA"
        )

        return 1

    if (
        result.extracted
        != target_articles
    ):

        print(
            "RESULTADO FINAL: INCOMPLETO"
        )

        return 1

    expected_valid_articles = (
        skip_articles
        + target_articles
    )

    if (
        result.valid_articles
        != expected_valid_articles
    ):

        print(
            "RESULTADO FINAL: "
            "SECUENCIA DE SELECCIÓN INCONSISTENTE"
        )

        return 1

    print(
        "RESULTADO FINAL: "
        "CONJUNTO HELD-OUT COMPLETADO"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )

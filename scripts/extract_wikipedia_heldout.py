from __future__ import annotations

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

TARGET_ARTICLES = 1_000


HELDOUT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "heldout"
    / "wikipedia_es_20260801_segment2"
)

DUMP_PATH = (
    HELDOUT_DIRECTORY
    / "dump"
    / SEGMENT_FILENAME
)

ARTICLE_DIRECTORY = (
    HELDOUT_DIRECTORY
    / "articles"
)

MANIFEST_PATH = (
    HELDOUT_DIRECTORY
    / "manifest.jsonl"
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
        "Objetivo               : "
        f"{TARGET_ARTICLES:,} artículos"
    )

    print(
        "Directorio artículos   : "
        f"{ARTICLE_DIRECTORY}"
    )

    print(
        "Manifiesto             : "
        f"{MANIFEST_PATH}"
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
                ARTICLE_DIRECTORY
            ),
            manifest_path=(
                MANIFEST_PATH
            ),
            snapshot_date=(
                SNAPSHOT_DATE
            ),
            target_articles=(
                TARGET_ARTICLES
            ),
        )
    )

    timer_start = (
        perf_counter()
    )

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
        "Directorio artículos     : "
        f"{ARTICLE_DIRECTORY}"
    )

    print(
        "Manifiesto               : "
        f"{MANIFEST_PATH}"
    )

    print("-" * 88)

    if (
        result.extracted
        != TARGET_ARTICLES
    ):

        print(
            "RESULTADO FINAL: INCOMPLETO"
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
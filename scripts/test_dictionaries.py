from __future__ import annotations

import re
import sys
from pathlib import Path


# =============================================================================
# CONFIGURACIÓN DEL PROYECTO
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
INPUT_PATH = PROJECT_ROOT / "docs" / "input"

sys.path.insert(
    0,
    str(SRC_PATH),
)


# =============================================================================
# IMPORTS DEL PROYECTO
# =============================================================================

from leximapsp16.dictionary import (  # noqa: E402
    LexiMapDictionary,
    LexiMapDictionaryError,
)


# =============================================================================
# PATRÓN DE DICCIONARIOS
#
# Ejemplos válidos:
#
# LexiCorpus_20000_Completo.csv
# LexiCorpus_65000_Wikipedia.csv
# LexiCorpus_30000_Literatura_Clasica.csv
#
# Quedan fuera automáticamente archivos analíticos como:
#
# LexiCorpus_Resultados.csv
# LexiCorpus_Cobertura_Cruzada.csv
# =============================================================================

DICTIONARY_FILENAME_PATTERN = re.compile(
    r"^LexiCorpus_(\d+)_(.+)\.csv$",
    re.IGNORECASE,
)


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def is_dictionary_file(path: Path) -> bool:
    return bool(
        DICTIONARY_FILENAME_PATTERN.fullmatch(
            path.name
        )
    )


def get_requested_size(path: Path) -> int | None:

    match = DICTIONARY_FILENAME_PATTERN.fullmatch(
        path.name
    )

    if match is None:
        return None

    return int(
        match.group(1)
    )


# =============================================================================
# PRUEBA
# =============================================================================

def main() -> int:

    print("=" * 78)
    print("VALIDACIÓN DE DICCIONARIOS LEXIMAPSP-16")
    print("=" * 78)

    print()
    print(f"Directorio: {INPUT_PATH}")

    if not INPUT_PATH.exists():
        print()
        print("ERROR: No existe el directorio de entrada.")
        return 1

    all_csv_files = sorted(
        INPUT_PATH.glob("*.csv")
    )

    dictionary_files = [
        path
        for path in all_csv_files
        if is_dictionary_file(path)
    ]

    ignored_files = [
        path
        for path in all_csv_files
        if not is_dictionary_file(path)
    ]

    if not dictionary_files:
        print()
        print(
            "ERROR: No se encontraron archivos "
            "de diccionario LexiCorpus."
        )
        return 1

    print(
        f"CSV encontrados             : "
        f"{len(all_csv_files):,}"
    )

    print(
        f"Diccionarios identificados  : "
        f"{len(dictionary_files):,}"
    )

    print(
        f"CSV analíticos ignorados     : "
        f"{len(ignored_files):,}"
    )

    if ignored_files:
        print()

        print(
            "Archivos ignorados:"
        )

        for path in ignored_files:
            print(
                f"  - {path.name}"
            )

    print()
    print("-" * 78)
    print()

    valid_files = 0
    invalid_files = 0

    for index, csv_path in enumerate(
        dictionary_files,
        start=1,
    ):

        requested_size = get_requested_size(
            csv_path
        )

        print(
            f"[{index:02d}/{len(dictionary_files):02d}] "
            f"{csv_path.name}"
        )

        try:
            dictionary = LexiMapDictionary.from_csv(
                csv_path
            )

            actual_size = dictionary.size

            if (
                requested_size is not None
                and actual_size > requested_size
            ):
                raise LexiMapDictionaryError(
                    "El diccionario contiene más entradas "
                    "que el tamaño solicitado en el nombre "
                    f"del archivo: solicitado={requested_size:,}, "
                    f"real={actual_size:,}."
                )

            size_status = ""

            if (
                requested_size is not None
                and actual_size < requested_size
            ):
                size_status = (
                    f" | vocabulario disponible="
                    f"{actual_size:,}"
                )

            print(
                "     OK"
                f" | entradas={actual_size:,}"
                f" | tokens="
                f"{dictionary.first_token:,}-"
                f"{dictionary.last_token:,}"
                f"{size_status}"
            )

            valid_files += 1

        except (
            LexiMapDictionaryError,
            FileNotFoundError,
            OSError,
            UnicodeError,
        ) as exc:

            print(
                f"     ERROR | {exc}"
            )

            invalid_files += 1

    print()
    print("=" * 78)
    print("RESUMEN")
    print("=" * 78)

    print(
        f"CSV de diccionario evaluados : "
        f"{len(dictionary_files):,}"
    )

    print(
        f"Válidos                      : "
        f"{valid_files:,}"
    )

    print(
        f"Inválidos                    : "
        f"{invalid_files:,}"
    )

    print(
        f"CSV analíticos ignorados      : "
        f"{len(ignored_files):,}"
    )

    print()

    if invalid_files == 0:

        print(
            "RESULTADO: TODOS LOS DICCIONARIOS "
            "SON COMPATIBLES CON LEXIMAPSP-16."
        )

        return 0

    print(
        "RESULTADO: EXISTEN DICCIONARIOS "
        "QUE REQUIEREN REVISIÓN."
    )

    return 1


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )
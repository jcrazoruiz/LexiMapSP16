from __future__ import annotations

import sys
from pathlib import Path


# =============================================================================
# CONFIGURACIÓN DEL PROYECTO
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_PATH),
)


# =============================================================================
# IMPORTS
# =============================================================================

from leximapsp16.dictionary import LexiMapDictionary  # noqa: E402
from leximapsp16.encoder import LexiMapEncoder  # noqa: E402


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DICTIONARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "input"
    / "LexiCorpus_65000_Completo.csv"
)


# =============================================================================
# CASOS DE PRUEBA
#
# Conservamos los mismos casos utilizados durante las etapas anteriores.
# Esto permite seguir la evolución:
#
# tokenizer -> representation -> encoder
# =============================================================================

TEST_CASES = [
    "Hola mundo.",
    "¿Eres Mexicanico?",
    "México, España y Perú.",
    "TODO está BIEN.",
    "A e o u y.",
    "2026",
    "Hola   mundo",
    "Primera línea\nSegunda línea",
    "Uno\tDos",
    "«Hola», dijo él —y continuó…",
]


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def visible_text(text: str) -> str:
    """
    Hace visibles tabuladores y saltos de línea.
    """

    return (
        text
        .replace("\t", "\\t")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def format_hex(
    data: bytes,
) -> str:
    """
    Representa una secuencia de bytes en hexadecimal.

    Ejemplo:

        b"\\x00\\x02\\x02\\x18"

    se muestra como:

        00 02 02 18
    """

    return " ".join(
        f"{byte:02X}"
        for byte in data
    )


def print_statistics(
    statistics,
) -> None:

    print()
    print("Estadísticas:")

    print(
        f"  Caracteres fuente          : "
        f"{statistics.source_characters:,}"
    )

    print(
        f"  Bytes UTF-8                : "
        f"{statistics.source_utf8_bytes:,}"
    )

    print(
        f"  Bytes LexiMapSp-16         : "
        f"{statistics.encoded_bytes:,}"
    )

    print(
        f"  Diferencia                 : "
        f"{statistics.reduction_bytes:+,} bytes"
    )

    print(
        f"  Reducción                  : "
        f"{statistics.reduction_percentage:.2f}%"
    )

    print()

    print(
        f"  Elementos léxicos          : "
        f"{statistics.lexical_items:,}"
    )

    print(
        f"  Elementos directos         : "
        f"{statistics.direct_items:,}"
    )

    print(
        f"  Elementos literales        : "
        f"{statistics.literal_items:,}"
    )

    print(
        f"  Modificadores              : "
        f"{statistics.modifier_items:,}"
    )

    print(
        f"  Espacios implícitos        : "
        f"{statistics.implicit_spaces:,}"
    )

    print(
        f"  Whitespace explícito       : "
        f"{statistics.explicit_whitespace_items:,}"
    )

    print(
        f"  Payload literal UTF-8      : "
        f"{statistics.literal_payload_bytes:,} bytes"
    )


# =============================================================================
# PRUEBA
# =============================================================================

def main() -> int:

    print("=" * 78)
    print("PRUEBA DEL ENCODER LEXIMAPSP-16")
    print("=" * 78)

    print()
    print(
        f"Diccionario : {DICTIONARY_PATH.name}"
    )

    if not DICTIONARY_PATH.exists():

        print()
        print(
            "ERROR: No se encontró el diccionario."
        )

        print(
            f"Ruta esperada: {DICTIONARY_PATH}"
        )

        return 1

    # =========================================================================
    # CARGA DEL DICCIONARIO
    # =========================================================================

    dictionary = LexiMapDictionary.from_csv(
        DICTIONARY_PATH
    )

    print(
        f"Entradas    : {dictionary.size:,}"
    )

    print(
        f"Tokens      : "
        f"{dictionary.first_token:,}-"
        f"{dictionary.last_token:,}"
    )

    # =========================================================================
    # ENCODER
    # =========================================================================

    encoder = LexiMapEncoder(
        dictionary
    )

    # =========================================================================
    # CASOS
    # =========================================================================

    total_utf8_bytes = 0
    total_encoded_bytes = 0

    for case_number, text in enumerate(
        TEST_CASES,
        start=1,
    ):

        print()
        print("-" * 78)
        print(
            f"CASO {case_number}"
        )
        print("-" * 78)

        print(
            f"Texto: {visible_text(text)}"
        )

        # ---------------------------------------------------------------------
        # UTF-8 ORIGINAL
        # ---------------------------------------------------------------------

        utf8_data = text.encode(
            "utf-8"
        )

        # ---------------------------------------------------------------------
        # LEXIMAPSP-16
        # ---------------------------------------------------------------------

        result = encoder.encode_with_statistics(
            text
        )

        encoded_data = result.data
        statistics = result.statistics

        # ---------------------------------------------------------------------
        # ACUMULADOS
        # ---------------------------------------------------------------------

        total_utf8_bytes += len(
            utf8_data
        )

        total_encoded_bytes += len(
            encoded_data
        )

        # ---------------------------------------------------------------------
        # INFORMACIÓN BINARIA
        # ---------------------------------------------------------------------

        print()

        print(
            f"UTF-8 HEX:"
        )

        print(
            f"  {format_hex(utf8_data)}"
        )

        print()

        print(
            "LexiMapSp-16 HEX:"
        )

        print(
            f"  {format_hex(encoded_data)}"
        )

        # ---------------------------------------------------------------------
        # ESTADÍSTICAS
        # ---------------------------------------------------------------------

        print_statistics(
            statistics
        )

    # =========================================================================
    # RESUMEN GLOBAL
    # =========================================================================

    print()
    print("=" * 78)
    print("RESUMEN GLOBAL")
    print("=" * 78)

    total_difference = (
        total_utf8_bytes
        - total_encoded_bytes
    )

    if total_utf8_bytes:

        total_reduction = (
            total_difference
            / total_utf8_bytes
            * 100.0
        )

    else:

        total_reduction = 0.0

    print()

    print(
        f"Casos evaluados          : "
        f"{len(TEST_CASES):,}"
    )

    print(
        f"Bytes UTF-8              : "
        f"{total_utf8_bytes:,}"
    )

    print(
        f"Bytes LexiMapSp-16       : "
        f"{total_encoded_bytes:,}"
    )

    print(
        f"Diferencia               : "
        f"{total_difference:+,} bytes"
    )

    print(
        f"Reducción global         : "
        f"{total_reduction:.2f}%"
    )

    print()

    print(
        "IMPORTANTE:"
    )

    print(
        "Estos valores corresponden únicamente a casos funcionales "
        "controlados."
    )

    print(
        "Todavía NO constituyen el benchmark experimental de "
        "LexiMapSp-16."
    )

    print()
    print("=" * 78)
    print("RESULTADO")
    print("=" * 78)

    print(
        "CODIFICACIÓN COMPLETADA SIN ERRORES."
    )

    return 0


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )
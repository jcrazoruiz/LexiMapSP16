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
# PARES DE PRUEBA
#
# Cada par contiene dos textos que NO deben producir exactamente la misma
# salida binaria.
#
# El objetivo es comprobar que la nueva regla de espacios distingue:
#
#   - espacio antes de puntuación
#   - espacio después de puntuación
#   - espacio alrededor de signos especiales
#   - espacio entre palabras, que puede seguir siendo implícito
# =============================================================================

TEST_PAIRS = [
    (
        "hola mundo",
        "holamundo",
        "Espacio ordinario entre palabras",
    ),
    (
        "hola, mundo",
        "hola,mundo",
        "Espacio después de coma",
    ),
    (
        "hola ,mundo",
        "hola,mundo",
        "Espacio antes de coma",
    ),
    (
        "él —y",
        "él—y",
        "Espacio antes de raya",
    ),
    (
        "él— y",
        "él—y",
        "Espacio después de raya",
    ),
    (
        "¿ hola",
        "¿hola",
        "Espacio después de signo de apertura",
    ),
    (
        "hola ?",
        "hola?",
        "Espacio antes de interrogación",
    ),
    (
        "a b",
        "ab",
        "Espacio entre palabras de una letra",
    ),
]


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def format_hex(
    data: bytes,
) -> str:
    """
    Convierte bytes a representación hexadecimal legible.
    """

    return " ".join(
        f"{byte:02X}"
        for byte in data
    )


def encode_text(
    encoder: LexiMapEncoder,
    text: str,
):
    """
    Codifica un texto y devuelve el resultado completo.
    """

    return encoder.encode_with_statistics(
        text
    )


# =============================================================================
# PRUEBA
# =============================================================================

def main() -> int:

    print("=" * 78)
    print("PRUEBA DE COLISIONES DE ESPACIADO - LEXIMAPSP-16")
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

    dictionary = LexiMapDictionary.from_csv(
        DICTIONARY_PATH
    )

    encoder = LexiMapEncoder(
        dictionary
    )

    print(
        f"Entradas    : {dictionary.size:,}"
    )

    print(
        f"Tokens      : "
        f"{dictionary.first_token:,}-"
        f"{dictionary.last_token:,}"
    )

    collisions = 0

    print()

    for index, (
        text_a,
        text_b,
        description,
    ) in enumerate(
        TEST_PAIRS,
        start=1,
    ):

        result_a = encode_text(
            encoder,
            text_a,
        )

        result_b = encode_text(
            encoder,
            text_b,
        )

        data_a = result_a.data
        data_b = result_b.data

        collision = (
            data_a == data_b
        )

        if collision:
            collisions += 1

        print("-" * 78)
        print(
            f"CASO {index}: {description}"
        )
        print("-" * 78)

        print(
            f"A: {text_a!r}"
        )

        print(
            f"   HEX: {format_hex(data_a)}"
        )

        print(
            f"   Bytes: {len(data_a)}"
        )

        print(
            f"   Espacios implícitos: "
            f"{result_a.statistics.implicit_spaces}"
        )

        print(
            f"   Espacios simples explícitos: "
            f"{result_a.statistics.explicit_single_spaces}"
        )

        print()

        print(
            f"B: {text_b!r}"
        )

        print(
            f"   HEX: {format_hex(data_b)}"
        )

        print(
            f"   Bytes: {len(data_b)}"
        )

        print(
            f"   Espacios implícitos: "
            f"{result_b.statistics.implicit_spaces}"
        )

        print(
            f"   Espacios simples explícitos: "
            f"{result_b.statistics.explicit_single_spaces}"
        )

        print()

        if collision:

            print(
                "RESULTADO: COLISIÓN DETECTADA"
            )

        else:

            print(
                "RESULTADO: DIFERENTES"
            )

        print()

    print("=" * 78)
    print("RESUMEN")
    print("=" * 78)

    print()

    print(
        f"Pares evaluados : {len(TEST_PAIRS):,}"
    )

    print(
        f"Colisiones      : {collisions:,}"
    )

    print()

    if collisions == 0:

        print(
            "RESULTADO: NO SE DETECTARON COLISIONES "
            "EN LOS CASOS EVALUADOS."
        )

        return 0

    print(
        "RESULTADO: EXISTEN COLISIONES Y EL DISEÑO "
        "DEBE REVISARSE."
    )

    return 1


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )
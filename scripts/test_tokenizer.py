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

from leximapsp16.tokenizer import (  # noqa: E402
    LexiMapTokenizer,
    TokenType,
)


# =============================================================================
# CASOS DE PRUEBA
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
# UTILIDADES
# =============================================================================

def visible_text(text: str) -> str:
    """
    Hace visibles tabuladores y saltos de línea
    para facilitar la inspección.
    """

    return (
        text
        .replace("\t", "\\t")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


# =============================================================================
# PRUEBA
# =============================================================================

def main() -> int:

    tokenizer = LexiMapTokenizer()

    print("=" * 78)
    print("PRUEBA DEL TOKENIZADOR LEXIMAPSP-16")
    print("=" * 78)

    for case_number, text in enumerate(
        TEST_CASES,
        start=1,
    ):

        print()
        print("-" * 78)
        print(f"CASO {case_number}")
        print("-" * 78)

        print(
            f"Texto: {visible_text(text)}"
        )

        tokens = tokenizer.tokenize(
            text
        )

        tokenizer.validate(
            text,
            tokens,
        )

        print()

        for index, token in enumerate(
            tokens,
            start=1,
        ):

            token_text = visible_text(
                token.text
            )

            print(
                f"{index:02d} | "
                f"{token.type.value:<10} | "
                f"{token.start:>3}-{token.end:<3} | "
                f"{token_text!r}"
            )

        reconstructed = tokenizer.reconstruct(
            tokens
        )

        print()

        print(
            "Reconstrucción: "
            f"{visible_text(reconstructed)}"
        )

        print(
            "Coincidencia   : "
            f"{reconstructed == text}"
        )

        if reconstructed != text:
            print()
            print(
                "ERROR: La reconstrucción no coincide "
                "con el texto original."
            )

            return 1

    print()
    print("=" * 78)
    print("RESULTADO")
    print("=" * 78)

    print(
        "TODOS LOS CASOS FUERON TOKENIZADOS "
        "Y RECONSTRUIDOS CORRECTAMENTE."
    )

    return 0


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )
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
from leximapsp16.tokenizer import LexiMapTokenizer  # noqa: E402
from leximapsp16.representation import (  # noqa: E402
    CaseModifier,
    LexiMapRepresentationResolver,
    RepresentationType,
)


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

DICTIONARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "input"
    / "LexiCorpus_65000_Completo.csv"
)

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
    return (
        text
        .replace("\t", "\\t")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def describe_representation(rep) -> str:

    rep_type = rep.representation_type

    if rep_type == RepresentationType.LEXICAL:
        result = (
            f"LEXICAL token={rep.lexical_token}"
        )

    elif rep_type == RepresentationType.DIRECT:
        result = (
            f"DIRECT code={rep.direct_code}"
        )

    elif rep_type == RepresentationType.LITERAL:
        result = (
            f"LITERAL {rep.literal_text!r}"
        )

    elif rep_type == RepresentationType.IMPLICIT_SPACE:
        result = "IMPLICIT_SPACE"

    elif rep_type == RepresentationType.EXPLICIT_WHITESPACE:
        result = (
            "EXPLICIT_WHITESPACE "
            f"{visible_text(rep.literal_text or '')!r}"
        )

    else:
        result = str(rep_type)

    if rep.modifier != CaseModifier.NONE:
        result += (
            f" + {rep.modifier.value}"
        )

    return result


# =============================================================================
# PRUEBA
# =============================================================================

def main() -> int:

    print("=" * 78)
    print("PRUEBA DE REPRESENTACIÓN LEXIMAPSP-16")
    print("=" * 78)

    print()
    print(f"Diccionario: {DICTIONARY_PATH.name}")

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

    tokenizer = LexiMapTokenizer()

    resolver = LexiMapRepresentationResolver(
        dictionary
    )

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

        representations = resolver.resolve_all(
            tokens
        )

        print()

        for index, representation in enumerate(
            representations,
            start=1,
        ):

            source_text = visible_text(
                representation.source.text
            )

            print(
                f"{index:02d} | "
                f"{source_text!r:<20} | "
                f"{describe_representation(representation)}"
            )

    print()
    print("=" * 78)
    print("RESULTADO")
    print("=" * 78)

    print(
        "REPRESENTACIÓN COMPLETADA SIN ERRORES."
    )

    return 0


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )
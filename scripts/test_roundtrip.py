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

from leximapsp16.decoder import LexiMapDecoder  # noqa: E402
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
    "hola, mundo",
    "hola,mundo",
    "hola ,mundo",
    "él —y",
    "él—y",
    "él— y",
    "¿ hola",
    "¿hola",
    "hola ?",
    "hola?",
    "a b",
    "ab",
]


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def visible_text(
    text: str,
) -> str:
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
    """

    return " ".join(
        f"{byte:02X}"
        for byte in data
    )


# =============================================================================
# PRUEBA
# =============================================================================

def main() -> int:

    print("=" * 78)
    print("PRUEBA ROUND-TRIP LEXIMAPSP-16")
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

    decoder = LexiMapDecoder(
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

    print()

    successful = 0
    failed = 0

    total_utf8_bytes = 0
    total_encoded_bytes = 0

    for case_number, original_text in enumerate(
        TEST_CASES,
        start=1,
    ):

        print("-" * 78)
        print(
            f"CASO {case_number}"
        )
        print("-" * 78)

        try:

            # -----------------------------------------------------------------
            # ENCODE
            # -----------------------------------------------------------------

            encoding_result = (
                encoder.encode_with_statistics(
                    original_text
                )
            )

            encoded_data = (
                encoding_result.data
            )

            # -----------------------------------------------------------------
            # DECODE
            # -----------------------------------------------------------------

            decoding_result = (
                decoder.decode_with_statistics(
                    encoded_data
                )
            )

            reconstructed_text = (
                decoding_result.text
            )

            # -----------------------------------------------------------------
            # COMPARACIÓN EXACTA
            # -----------------------------------------------------------------

            is_equal = (
                reconstructed_text
                == original_text
            )

            total_utf8_bytes += len(
                original_text.encode(
                    "utf-8"
                )
            )

            total_encoded_bytes += len(
                encoded_data
            )

            print(
                "Original     : "
                f"{visible_text(original_text)!r}"
            )

            print(
                "Reconstruido : "
                f"{visible_text(reconstructed_text)!r}"
            )

            print()

            print(
                "HEX          : "
                f"{format_hex(encoded_data)}"
            )

            print(
                f"UTF-8 bytes  : "
                f"{len(original_text.encode('utf-8'))}"
            )

            print(
                f"LexiMap bytes: "
                f"{len(encoded_data)}"
            )

            print()

            print(
                "Espacios implícitos codificados     : "
                f"{encoding_result.statistics.implicit_spaces}"
            )

            print(
                "Espacios implícitos reconstruidos   : "
                f"{decoding_result.statistics.reconstructed_implicit_spaces}"
            )

            print(
                "Espacios simples explícitos         : "
                f"{encoding_result.statistics.explicit_single_spaces}"
            )

            print()

            if is_equal:

                successful += 1

                print(
                    "RESULTADO: OK"
                )

            else:

                failed += 1

                print(
                    "RESULTADO: ERROR DE ROUND-TRIP"
                )

                print()

                print(
                    f"Longitud original     : "
                    f"{len(original_text)}"
                )

                print(
                    f"Longitud reconstruida : "
                    f"{len(reconstructed_text)}"
                )

                max_length = max(
                    len(original_text),
                    len(reconstructed_text),
                )

                for index in range(
                    max_length
                ):

                    original_character = (
                        original_text[index]
                        if index < len(original_text)
                        else None
                    )

                    reconstructed_character = (
                        reconstructed_text[index]
                        if index < len(reconstructed_text)
                        else None
                    )

                    if (
                        original_character
                        != reconstructed_character
                    ):

                        print(
                            "Primera diferencia en posición "
                            f"{index}:"
                        )

                        print(
                            "  Original     : "
                            f"{original_character!r}"
                        )

                        print(
                            "  Reconstruido : "
                            f"{reconstructed_character!r}"
                        )

                        break

        except Exception as error:

            failed += 1

            print(
                "RESULTADO: EXCEPCIÓN"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

        print()

    # =========================================================================
    # RESUMEN
    # =========================================================================

    print("=" * 78)
    print("RESUMEN")
    print("=" * 78)

    print()

    print(
        f"Casos evaluados : "
        f"{len(TEST_CASES):,}"
    )

    print(
        f"Correctos       : "
        f"{successful:,}"
    )

    print(
        f"Fallidos        : "
        f"{failed:,}"
    )

    print()

    print(
        f"Bytes UTF-8     : "
        f"{total_utf8_bytes:,}"
    )

    print(
        f"Bytes LexiMap   : "
        f"{total_encoded_bytes:,}"
    )

    if total_utf8_bytes:

        difference = (
            total_utf8_bytes
            - total_encoded_bytes
        )

        reduction_percentage = (
            difference
            / total_utf8_bytes
            * 100.0
        )

    else:

        difference = 0
        reduction_percentage = 0.0

    print(
        f"Diferencia      : "
        f"{difference:+,} bytes"
    )

    print(
        f"Reducción       : "
        f"{reduction_percentage:.2f}%"
    )

    print()

    if failed == 0:

        print(
            "RESULTADO FINAL: ROUND-TRIP CORRECTO EN TODOS LOS CASOS."
        )

        print()

        print(
            "Se cumple:"
        )

        print(
            "decode(encode(text)) == text"
        )

        return 0

    print(
        "RESULTADO FINAL: EXISTEN ERRORES DE ROUND-TRIP."
    )

    return 1


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )
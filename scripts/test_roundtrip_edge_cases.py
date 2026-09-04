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
# CASOS ADVERSARIALES
#
# El objetivo NO es medir compresión.
#
# El objetivo es buscar fallos de reversibilidad en:
#
#   - texto vacío
#   - espacios al inicio/final
#   - whitespace complejo
#   - CR, LF y CRLF
#   - OOV
#   - Unicode fuera de ISO-8859-1
#   - emojis
#   - caracteres combinantes
#   - capitalización irregular
#   - puntuación consecutiva
#   - literales consecutivos
#   - números y símbolos
#   - secuencias que puedan confundirse con controles
# =============================================================================

TEST_CASES = [
    # -------------------------------------------------------------------------
    # VACÍO
    # -------------------------------------------------------------------------
    (
        "Texto vacío",
        "",
    ),

    # -------------------------------------------------------------------------
    # ESPACIADO
    # -------------------------------------------------------------------------
    (
        "Un espacio",
        " ",
    ),
    (
        "Dos espacios",
        "  ",
    ),
    (
        "Tres espacios",
        "   ",
    ),
    (
        "Espacio inicial",
        " hola",
    ),
    (
        "Espacio final",
        "hola ",
    ),
    (
        "Espacio inicial y final",
        " hola ",
    ),
    (
        "Múltiples espacios entre palabras",
        "hola     mundo",
    ),
    (
        "Espacios alrededor de coma",
        "hola , mundo",
    ),
    (
        "Espacios alrededor de raya",
        "él — y",
    ),

    # -------------------------------------------------------------------------
    # TAB / LF / CR / CRLF
    # -------------------------------------------------------------------------
    (
        "TAB inicial",
        "\thola",
    ),
    (
        "TAB final",
        "hola\t",
    ),
    (
        "Dos TAB",
        "hola\t\tmundo",
    ),
    (
        "LF inicial",
        "\nhola",
    ),
    (
        "LF final",
        "hola\n",
    ),
    (
        "Dos LF",
        "hola\n\nmundo",
    ),
    (
        "CR",
        "hola\rmundo",
    ),
    (
        "CRLF",
        "hola\r\nmundo",
    ),
    (
        "Dos CRLF",
        "hola\r\n\r\nmundo",
    ),
    (
        "Whitespace combinado",
        " \t hola \r\n mundo \n ",
    ),

    # -------------------------------------------------------------------------
    # OOV SIMPLES
    # -------------------------------------------------------------------------
    (
        "OOV minúscula",
        "mexicanico",
    ),
    (
        "OOV capitalizada",
        "Mexicanico",
    ),
    (
        "OOV mayúsculas",
        "MEXICANICO",
    ),
    (
        "Dos OOV separados",
        "mexicanico palabranueva",
    ),
    (
        "OOV junto a palabra conocida",
        "hola mexicanico mundo",
    ),
    (
        "OOV con puntuación",
        "¿mexicanico?",
    ),

    # -------------------------------------------------------------------------
    # CAPITALIZACIÓN IRREGULAR
    # -------------------------------------------------------------------------
    (
        "Capitalización mixta",
        "mExIcO",
    ),
    (
        "CamelCase",
        "LexiMapSp",
    ),
    (
        "Mayúscula interna",
        "meXico",
    ),

    # -------------------------------------------------------------------------
    # UNICODE
    # -------------------------------------------------------------------------
    (
        "Emoji",
        "hola 😀 mundo",
    ),
    (
        "Dos emojis",
        "😀😃",
    ),
    (
        "Emoji entre palabras",
        "hola😀mundo",
    ),
    (
        "Símbolos Unicode",
        "α β γ",
    ),
    (
        "Cirílico",
        "Привет",
    ),
    (
        "Japonés",
        "こんにちは",
    ),
    (
        "Chino",
        "你好世界",
    ),
    (
        "Árabe",
        "مرحبا",
    ),

    # -------------------------------------------------------------------------
    # CARACTERES COMBINANTES
    #
    # Primer ejemplo:
    #   "é" precompuesto
    #
    # Segundo:
    #   "e" + U+0301
    #
    # Deben preservarse exactamente como fueron recibidos.
    # -------------------------------------------------------------------------
    (
        "Unicode precompuesto",
        "café",
    ),
    (
        "Unicode combinante",
        "cafe\u0301",
    ),
    (
        "Combinante aislado",
        "\u0301",
    ),

    # -------------------------------------------------------------------------
    # TIPOGRAFÍA
    # -------------------------------------------------------------------------
    (
        "Comillas curvas",
        "“Hola”",
    ),
    (
        "Comillas simples curvas",
        "‘Hola’",
    ),
    (
        "Elipsis",
        "Hola…",
    ),
    (
        "Raya",
        "hola—mundo",
    ),
    (
        "En dash",
        "2025–2026",
    ),
    (
        "Bullet",
        "• elemento",
    ),
    (
        "Euro",
        "€100",
    ),
    (
        "Trademark",
        "LexiMap™",
    ),

    # -------------------------------------------------------------------------
    # PUNTUACIÓN
    # -------------------------------------------------------------------------
    (
        "Puntuación consecutiva",
        "Hola!!!",
    ),
    (
        "Interrogación y admiración",
        "¿¡Hola!?",
    ),
    (
        "Paréntesis",
        "(hola)",
    ),
    (
        "Corchetes",
        "[hola]",
    ),
    (
        "Llaves",
        "{hola}",
    ),
    (
        "Comillas ASCII",
        '"hola"',
    ),
    (
        "Apóstrofo",
        "l'amour",
    ),
    (
        "Slash",
        "hola/mundo",
    ),
    (
        "Backslash",
        "hola\\mundo",
    ),

    # -------------------------------------------------------------------------
    # NÚMEROS Y SÍMBOLOS
    # -------------------------------------------------------------------------
    (
        "Número entero",
        "1234567890",
    ),
    (
        "Número decimal",
        "123.45",
    ),
    (
        "Porcentaje",
        "99%",
    ),
    (
        "Moneda",
        "$100",
    ),
    (
        "Correo simplificado",
        "a@b.com",
    ),
    (
        "URL simplificada",
        "https://ejemplo.com",
    ),

    # -------------------------------------------------------------------------
    # PALABRAS DIRECTAS DE UNA LETRA
    # -------------------------------------------------------------------------
    (
        "Palabras de una letra",
        "a e o u y",
    ),
    (
        "Palabras de una letra mayúsculas",
        "A E O U Y",
    ),
    (
        "Una letra junto a puntuación",
        "a,y",
    ),
    (
        "Una letra con espacio y puntuación",
        "a, y",
    ),

    # -------------------------------------------------------------------------
    # SECUENCIAS COMPLEJAS
    # -------------------------------------------------------------------------
    (
        "Frase compleja 1",
        "¿Hola, mundo? Sí: todo está bien.",
    ),
    (
        "Frase compleja 2",
        "«Hola» —dijo él—. Después continuó…",
    ),
    (
        "Frase compleja 3",
        "Uno\tDos\nTres\r\nCuatro",
    ),
    (
        "Frase compleja 4",
        "Inicio 😀 mexicanico — fin.",
    ),
]


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def visible_text(
    text: str,
) -> str:
    """
    Hace visibles los principales caracteres de control.
    """

    return (
        text
        .replace("\\", "\\\\")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def format_hex(
    data: bytes,
) -> str:
    """
    Convierte bytes a hexadecimal legible.
    """

    return " ".join(
        f"{byte:02X}"
        for byte in data
    )


def find_first_difference(
    original: str,
    reconstructed: str,
) -> tuple[int, str | None, str | None] | None:
    """
    Devuelve la primera posición diferente.
    """

    max_length = max(
        len(original),
        len(reconstructed),
    )

    for index in range(
        max_length
    ):

        original_character = (
            original[index]
            if index < len(original)
            else None
        )

        reconstructed_character = (
            reconstructed[index]
            if index < len(reconstructed)
            else None
        )

        if (
            original_character
            != reconstructed_character
        ):
            return (
                index,
                original_character,
                reconstructed_character,
            )

    return None


# =============================================================================
# PRUEBA
# =============================================================================

def main() -> int:

    print("=" * 78)
    print("PRUEBA ADVERSARIAL ROUND-TRIP - LEXIMAPSP-16")
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

    for case_number, (
        description,
        original_text,
    ) in enumerate(
        TEST_CASES,
        start=1,
    ):

        print("-" * 78)

        print(
            f"CASO {case_number}: {description}"
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

            original_utf8_bytes = len(
                original_text.encode(
                    "utf-8"
                )
            )

            encoded_bytes = len(
                encoded_data
            )

            total_utf8_bytes += (
                original_utf8_bytes
            )

            total_encoded_bytes += (
                encoded_bytes
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
                f"UTF-8 bytes  : "
                f"{original_utf8_bytes}"
            )

            print(
                f"LexiMap bytes: "
                f"{encoded_bytes}"
            )

            print()

            print(
                "Implícitos codificados   : "
                f"{encoding_result.statistics.implicit_spaces}"
            )

            print(
                "Implícitos reconstruidos : "
                f"{decoding_result.statistics.reconstructed_implicit_spaces}"
            )

            print(
                "Espacios explícitos      : "
                f"{encoding_result.statistics.explicit_single_spaces}"
            )

            print(
                "Literales                 : "
                f"{encoding_result.statistics.literal_items}"
            )

            print(
                "Payload literal UTF-8     : "
                f"{encoding_result.statistics.literal_payload_bytes}"
            )

            if is_equal:

                successful += 1

                print()

                print(
                    "RESULTADO: OK"
                )

            else:

                failed += 1

                print()

                print(
                    "RESULTADO: ERROR DE ROUND-TRIP"
                )

                difference = find_first_difference(
                    original_text,
                    reconstructed_text,
                )

                if difference is not None:

                    (
                        difference_position,
                        original_character,
                        reconstructed_character,
                    ) = difference

                    print()

                    print(
                        "Primera diferencia:"
                    )

                    print(
                        f"  Posición      : "
                        f"{difference_position}"
                    )

                    print(
                        f"  Original      : "
                        f"{original_character!r}"
                    )

                    print(
                        f"  Reconstruido  : "
                        f"{reconstructed_character!r}"
                    )

                print()

                print(
                    "HEX LexiMap:"
                )

                print(
                    f"  {format_hex(encoded_data)}"
                )

        except Exception as error:

            failed += 1

            print(
                "RESULTADO: EXCEPCIÓN"
            )

            print()

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
            "RESULTADO FINAL: TODOS LOS CASOS ADVERSARIALES "
            "SON REVERSIBLES."
        )

        print()

        print(
            "Se cumple para todos los casos:"
        )

        print(
            "decode(encode(text)) == text"
        )

        return 0

    print(
        "RESULTADO FINAL: SE DETECTARON CASOS NO REVERSIBLES."
    )

    print()

    print(
        "No debe avanzarse al benchmark hasta analizar "
        "y corregir los fallos."
    )

    return 1


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )
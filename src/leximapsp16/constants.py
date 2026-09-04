from __future__ import annotations


# =============================================================================
# IDENTIDAD DEL FORMATO
# =============================================================================

FORMAT_NAME = "LexiMapSp-16"
FORMAT_VERSION = "0.1.0"


# =============================================================================
# ESPACIO DE CODIFICACIÓN
# =============================================================================

MIN_CODE = 0
MAX_CODE = 65_535

CODE_UNIT_BITS = 16
CODE_UNIT_BYTES = 2

BYTE_ORDER = "big"


# =============================================================================
# RANGOS PRINCIPALES
# =============================================================================

DIRECT_CODE_START = 0
DIRECT_CODE_END = 255

RESERVED_EXTENSION_START = 256
RESERVED_EXTENSION_END = 535

LEXICAL_TOKEN_START = 536
LEXICAL_TOKEN_END = 65_535

LEXICAL_TOKEN_COUNT = (
    LEXICAL_TOKEN_END
    - LEXICAL_TOKEN_START
    + 1
)

EXPECTED_LEXICAL_TOKEN_COUNT = 65_000


# =============================================================================
# CÓDIGOS DE CONTROL
# =============================================================================

ISO_START = 0
ISO_END = 1

CAPITALIZED = 2
UPPERCASE = 3


# =============================================================================
# CONTROLES DE TEXTO
# =============================================================================

TAB = 9
LF = 10
CR = 13


# =============================================================================
# ASCII
# =============================================================================

ASCII_START = 0
ASCII_END = 127

SPACE = 32

ASCII_PRINTABLE_START = 32
ASCII_PRINTABLE_END = 126


# =============================================================================
# EXTENSIÓN TIPOGRÁFICA MODERNA
#
# LexiMapSp-16 reutiliza selectivamente posiciones del rango 128-159,
# equivalentes a determinadas posiciones de Windows-1252.
#
# IMPORTANTE:
# Estos códigos NO forman parte de ISO-8859-1.
# Son una extensión propia de LexiMapSp-16.
# =============================================================================

TYPOGRAPHIC_EXTENSION = {
    128: "€",
    133: "…",
    137: "‰",
    139: "‹",
    145: "‘",
    146: "’",
    147: "“",
    148: "”",
    149: "•",
    150: "–",
    151: "—",
    153: "™",
    155: "›",
}

TYPOGRAPHIC_EXTENSION_REVERSE = {
    character: code
    for code, character in TYPOGRAPHIC_EXTENSION.items()
}


# =============================================================================
# ISO-8859-1
# =============================================================================

ISO_8859_1_START = 160
ISO_8859_1_END = 255


# =============================================================================
# PALABRAS DE UNA SOLA LETRA
#
# Estas palabras no necesitan ocupar una entrada dentro del diccionario
# léxico si pueden representarse directamente mediante su valor ASCII.
# =============================================================================

SINGLE_LETTER_WORDS = {
    "a",
    "e",
    "o",
    "u",
    "y",
}


# =============================================================================
# RANGO RESERVADO
#
# 256-535 permanece disponible para futuras extensiones del formato.
#
# En la versión actual NO se asignan mecanismos morfológicos de:
# - género
# - plural
# - género + plural
#
# Cada forma léxica permanece como entrada independiente del diccionario
# según su frecuencia.
# =============================================================================

RESERVED_CODES = range(
    RESERVED_EXTENSION_START,
    RESERVED_EXTENSION_END + 1,
)


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def is_valid_code(value: int) -> bool:
    return MIN_CODE <= value <= MAX_CODE


def is_direct_code(value: int) -> bool:
    return (
        DIRECT_CODE_START
        <= value
        <= DIRECT_CODE_END
    )


def is_reserved_extension(value: int) -> bool:
    return (
        RESERVED_EXTENSION_START
        <= value
        <= RESERVED_EXTENSION_END
    )


def is_lexical_token(value: int) -> bool:
    return (
        LEXICAL_TOKEN_START
        <= value
        <= LEXICAL_TOKEN_END
    )


def ranking_to_token(ranking: int) -> int:
    if not 1 <= ranking <= EXPECTED_LEXICAL_TOKEN_COUNT:
        raise ValueError(
            "El ranking debe estar entre "
            f"1 y {EXPECTED_LEXICAL_TOKEN_COUNT:,}."
        )

    return LEXICAL_TOKEN_START + ranking - 1


def token_to_ranking(token: int) -> int:
    if not is_lexical_token(token):
        raise ValueError(
            "El token debe estar dentro del rango léxico "
            f"{LEXICAL_TOKEN_START:,}-"
            f"{LEXICAL_TOKEN_END:,}."
        )

    return token - LEXICAL_TOKEN_START + 1


def code_to_bytes(value: int) -> bytes:
    if not is_valid_code(value):
        raise ValueError(
            f"El código {value} está fuera del rango de 16 bits."
        )

    return value.to_bytes(
        CODE_UNIT_BYTES,
        byteorder=BYTE_ORDER,
        signed=False,
    )


def bytes_to_code(data: bytes) -> int:
    if len(data) != CODE_UNIT_BYTES:
        raise ValueError(
            "Una unidad LexiMapSp-16 debe contener exactamente "
            f"{CODE_UNIT_BYTES} bytes."
        )

    return int.from_bytes(
        data,
        byteorder=BYTE_ORDER,
        signed=False,
    )


# =============================================================================
# VALIDACIÓN INTERNA DE LA ESPECIFICACIÓN
# =============================================================================

assert LEXICAL_TOKEN_COUNT == EXPECTED_LEXICAL_TOKEN_COUNT

assert ranking_to_token(1) == 536
assert ranking_to_token(65_000) == 65_535

assert token_to_ranking(536) == 1
assert token_to_ranking(65_535) == 65_000

assert code_to_bytes(536) == bytes.fromhex("0218")
assert code_to_bytes(65_535) == bytes.fromhex("FFFF")
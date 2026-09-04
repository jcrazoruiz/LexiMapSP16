from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .constants import (
    CAPITALIZED,
    UPPERCASE,
    TYPOGRAPHIC_EXTENSION_REVERSE,
    SINGLE_LETTER_WORDS,
)
from .dictionary import LexiMapDictionary
from .tokenizer import (
    TextToken,
    TokenType,
)


# =============================================================================
# TIPOS DE REPRESENTACIÓN
# =============================================================================

class RepresentationType(Enum):
    """
    Tipo de representación elegida para un token del texto.

    LEXICAL
        Palabra representada mediante token del diccionario.

    DIRECT
        Carácter representado directamente mediante código LexiMapSp-16.

    LITERAL
        Elemento que no puede representarse mediante los mecanismos actuales
        y deberá codificarse mediante bloque literal.

    IMPLICIT_SPACE
        Espacio simple " " que no ocupa bytes.

    EXPLICIT_WHITESPACE
        Whitespace distinto de " " que deberá codificarse explícitamente.
    """

    LEXICAL = "LEXICAL"
    DIRECT = "DIRECT"
    LITERAL = "LITERAL"
    IMPLICIT_SPACE = "IMPLICIT_SPACE"
    EXPLICIT_WHITESPACE = "EXPLICIT_WHITESPACE"


# =============================================================================
# MODIFICADORES
# =============================================================================

class CaseModifier(Enum):
    NONE = "NONE"
    CAPITALIZED = "CAPITALIZED"
    UPPERCASE = "UPPERCASE"


# =============================================================================
# REPRESENTACIÓN
# =============================================================================

@dataclass(frozen=True)
class TokenRepresentation:
    """
    Describe cómo será representado un token del texto.

    Todavía no contiene bytes finales. Esa responsabilidad corresponderá
    al encoder.
    """

    source: TextToken
    representation_type: RepresentationType

    lexical_token: int | None = None
    direct_code: int | None = None
    modifier: CaseModifier = CaseModifier.NONE

    normalized_word: str | None = None
    literal_text: str | None = None


# =============================================================================
# FUNCIONES AUXILIARES DE CAPITALIZACIÓN
# =============================================================================

def detect_case_modifier(
    word: str,
) -> tuple[str, CaseModifier]:
    """
    Determina la forma base que se buscará en el diccionario y el posible
    modificador de capitalización.

    Casos admitidos:

        hola  -> hola + NONE
        Hola  -> hola + CAPITALIZED
        HOLA  -> hola + UPPERCASE

    Cualquier patrón de capitalización más complejo permanece sin modificar
    y normalmente terminará como bloque literal si no existe exactamente
    en el diccionario.
    """

    if not word:
        return word, CaseModifier.NONE

    lower_word = word.lower()

    # -------------------------------------------------------------------------
    # Minúsculas
    # -------------------------------------------------------------------------

    if word == lower_word:
        return lower_word, CaseModifier.NONE

    # -------------------------------------------------------------------------
    # MAYÚSCULAS
    # -------------------------------------------------------------------------

    if word == word.upper():
        return lower_word, CaseModifier.UPPERCASE

    # -------------------------------------------------------------------------
    # Capitalización inicial
    #
    # Debe existir al menos una letra después de la primera posición para
    # distinguir adecuadamente palabras de una sola letra.
    # -------------------------------------------------------------------------

    if (
        len(word) >= 1
        and word[0] == word[0].upper()
        and word[1:] == word[1:].lower()
    ):
        return lower_word, CaseModifier.CAPITALIZED

    # -------------------------------------------------------------------------
    # Capitalización irregular
    # -------------------------------------------------------------------------

    return word, CaseModifier.NONE


# =============================================================================
# CÓDIGOS DIRECTOS
# =============================================================================

def get_direct_character_code(
    character: str,
) -> int | None:
    """
    Devuelve el código directo LexiMapSp-16 de un carácter cuando existe.

    Orden:

    1. ASCII 0-127
    2. extensión tipográfica propia 128-159
    3. ISO-8859-1 160-255

    Retorna None cuando el carácter no puede representarse directamente.
    """

    if len(character) != 1:
        return None

    code_point = ord(character)

    # -------------------------------------------------------------------------
    # ASCII
    # -------------------------------------------------------------------------

    if 0 <= code_point <= 127:
        return code_point

    # -------------------------------------------------------------------------
    # Extensión tipográfica LexiMapSp-16
    # -------------------------------------------------------------------------

    extension_code = TYPOGRAPHIC_EXTENSION_REVERSE.get(
        character
    )

    if extension_code is not None:
        return extension_code

    # -------------------------------------------------------------------------
    # ISO-8859-1
    # -------------------------------------------------------------------------

    if 160 <= code_point <= 255:
        return code_point

    return None


# =============================================================================
# CLASIFICADOR DE REPRESENTACIÓN
# =============================================================================

class LexiMapRepresentationResolver:
    """
    Decide cómo será representado cada token producido por el tokenizador.

    No genera bytes.
    """

    def __init__(
        self,
        dictionary: LexiMapDictionary,
    ) -> None:

        self.dictionary = dictionary


    # =========================================================================
    # INTERFAZ PRINCIPAL
    # =========================================================================

    def resolve(
        self,
        token: TextToken,
    ) -> TokenRepresentation:

        if token.type == TokenType.WORD:
            return self._resolve_word(
                token
            )

        if token.type == TokenType.WHITESPACE:
            return self._resolve_whitespace(
                token
            )

        if token.type == TokenType.CHARACTER:
            return self._resolve_character(
                token
            )

        raise ValueError(
            f"Tipo de token no soportado: {token.type}"
        )


    def resolve_all(
        self,
        tokens: list[TextToken],
    ) -> list[TokenRepresentation]:

        return [
            self.resolve(token)
            for token in tokens
        ]


    # =========================================================================
    # PALABRAS
    # =========================================================================

    def _resolve_word(
        self,
        token: TextToken,
    ) -> TokenRepresentation:

        word = token.text

        normalized_word, modifier = detect_case_modifier(
            word
        )

        # ---------------------------------------------------------------------
        # PALABRAS DE UNA SOLA LETRA
        # ---------------------------------------------------------------------

        lower_word = word.lower()

        if (
            len(lower_word) == 1
            and lower_word in SINGLE_LETTER_WORDS
        ):

            direct_code = ord(
                lower_word
            )

            return TokenRepresentation(
                source=token,
                representation_type=RepresentationType.DIRECT,
                direct_code=direct_code,
                modifier=modifier,
                normalized_word=lower_word,
            )

        # ---------------------------------------------------------------------
        # PALABRA EN DICCIONARIO
        # ---------------------------------------------------------------------

        lexical_token = self.dictionary.get_token(
            normalized_word
        )

        if lexical_token is not None:

            return TokenRepresentation(
                source=token,
                representation_type=RepresentationType.LEXICAL,
                lexical_token=lexical_token,
                modifier=modifier,
                normalized_word=normalized_word,
            )

        # ---------------------------------------------------------------------
        # OOV
        # ---------------------------------------------------------------------

        return TokenRepresentation(
            source=token,
            representation_type=RepresentationType.LITERAL,
            normalized_word=normalized_word,
            literal_text=word,
        )


    # =========================================================================
    # WHITESPACE
    # =========================================================================

    def _resolve_whitespace(
        self,
        token: TextToken,
    ) -> TokenRepresentation:

        # ---------------------------------------------------------------------
        # ESPACIO SIMPLE
        #
        # Regla LexiMapSp-16:
        #
        # " " -> implícito -> 0 bytes
        # ---------------------------------------------------------------------

        if token.text == " ":

            return TokenRepresentation(
                source=token,
                representation_type=RepresentationType.IMPLICIT_SPACE,
            )

        # ---------------------------------------------------------------------
        # WHITESPACE EXPLÍCITO
        # ---------------------------------------------------------------------

        return TokenRepresentation(
            source=token,
            representation_type=RepresentationType.EXPLICIT_WHITESPACE,
            literal_text=token.text,
        )


    # =========================================================================
    # CARACTERES
    # =========================================================================

    def _resolve_character(
        self,
        token: TextToken,
    ) -> TokenRepresentation:

        character = token.text

        direct_code = get_direct_character_code(
            character
        )

        if direct_code is not None:

            return TokenRepresentation(
                source=token,
                representation_type=RepresentationType.DIRECT,
                direct_code=direct_code,
            )

        return TokenRepresentation(
            source=token,
            representation_type=RepresentationType.LITERAL,
            literal_text=character,
        )


# =============================================================================
# FUNCIONES DE APOYO
# =============================================================================

def modifier_to_control_code(
    modifier: CaseModifier,
) -> int | None:
    """
    Traduce un modificador lógico al código de control LexiMapSp-16.
    """

    if modifier == CaseModifier.NONE:
        return None

    if modifier == CaseModifier.CAPITALIZED:
        return CAPITALIZED

    if modifier == CaseModifier.UPPERCASE:
        return UPPERCASE

    raise ValueError(
        f"Modificador no soportado: {modifier}"
    )
from __future__ import annotations

import unicodedata

from dataclasses import dataclass
from enum import Enum


# =============================================================================
# TIPOS DE TOKEN
# =============================================================================

class TokenType(Enum):
    """
    Tipos de elementos detectados dentro del texto fuente.

    WORD
        Secuencia continua de letras Unicode.

    WHITESPACE
        Uno o más caracteres de espacio en blanco.

    CHARACTER
        Carácter individual que no pertenece a una palabra
        ni es espacio en blanco.

        Incluye, entre otros:
        - puntuación
        - números
        - símbolos
        - operadores
        - caracteres tipográficos
    """

    WORD = "WORD"
    WHITESPACE = "WHITESPACE"
    CHARACTER = "CHARACTER"


# =============================================================================
# TOKEN
# =============================================================================

@dataclass(frozen=True)
class TextToken:
    """
    Unidad léxica producida por el tokenizador.

    Attributes
    ----------
    type:
        Tipo de token.

    text:
        Texto original exacto.

    start:
        Posición inicial dentro del texto fuente.

    end:
        Posición final exclusiva.
    """

    type: TokenType
    text: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


# =============================================================================
# FUNCIONES DE CLASIFICACIÓN UNICODE
# =============================================================================

def _unicode_category(character: str) -> str:
    return unicodedata.category(character)


def is_letter(character: str) -> bool:
    """
    Indica si un carácter pertenece a alguna categoría Unicode
    de letras.

    Ejemplos:

        a
        ñ
        Á
        ü
        é
    """

    return _unicode_category(character).startswith("L")


def is_combining_mark(character: str) -> bool:
    """
    Detecta marcas combinantes Unicode.

    Esto permite reconocer correctamente palabras escritas mediante
    secuencias Unicode descompuestas.

    Ejemplo:

        e + COMBINING ACUTE ACCENT
    """

    return _unicode_category(character).startswith("M")


def is_word_continuation(character: str) -> bool:
    """
    Un carácter puede continuar una palabra cuando es:

    - una letra Unicode
    - una marca combinante Unicode

    Los números, guiones, apóstrofos y signos de puntuación
    NO forman parte de la palabra.
    """

    return (
        is_letter(character)
        or is_combining_mark(character)
    )


# =============================================================================
# TOKENIZADOR
# =============================================================================

class LexiMapTokenizer:
    """
    Tokenizador determinista para LexiMapSp-16.

    Su responsabilidad consiste exclusivamente en separar el texto
    fuente sin modificarlo.

    No realiza:

    - búsqueda en diccionario
    - normalización
    - conversión a minúsculas
    - asignación de tokens LexiMapSp-16
    - generación de bloques literales
    - codificación binaria

    Esas operaciones corresponden a etapas posteriores.
    """

    def tokenize(
        self,
        text: str,
    ) -> list[TextToken]:

        if not isinstance(text, str):
            raise TypeError(
                "El texto de entrada debe ser una cadena."
            )

        if not text:
            return []

        tokens: list[TextToken] = []

        position = 0
        text_length = len(text)

        while position < text_length:

            character = text[position]

            # =================================================================
            # PALABRA
            # =================================================================

            if is_letter(character):

                start = position
                position += 1

                while (
                    position < text_length
                    and is_word_continuation(
                        text[position]
                    )
                ):
                    position += 1

                tokens.append(
                    TextToken(
                        type=TokenType.WORD,
                        text=text[start:position],
                        start=start,
                        end=position,
                    )
                )

                continue

            # =================================================================
            # ESPACIOS EN BLANCO
            # =================================================================

            if character.isspace():

                start = position
                position += 1

                while (
                    position < text_length
                    and text[position].isspace()
                ):
                    position += 1

                tokens.append(
                    TextToken(
                        type=TokenType.WHITESPACE,
                        text=text[start:position],
                        start=start,
                        end=position,
                    )
                )

                continue

            # =================================================================
            # RESTO DE CARACTERES
            # =================================================================

            tokens.append(
                TextToken(
                    type=TokenType.CHARACTER,
                    text=character,
                    start=position,
                    end=position + 1,
                )
            )

            position += 1

        return tokens


    # =========================================================================
    # RECONSTRUCCIÓN
    # =========================================================================

    @staticmethod
    def reconstruct(
        tokens: list[TextToken],
    ) -> str:
        """
        Reconstruye el texto original a partir de los tokens.

        Esta función permite comprobar que la tokenización es
        completamente reversible.
        """

        return "".join(
            token.text
            for token in tokens
        )


    # =========================================================================
    # VALIDACIÓN
    # =========================================================================

    @staticmethod
    def validate(
        text: str,
        tokens: list[TextToken],
    ) -> None:
        """
        Verifica que:

        1. ningún token esté vacío;
        2. las posiciones sean consecutivas;
        3. no existan huecos ni solapamientos;
        4. la reconstrucción sea idéntica al texto original.
        """

        expected_position = 0

        for index, token in enumerate(
            tokens,
            start=1,
        ):

            if not token.text:
                raise ValueError(
                    f"Token vacío en posición {index}."
                )

            if token.start != expected_position:
                raise ValueError(
                    "Secuencia de tokens no consecutiva. "
                    f"Se esperaba posición "
                    f"{expected_position} y se encontró "
                    f"{token.start}."
                )

            if token.end <= token.start:
                raise ValueError(
                    "Rango inválido en token "
                    f"{index}: "
                    f"{token.start}-{token.end}."
                )

            if (
                text[token.start:token.end]
                != token.text
            ):
                raise ValueError(
                    "El contenido del token no coincide "
                    "con el texto fuente en la posición "
                    f"{index}."
                )

            expected_position = token.end

        if expected_position != len(text):
            raise ValueError(
                "La tokenización no cubre completamente "
                "el texto fuente."
            )

        reconstructed = (
            LexiMapTokenizer.reconstruct(
                tokens
            )
        )

        if reconstructed != text:
            raise ValueError(
                "La reconstrucción no coincide "
                "con el texto original."
            )


# =============================================================================
# INTERFAZ SIMPLE
# =============================================================================

def tokenize(
    text: str,
) -> list[TextToken]:
    """
    Interfaz simplificada para tokenizar texto sin instanciar
    explícitamente LexiMapTokenizer.
    """

    tokenizer = LexiMapTokenizer()

    tokens = tokenizer.tokenize(
        text
    )

    tokenizer.validate(
        text,
        tokens,
    )

    return tokens
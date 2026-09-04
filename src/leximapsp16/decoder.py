from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from leximapsp16.constants import (
    CAPITALIZED,
    CR,
    ISO_END,
    ISO_START,
    LF,
    SPACE,
    TAB,
    TYPOGRAPHIC_EXTENSION,
    UPPERCASE,
    bytes_to_code,
    is_lexical_token,
    is_reserved_extension,
)
from leximapsp16.dictionary import LexiMapDictionary


class LexiMapDecodingError(Exception):
    """
    Error producido durante la decodificación LexiMapSp-16.
    """


class DecodedUnitType(Enum):
    """
    Tipo lógico de una unidad ya decodificada.
    """

    WORD = auto()
    CHARACTER = auto()
    WHITESPACE = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class DecodedUnit:
    """
    Unidad lógica obtenida a partir del flujo binario.
    """

    text: str
    unit_type: DecodedUnitType
    unambiguous_word: bool = False


@dataclass
class DecodingStatistics:
    """
    Estadísticas producidas durante la decodificación.
    """

    encoded_bytes: int = 0
    decoded_characters: int = 0
    decoded_utf8_bytes: int = 0

    lexical_items: int = 0
    direct_items: int = 0
    literal_items: int = 0
    modifier_items: int = 0

    explicit_whitespace_items: int = 0
    reconstructed_implicit_spaces: int = 0


@dataclass(frozen=True)
class DecodingResult:
    """
    Resultado de una operación de decodificación.
    """

    text: str
    statistics: DecodingStatistics


class LexiMapDecoder:
    """
    Decoder de LexiMapSp-16.

    Interpreta:

        00 00       inicio de literal UTF-8
        00 01       fin de literal
        00 02       CAPITALIZED
        00 03       UPPERCASE

        00 09       TAB
        00 0A       LF
        00 0D       CR
        00 20       SPACE explícito

        00 xx       caracteres directos

        02 18
          ...
        FF FF       tokens léxicos 536-65535

    Espacios implícitos
    -------------------

    El encoder omite un espacio simple exclusivamente cuando éste separa
    dos palabras inequívocamente identificables.

    Por ello, durante la reconstrucción, si dos unidades consecutivas son
    palabras inequívocas y no existe whitespace explícito entre ellas, el
    decoder inserta exactamente un espacio.
    """

    def __init__(
        self,
        dictionary: LexiMapDictionary,
    ) -> None:

        self._dictionary = dictionary

    @property
    def dictionary(
        self,
    ) -> LexiMapDictionary:
        return self._dictionary

    def decode(
        self,
        data: bytes,
    ) -> str:
        """
        Decodifica bytes LexiMapSp-16 y devuelve texto.
        """

        return self.decode_with_statistics(
            data
        ).text

    def decode_with_statistics(
        self,
        data: bytes,
    ) -> DecodingResult:
        """
        Decodifica bytes LexiMapSp-16 y devuelve texto más estadísticas.
        """

        if not isinstance(
            data,
            (bytes, bytearray),
        ):
            raise LexiMapDecodingError(
                "Los datos a decodificar deben ser bytes o bytearray."
            )

        binary = bytes(data)

        statistics = DecodingStatistics(
            encoded_bytes=len(binary)
        )

        units: list[DecodedUnit] = []

        position = 0
        pending_modifier: int | None = None

        while position < len(binary):

            # -------------------------------------------------------------
            # Necesitamos al menos dos bytes para interpretar una unidad
            # normal de LexiMapSp-16.
            # -------------------------------------------------------------

            if position + 2 > len(binary):
                raise LexiMapDecodingError(
                    "Flujo binario incompleto en la posición "
                    f"{position}."
                )

            code = bytes_to_code(
                binary[position:position + 2]
            )

            # -------------------------------------------------------------
            # LITERAL
            # -------------------------------------------------------------

            if code == ISO_START:

                if pending_modifier is not None:
                    raise LexiMapDecodingError(
                        "Se encontró un literal después de un "
                        "modificador de capitalización."
                    )

                literal_text, position = (
                    self._decode_literal(
                        binary,
                        position + 2,
                    )
                )

                units.append(
                    DecodedUnit(
                        text=literal_text,
                        unit_type=DecodedUnitType.LITERAL,
                        unambiguous_word=False,
                    )
                )

                statistics.literal_items += 1
                continue

            # -------------------------------------------------------------
            # ISO_END fuera de literal
            # -------------------------------------------------------------

            if code == ISO_END:
                raise LexiMapDecodingError(
                    "ISO_END encontrado fuera de un bloque literal "
                    f"en la posición {position}."
                )

            # -------------------------------------------------------------
            # MODIFICADORES
            # -------------------------------------------------------------

            if code in (
                CAPITALIZED,
                UPPERCASE,
            ):

                if pending_modifier is not None:
                    raise LexiMapDecodingError(
                        "Se encontraron modificadores consecutivos "
                        f"en la posición {position}."
                    )

                pending_modifier = code
                statistics.modifier_items += 1

                position += 2
                continue

            # -------------------------------------------------------------
            # RANGO RESERVADO
            # -------------------------------------------------------------

            if is_reserved_extension(
                code
            ):
                raise LexiMapDecodingError(
                    "Código reservado encontrado en el flujo: "
                    f"{code} (0x{code:04X})."
                )

            # -------------------------------------------------------------
            # TOKEN LÉXICO
            # -------------------------------------------------------------

            if is_lexical_token(
                code
            ):

                word = self._dictionary.get_word(
                    code
                )

                if word is None:
                    raise LexiMapDecodingError(
                        "El token léxico no existe en el "
                        f"diccionario cargado: {code}."
                    )

                word = self._apply_modifier(
                    word,
                    pending_modifier,
                )

                pending_modifier = None

                units.append(
                    DecodedUnit(
                        text=word,
                        unit_type=DecodedUnitType.WORD,
                        unambiguous_word=True,
                    )
                )

                statistics.lexical_items += 1

                position += 2
                continue

            # -------------------------------------------------------------
            # CÓDIGO DIRECTO
            # -------------------------------------------------------------

            character = self._decode_direct_code(
                code
            )

            # -------------------------------------------------------------
            # MODIFICADOR SOBRE DIRECT
            #
            # Actualmente sólo debería utilizarse para palabras directas
            # de una letra. El decoder aplica el modificador y marca la
            # unidad como palabra cuando corresponde.
            # -------------------------------------------------------------

            if pending_modifier is not None:

                if not self._is_direct_single_letter_word(
                    character
                ):
                    raise LexiMapDecodingError(
                        "Un modificador de capitalización fue aplicado "
                        "a un código directo que no representa una "
                        f"palabra válida de una letra: {character!r}."
                    )

                character = self._apply_modifier(
                    character,
                    pending_modifier,
                )

                pending_modifier = None

                units.append(
                    DecodedUnit(
                        text=character,
                        unit_type=DecodedUnitType.WORD,
                        unambiguous_word=True,
                    )
                )

                statistics.direct_items += 1

                position += 2
                continue

            # -------------------------------------------------------------
            # WHITESPACE DIRECTO
            # -------------------------------------------------------------

            if code in (
                SPACE,
                TAB,
                LF,
                CR,
            ):

                units.append(
                    DecodedUnit(
                        text=character,
                        unit_type=DecodedUnitType.WHITESPACE,
                        unambiguous_word=False,
                    )
                )

                statistics.direct_items += 1
                statistics.explicit_whitespace_items += 1

                position += 2
                continue

            # -------------------------------------------------------------
            # PALABRAS DIRECTAS DE UNA LETRA
            # -------------------------------------------------------------

            if self._is_direct_single_letter_word(
                character
            ):

                units.append(
                    DecodedUnit(
                        text=character,
                        unit_type=DecodedUnitType.WORD,
                        unambiguous_word=True,
                    )
                )

                statistics.direct_items += 1

                position += 2
                continue

            # -------------------------------------------------------------
            # CARÁCTER DIRECTO NORMAL
            # -------------------------------------------------------------

            units.append(
                DecodedUnit(
                    text=character,
                    unit_type=DecodedUnitType.CHARACTER,
                    unambiguous_word=False,
                )
            )

            statistics.direct_items += 1

            position += 2

        # -----------------------------------------------------------------
        # MODIFICADOR SIN OBJETIVO
        # -----------------------------------------------------------------

        if pending_modifier is not None:
            raise LexiMapDecodingError(
                "El flujo termina con un modificador de "
                "capitalización sin unidad asociada."
            )

        # -----------------------------------------------------------------
        # RECONSTRUCCIÓN
        # -----------------------------------------------------------------

        text = self._reconstruct(
            units,
            statistics,
        )

        statistics.decoded_characters = len(
            text
        )

        statistics.decoded_utf8_bytes = len(
            text.encode("utf-8")
        )

        return DecodingResult(
            text=text,
            statistics=statistics,
        )

    def _decode_literal(
        self,
        data: bytes,
        payload_start: int,
    ) -> tuple[str, int]:
        """
        Lee un bloque literal UTF-8.

        payload_start apunta al primer byte posterior a ISO_START.

        El bloque termina cuando aparece:

            00 01

        Devuelve:

            texto_literal,
            posición posterior a ISO_END
        """

        delimiter = bytes(
            (
                0x00,
                0x01,
            )
        )

        end = data.find(
            delimiter,
            payload_start,
        )

        if end == -1:
            raise LexiMapDecodingError(
                "Bloque literal sin ISO_END."
            )

        payload = data[
            payload_start:end
        ]

        try:
            text = payload.decode(
                "utf-8"
            )

        except UnicodeDecodeError as error:
            raise LexiMapDecodingError(
                "El payload de un bloque literal no contiene "
                "UTF-8 válido."
            ) from error

        return (
            text,
            end + 2,
        )

    def _decode_direct_code(
        self,
        code: int,
    ) -> str:
        """
        Convierte un código directo LexiMapSp-16 en carácter.

        0-127
            correspondencia ASCII directa.

        128-159
            sólo posiciones asignadas en la extensión tipográfica.

        160-255
            correspondencia ISO-8859-1 directa.
        """

        if 0 <= code <= 127:
            return chr(
                code
            )

        if code in TYPOGRAPHIC_EXTENSION:
            return TYPOGRAPHIC_EXTENSION[
                code
            ]

        if 160 <= code <= 255:
            return chr(
                code
            )

        raise LexiMapDecodingError(
            "Código directo no asignado: "
            f"{code} (0x{code:04X})."
        )

    @staticmethod
    def _is_direct_single_letter_word(
        character: str,
    ) -> bool:
        """
        Indica si el carácter corresponde a una de las palabras
        españolas de una letra representadas directamente.
        """

        return character.lower() in {
            "a",
            "e",
            "o",
            "u",
            "y",
        }

    @staticmethod
    def _apply_modifier(
        text: str,
        modifier: int | None,
    ) -> str:
        """
        Aplica CAPITALIZED o UPPERCASE.
        """

        if modifier is None:
            return text

        if modifier == CAPITALIZED:

            if not text:
                return text

            return (
                text[0].upper()
                + text[1:]
            )

        if modifier == UPPERCASE:
            return text.upper()

        raise LexiMapDecodingError(
            "Modificador desconocido: "
            f"{modifier}."
        )

    @staticmethod
    def _reconstruct(
        units: list[DecodedUnit],
        statistics: DecodingStatistics,
    ) -> str:
        """
        Reconstruye el texto e inserta los espacios implícitos.

        Se inserta exactamente un espacio cuando dos unidades
        consecutivas son palabras inequívocas.
        """

        if not units:
            return ""

        output: list[str] = []

        previous: DecodedUnit | None = None

        for unit in units:

            if (
                previous is not None
                and previous.unambiguous_word
                and unit.unambiguous_word
            ):
                output.append(
                    " "
                )

                statistics.reconstructed_implicit_spaces += 1

            output.append(
                unit.text
            )

            previous = unit

        return "".join(
            output
        )
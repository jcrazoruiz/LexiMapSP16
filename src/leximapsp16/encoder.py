from __future__ import annotations

from dataclasses import dataclass

from leximapsp16.constants import (
    CR,
    ISO_END,
    ISO_START,
    LF,
    SINGLE_LETTER_WORDS,
    SPACE,
    TAB,
    code_to_bytes,
    is_protocol_control_code,
)
from leximapsp16.dictionary import LexiMapDictionary
from leximapsp16.representation import (
    LexiMapRepresentationResolver,
    RepresentationType,
    TokenRepresentation,
    modifier_to_control_code,
)
from leximapsp16.tokenizer import (
    LexiMapTokenizer,
    TextToken,
    TokenType,
)


class LexiMapEncodingError(Exception):
    """
    Error producido durante la codificación LexiMapSp-16.
    """


@dataclass
class EncodingStatistics:
    """
    Estadísticas obtenidas durante una operación de codificación.
    """

    source_characters: int = 0
    source_utf8_bytes: int = 0
    encoded_bytes: int = 0

    lexical_items: int = 0
    direct_items: int = 0
    literal_items: int = 0

    implicit_spaces: int = 0
    explicit_single_spaces: int = 0
    explicit_whitespace_items: int = 0

    modifier_items: int = 0
    literal_payload_bytes: int = 0

    @property
    def reduction_bytes(self) -> int:
        """
        Diferencia entre UTF-8 original y LexiMapSp-16.

        Positivo:
            LexiMapSp-16 utiliza menos bytes.

        Negativo:
            LexiMapSp-16 utiliza más bytes.
        """

        return (
            self.source_utf8_bytes
            - self.encoded_bytes
        )

    @property
    def reduction_percentage(self) -> float:
        """
        Porcentaje de reducción respecto al tamaño UTF-8 original.
        """

        if self.source_utf8_bytes == 0:
            return 0.0

        return (
            self.reduction_bytes
            / self.source_utf8_bytes
            * 100.0
        )


@dataclass(frozen=True)
class EncodingResult:
    """
    Resultado de una operación de codificación.
    """

    data: bytes
    statistics: EncodingStatistics


class LexiMapEncoder:
    """
    Codificador LexiMapSp-16.

    El encoder recibe texto Unicode y produce una secuencia binaria
    conforme a las reglas actuales del formato.

    La codificación se realiza en las siguientes etapas:

        texto
          ↓
        tokenizer
          ↓
        representations
          ↓
        análisis contextual de espacios
          ↓
        bytes LexiMapSp-16

    Regla de espacios
    -----------------

    Un espacio simple se omite únicamente cuando su presencia puede
    reconstruirse de manera inequívoca a partir de las unidades que
    aparecen a ambos lados.

    Actualmente sólo se considera implícito cuando separa dos palabras
    representables inequívocamente como tales:

        LEXICAL ↔ LEXICAL
        LEXICAL ↔ DIRECT de una letra
        DIRECT de una letra ↔ LEXICAL
        DIRECT de una letra ↔ DIRECT de una letra

    En cualquier otro contexto el espacio simple se almacena
    explícitamente como:

        00 20

    Todo whitespace distinto de exactamente un espacio también se
    almacena explícitamente.
    """

    def __init__(
        self,
        dictionary: LexiMapDictionary,
    ) -> None:

        self._dictionary = dictionary

        self._tokenizer = LexiMapTokenizer()

        self._resolver = LexiMapRepresentationResolver(
            dictionary
        )

    @property
    def dictionary(
        self,
    ) -> LexiMapDictionary:
        return self._dictionary

    def encode(
        self,
        text: str,
    ) -> bytes:
        """
        Codifica texto y devuelve solamente los bytes resultantes.
        """

        return self.encode_with_statistics(
            text
        ).data

    def encode_with_statistics(
        self,
        text: str,
    ) -> EncodingResult:
        """
        Codifica texto y devuelve bytes más estadísticas.
        """

        if not isinstance(text, str):
            raise LexiMapEncodingError(
                "El valor a codificar debe ser str."
            )

        statistics = EncodingStatistics(
            source_characters=len(text),
            source_utf8_bytes=len(
                text.encode("utf-8")
            ),
        )

        tokens = self._tokenizer.tokenize(
            text
        )

        self._tokenizer.validate(
            text,
            tokens,
        )

        representations = [
            self._resolver.resolve(token)
            for token in tokens
        ]

        output = bytearray()

        for index, representation in enumerate(
            representations
        ):

            token = tokens[index]

            if (
                token.type == TokenType.WHITESPACE
                and token.text == " "
            ):
                self._encode_single_space(
                    index=index,
                    tokens=tokens,
                    representations=representations,
                    output=output,
                    statistics=statistics,
                )
                continue

            self._encode_representation(
                representation,
                output,
                statistics,
            )

        statistics.encoded_bytes = len(
            output
        )

        return EncodingResult(
            data=bytes(output),
            statistics=statistics,
        )

    def _encode_single_space(
        self,
        *,
        index: int,
        tokens: list[TextToken],
        representations: list[TokenRepresentation],
        output: bytearray,
        statistics: EncodingStatistics,
    ) -> None:
        """
        Decide si un espacio simple puede omitirse.

        El espacio sólo será implícito cuando ambas unidades adyacentes
        sean palabras representadas de manera inequívoca.
        """

        previous_index = index - 1
        next_index = index + 1

        if (
            previous_index >= 0
            and next_index < len(tokens)
            and self._is_unambiguous_word(
                tokens[previous_index],
                representations[previous_index],
            )
            and self._is_unambiguous_word(
                tokens[next_index],
                representations[next_index],
            )
        ):
            statistics.implicit_spaces += 1
            return

        output.extend(
            code_to_bytes(
                SPACE
            )
        )

        statistics.explicit_single_spaces += 1

    def _is_unambiguous_word(
        self,
        token: TextToken,
        representation: TokenRepresentation,
    ) -> bool:
        """
        Indica si una unidad puede ser reconocida inequívocamente como
        palabra por el decoder.

        Se aceptan:

        1. Palabras codificadas mediante token léxico.
        2. Palabras de una sola letra definidas por LexiMapSp-16 y
           codificadas mediante código directo.

        Los OOV almacenados como literal NO se consideran inequívocos,
        porque un bloque literal también puede representar otros tipos
        de contenido Unicode.
        """

        if token.type != TokenType.WORD:
            return False

        if (
            representation.representation_type
            == RepresentationType.LEXICAL
        ):
            return True

        if (
            representation.representation_type
            == RepresentationType.DIRECT
            and representation.normalized_word
            in SINGLE_LETTER_WORDS
        ):
            return True

        return False

    def _encode_representation(
        self,
        representation: TokenRepresentation,
        output: bytearray,
        statistics: EncodingStatistics,
    ) -> None:
        """
        Convierte una representación lógica en bytes.
        """

        representation_type = (
            representation.representation_type
        )

        if (
            representation_type
            == RepresentationType.IMPLICIT_SPACE
        ):
            statistics.implicit_spaces += 1
            return

        if (
            representation_type
            == RepresentationType.LEXICAL
        ):
            self._append_modifier(
                representation,
                output,
                statistics,
            )

            lexical_token = (
                representation.lexical_token
            )

            if lexical_token is None:
                raise LexiMapEncodingError(
                    "Representación LEXICAL sin token."
                )

            output.extend(
                code_to_bytes(
                    lexical_token
                )
            )

            statistics.lexical_items += 1
            return

        if (
            representation_type
            == RepresentationType.DIRECT
        ):
            direct_code = (
                representation.direct_code
            )

            if direct_code is None:
                raise LexiMapEncodingError(
                    "Representación DIRECT sin código."
                )

            # ---------------------------------------------------------
            # RANGO DE CONTROL RESERVADO DEL PROTOCOLO
            #
            # 00 00..00 0F pertenecen al espacio de control LexiMapSp-16.
            #
            # 00-03 tienen significado asignado; 09, 0A y 0D conservan
            # TAB, LF y CR; 04,05,06,07,08,0B,0C,0E,0F quedan reservados.
            #
            # Los códigos no asignados como caracteres directos válidos
            # se preservan mediante literal UTF-8.
            # ---------------------------------------------------------
            if (
                is_protocol_control_code(direct_code)
                and direct_code not in (TAB, LF, CR)
            ):
                self._encode_literal(
                    chr(direct_code),
                    output,
                    statistics,
                )

                statistics.literal_items += 1
                return

            self._append_modifier(
                representation,
                output,
                statistics,
            )

            output.extend(
                code_to_bytes(
                    direct_code
                )
            )

            statistics.direct_items += 1
            return

        if (
            representation_type
            == RepresentationType.LITERAL
        ):
            literal_text = (
                representation.literal_text
            )

            if literal_text is None:
                raise LexiMapEncodingError(
                    "Representación LITERAL sin contenido."
                )

            self._encode_literal(
                literal_text,
                output,
                statistics,
            )

            statistics.literal_items += 1
            return

        if (
            representation_type
            == RepresentationType.EXPLICIT_WHITESPACE
        ):
            literal_text = (
                representation.literal_text
            )

            if literal_text is None:
                raise LexiMapEncodingError(
                    "Whitespace explícito sin contenido."
                )

            self._encode_explicit_whitespace(
                literal_text,
                output,
                statistics,
            )

            statistics.explicit_whitespace_items += 1
            return

        raise LexiMapEncodingError(
            "Tipo de representación no soportado: "
            f"{representation_type}"
        )

    def _append_modifier(
        self,
        representation: TokenRepresentation,
        output: bytearray,
        statistics: EncodingStatistics,
    ) -> None:
        """
        Escribe CAPITALIZED o UPPERCASE cuando corresponda.
        """

        modifier_code = modifier_to_control_code(
            representation.modifier
        )

        if modifier_code is None:
            return

        output.extend(
            code_to_bytes(
                modifier_code
            )
        )

        statistics.modifier_items += 1

    def _encode_literal(
        self,
        text: str,
        output: bytearray,
        statistics: EncodingStatistics,
    ) -> None:
        """
        Codifica un bloque literal UTF-8.

        Formato:

            00 00
            <payload UTF-8 con byte stuffing>
            00 01

        Dentro del payload:

            00 00
                representa un byte NUL literal.

            00 01
                representa exclusivamente el cierre del bloque.
        """

        payload = text.encode(
            "utf-8"
        )

        # Byte stuffing dentro del bloque literal:
        #
        #   00 00 = byte NUL literal
        #   00 01 = cierre del bloque
        #
        # De esta forma una secuencia literal 00 01 nunca puede
        # confundirse con ISO_END.
        stuffed_payload = payload.replace(
            b"\x00",
            b"\x00\x00",
        )

        output.extend(
            code_to_bytes(
                ISO_START
            )
        )

        output.extend(
            stuffed_payload
        )

        output.extend(
            code_to_bytes(
                ISO_END
            )
        )

        # Se conserva la semántica histórica de esta estadística:
        # mide bytes útiles del payload UTF-8, no bytes de framing ni
        # bytes añadidos por stuffing.
        statistics.literal_payload_bytes += len(
            payload
        )

    def _encode_explicit_whitespace(
        self,
        text: str,
        output: bytearray,
        statistics: EncodingStatistics,
    ) -> None:
        """
        Codifica whitespace que no puede ser implícito.

        Los caracteres habituales tienen representación directa.

        Cualquier otro whitespace Unicode se almacena mediante bloque
        literal UTF-8.
        """

        for character in text:

            if character == " ":
                output.extend(
                    code_to_bytes(
                        SPACE
                    )
                )
                continue

            if character == "\t":
                output.extend(
                    code_to_bytes(
                        TAB
                    )
                )
                continue

            if character == "\n":
                output.extend(
                    code_to_bytes(
                        LF
                    )
                )
                continue

            if character == "\r":
                output.extend(
                    code_to_bytes(
                        CR
                    )
                )
                continue

            self._encode_literal(
                character,
                output,
                statistics,
            )
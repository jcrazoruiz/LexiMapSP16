from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .constants import (
    EXPECTED_LEXICAL_TOKEN_COUNT,
    LEXICAL_TOKEN_START,
    ranking_to_token,
)


# =============================================================================
# EXCEPCIONES
# =============================================================================

class LexiMapDictionaryError(Exception):
    """Error relacionado con la carga o validación de un diccionario LexiMapSp-16."""


# =============================================================================
# ESTRUCTURAS DE DATOS
# =============================================================================

@dataclass(frozen=True)
class DictionaryEntry:
    ranking: int
    token: int
    word: str
    frequency: int | None = None


# =============================================================================
# DICCIONARIO LEXIMAPSP-16
# =============================================================================

class LexiMapDictionary:
    """
    Representa un diccionario léxico utilizado por LexiMapSp-16.

    El diccionario se carga a partir de uno de los CSV generados por
    LexiCorpus.

    Cada entrada debe cumplir:

        Token = Ranking + 535

    Por lo tanto:

        Ranking 1      -> Token 536
        Ranking 20,000 -> Token 20,535
        Ranking 65,000 -> Token 65,535

    El tamaño del diccionario puede variar entre experimentos.
    """

    REQUIRED_COLUMNS = {
        "Ranking",
        "Token",
        "Palabra",
    }

    def __init__(
        self,
        entries: list[DictionaryEntry],
        source_path: Path | None = None,
    ) -> None:

        if not entries:
            raise LexiMapDictionaryError(
                "El diccionario no contiene entradas."
            )

        self._entries = tuple(entries)
        self._source_path = source_path

        self._word_to_token: dict[str, int] = {}
        self._token_to_word: dict[int, str] = {}

        self._build_indexes()
        self._validate()


    # =========================================================================
    # CONSTRUCCIÓN DESDE CSV
    # =========================================================================

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
    ) -> "LexiMapDictionary":

        path = Path(csv_path)

        if not path.exists():
            raise FileNotFoundError(
                f"No existe el archivo de diccionario: {path}"
            )

        if not path.is_file():
            raise LexiMapDictionaryError(
                f"La ruta no corresponde a un archivo: {path}"
            )

        entries: list[DictionaryEntry] = []

        with path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise LexiMapDictionaryError(
                    "El CSV no contiene encabezados."
                )

            missing_columns = (
                cls.REQUIRED_COLUMNS
                - set(reader.fieldnames)
            )

            if missing_columns:
                missing = ", ".join(
                    sorted(missing_columns)
                )

                raise LexiMapDictionaryError(
                    "El CSV no contiene las columnas requeridas: "
                    f"{missing}"
                )

            for line_number, row in enumerate(
                reader,
                start=2,
            ):
                try:
                    ranking = int(row["Ranking"])
                    token = int(row["Token"])
                    word = row["Palabra"]

                    frequency = None

                    if "Frecuencia" in row:
                        raw_frequency = row["Frecuencia"]

                        if (
                            raw_frequency is not None
                            and raw_frequency.strip()
                        ):
                            frequency = int(raw_frequency)

                except (TypeError, ValueError) as exc:
                    raise LexiMapDictionaryError(
                        "Datos inválidos en la línea "
                        f"{line_number} de {path.name}."
                    ) from exc

                if word is None:
                    raise LexiMapDictionaryError(
                        "Palabra vacía en la línea "
                        f"{line_number}."
                    )

                word = word.strip()

                if not word:
                    raise LexiMapDictionaryError(
                        "Palabra vacía en la línea "
                        f"{line_number}."
                    )

                entries.append(
                    DictionaryEntry(
                        ranking=ranking,
                        token=token,
                        word=word,
                        frequency=frequency,
                    )
                )

        return cls(
            entries=entries,
            source_path=path,
        )


    # =========================================================================
    # ÍNDICES
    # =========================================================================

    def _build_indexes(self) -> None:

        for entry in self._entries:

            if entry.word in self._word_to_token:
                raise LexiMapDictionaryError(
                    "Palabra duplicada en el diccionario: "
                    f"{entry.word!r}"
                )

            if entry.token in self._token_to_word:
                raise LexiMapDictionaryError(
                    "Token duplicado en el diccionario: "
                    f"{entry.token}"
                )

            self._word_to_token[entry.word] = entry.token
            self._token_to_word[entry.token] = entry.word


    # =========================================================================
    # VALIDACIÓN
    # =========================================================================

    def _validate(self) -> None:

        dictionary_size = len(self._entries)

        if dictionary_size > EXPECTED_LEXICAL_TOKEN_COUNT:
            raise LexiMapDictionaryError(
                "El diccionario contiene más de "
                f"{EXPECTED_LEXICAL_TOKEN_COUNT:,} entradas."
            )

        for expected_ranking, entry in enumerate(
            self._entries,
            start=1,
        ):

            if entry.ranking != expected_ranking:
                raise LexiMapDictionaryError(
                    "Ranking no consecutivo. "
                    f"Se esperaba {expected_ranking} "
                    f"y se encontró {entry.ranking}."
                )

            expected_token = ranking_to_token(
                entry.ranking
            )

            if entry.token != expected_token:
                raise LexiMapDictionaryError(
                    "Token incorrecto para ranking "
                    f"{entry.ranking}. "
                    f"Se esperaba {expected_token} "
                    f"y se encontró {entry.token}."
                )

        expected_last_token = (
            LEXICAL_TOKEN_START
            + dictionary_size
            - 1
        )

        if self._entries[-1].token != expected_last_token:
            raise LexiMapDictionaryError(
                "El token final del diccionario "
                "no corresponde con su tamaño."
            )


    # =========================================================================
    # CONSULTAS
    # =========================================================================

    def contains_word(self, word: str) -> bool:
        return word in self._word_to_token


    def contains_token(self, token: int) -> bool:
        return token in self._token_to_word


    def get_token(
        self,
        word: str,
    ) -> int | None:
        return self._word_to_token.get(word)


    def get_word(
        self,
        token: int,
    ) -> str | None:
        return self._token_to_word.get(token)


    def require_token(
        self,
        word: str,
    ) -> int:

        token = self.get_token(word)

        if token is None:
            raise KeyError(
                f"La palabra {word!r} no existe "
                "en el diccionario."
            )

        return token


    def require_word(
        self,
        token: int,
    ) -> str:

        word = self.get_word(token)

        if word is None:
            raise KeyError(
                f"El token {token} no existe "
                "en el diccionario."
            )

        return word


    # =========================================================================
    # PROPIEDADES
    # =========================================================================

    @property
    def size(self) -> int:
        return len(self._entries)


    @property
    def first_token(self) -> int:
        return self._entries[0].token


    @property
    def last_token(self) -> int:
        return self._entries[-1].token


    @property
    def source_path(self) -> Path | None:
        return self._source_path


    @property
    def entries(self) -> tuple[DictionaryEntry, ...]:
        return self._entries


    # =========================================================================
    # INFORMACIÓN
    # =========================================================================

    def summary(self) -> dict[str, object]:

        return {
            "source": (
                str(self._source_path)
                if self._source_path
                else None
            ),
            "entries": self.size,
            "first_token": self.first_token,
            "last_token": self.last_token,
        }


    def __len__(self) -> int:
        return self.size


    def __contains__(self, word: str) -> bool:
        return self.contains_word(word)


    def __repr__(self) -> str:

        return (
            "LexiMapDictionary("
            f"size={self.size:,}, "
            f"tokens={self.first_token:,}-"
            f"{self.last_token:,}"
            ")"
        )
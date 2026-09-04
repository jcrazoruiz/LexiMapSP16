from __future__ import annotations

import argparse
import csv
import json
import sys

from pathlib import Path
from time import perf_counter


# =============================================================================
# CONFIGURACIÓN DEL PROYECTO
# =============================================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SRC_PATH = (
    PROJECT_ROOT
    / "src"
)

sys.path.insert(
    0,
    str(SRC_PATH),
)


# =============================================================================
# IMPORTS LEXIMAPSP-16
# =============================================================================

from leximapsp16.decoder import LexiMapDecoder  # noqa: E402
from leximapsp16.dictionary import LexiMapDictionary  # noqa: E402
from leximapsp16.encoder import LexiMapEncoder  # noqa: E402


# =============================================================================
# CONFIGURACIÓN GENERAL
# =============================================================================

DICTIONARY_DIRECTORY = (
    PROJECT_ROOT
    / "docs"
    / "input"
)

HELDOUT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "heldout"
)

FIRST_LEXICAL_TOKEN = 536

DEFAULT_DATASET_ROLE = "HELD_OUT"


# =============================================================================
# COLUMNAS DEL CSV EXPERIMENTAL
# =============================================================================

CSV_FIELDS = [
    "Dictionary",
    "DictionarySize",
    "FirstToken",
    "LastToken",
    "Dataset",
    "DatasetRole",
    "SnapshotDate",
    "Segment",
    "DocumentsEvaluated",
    "RoundTripCorrect",
    "RoundTripErrors",
    "CodecExceptions",
    "StatisticsErrors",
    "SourceCharacters",
    "UTF8Bytes",
    "LexiMapBytes",
    "DifferenceBytes",
    "ReductionPercentage",
    "LexicalItems",
    "DirectItems",
    "LiteralItems",
    "LiteralPayloadBytes",
    "ImplicitSpaces",
    "ExplicitSingleSpaces",
    "ExplicitWhitespaceItems",
    "ModifierItems",
    "ElapsedSeconds",
    "RoundTripSuccessPercentage",
    "Result",
]


# =============================================================================
# ARGUMENTOS
# =============================================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Valida round-trip de LexiMapSp-16 sobre un conjunto "
            "held-out parametrizable usando un diccionario LexiCorpus."
        )
    )

    parser.add_argument(
        "--dictionary",
        required=True,
        help=(
            "Nombre del archivo CSV del diccionario dentro de docs/input. "
            "Ejemplo: LexiCorpus_65000_Completo.csv"
        ),
    )

    parser.add_argument(
        "--size",
        required=True,
        type=int,
        help=(
            "Número esperado de entradas del diccionario. "
            "Ejemplo: 20000, 30000, 40000, 50000 o 65000."
        ),
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Nombre del directorio del dataset dentro de data/heldout. "
            "Ejemplo: "
            "wikipedia_es_20260801_segment2_skip1000_count2000"
        ),
    )

    parser.add_argument(
        "--count",
        required=True,
        type=int,
        help=(
            "Cantidad esperada de documentos en el dataset held-out."
        ),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "CSV donde se almacenará o actualizará el resultado experimental. "
            "Ejemplo: reports/heldout_benchmark_results_sample2.csv"
        ),
    )

    return parser.parse_args()


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def format_elapsed(
    seconds: float,
) -> str:

    total = int(
        seconds
    )

    hours, remainder = divmod(
        total,
        3600,
    )

    minutes, seconds = divmod(
        remainder,
        60,
    )

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{seconds:02d}"
    )


def load_manifest(
    path: Path,
) -> list[dict]:

    records: list[dict] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:

                record = json.loads(
                    line
                )

            except json.JSONDecodeError as error:

                raise ValueError(
                    "JSON inválido en manifiesto, "
                    f"línea {line_number}: {error}"
                ) from error

            records.append(
                record
            )

    return records


def validate_manifest(
    records: list[dict],
    expected_documents: int,
) -> None:

    if len(records) != expected_documents:

        raise ValueError(
            "Cantidad incorrecta de registros en manifiesto. "
            f"Esperados: {expected_documents:,}; "
            f"encontrados: {len(records):,}."
        )

    page_ids: list[str] = []

    for expected_order, record in enumerate(
        records,
        start=1,
    ):

        selection_order = record.get(
            "selection_order"
        )

        if selection_order != expected_order:

            raise ValueError(
                "Orden de selección inconsistente. "
                f"Esperado: {expected_order}; "
                f"encontrado: {selection_order}."
            )

        page_id = str(
            record.get(
                "page_id",
                "",
            )
        ).strip()

        if not page_id:

            raise ValueError(
                "Registro sin page_id en posición "
                f"{expected_order}."
            )

        page_ids.append(
            page_id
        )

    if (
        len(set(page_ids))
        != expected_documents
    ):

        raise ValueError(
            "El manifiesto contiene page_id duplicados."
        )


def extract_manifest_metadata(
    records: list[dict],
) -> dict:

    if not records:

        raise ValueError(
            "El manifiesto está vacío."
        )

    snapshot_values = {
        str(
            record.get(
                "snapshot_date",
                "",
            )
        ).strip()
        for record in records
    }

    snapshot_values.discard(
        ""
    )

    if len(snapshot_values) > 1:

        raise ValueError(
            "El manifiesto contiene más de un snapshot_date."
        )

    segment_values = {
        str(
            record.get(
                "segment",
                "",
            )
        ).strip()
        for record in records
    }

    segment_values.discard(
        ""
    )

    if len(segment_values) > 1:

        raise ValueError(
            "El manifiesto contiene más de un segmento."
        )

    role_values = {
        str(
            record.get(
                "evaluation_role",
                "",
            )
        ).strip()
        for record in records
    }

    role_values.discard(
        ""
    )

    if len(role_values) > 1:

        raise ValueError(
            "El manifiesto contiene más de un evaluation_role."
        )

    snapshot_date = (
        next(
            iter(snapshot_values),
            "",
        )
    )

    segment = (
        next(
            iter(segment_values),
            "",
        )
    )

    dataset_role = (
        next(
            iter(role_values),
            DEFAULT_DATASET_ROLE,
        )
    )

    valid_orders: list[int] = []

    for record in records:

        value = record.get(
            "valid_article_order"
        )

        if value is None:
            continue

        try:

            valid_orders.append(
                int(value)
            )

        except (
            TypeError,
            ValueError,
        ) as error:

            raise ValueError(
                "valid_article_order inválido en manifiesto."
            ) from error

    first_valid_order = (
        min(valid_orders)
        if valid_orders
        else None
    )

    last_valid_order = (
        max(valid_orders)
        if valid_orders
        else None
    )

    if valid_orders:

        if (
            len(valid_orders)
            != len(records)
        ):

            raise ValueError(
                "valid_article_order está presente sólo "
                "en una parte del manifiesto."
            )

        expected_valid_orders = list(
            range(
                first_valid_order,
                last_valid_order + 1,
            )
        )

        if (
            valid_orders
            != expected_valid_orders
        ):

            raise ValueError(
                "valid_article_order no forma una secuencia "
                "continua y ordenada."
            )

    return {
        "snapshot_date": snapshot_date,
        "segment": segment,
        "dataset_role": dataset_role,
        "first_valid_order": first_valid_order,
        "last_valid_order": last_valid_order,
    }


def find_first_difference(
    original: str,
    reconstructed: str,
) -> tuple[int, str | None, str | None] | None:

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


def resolve_output_csv(
    output_csv: Path | None,
) -> Path | None:

    if output_csv is None:
        return None

    if output_csv.is_absolute():
        return output_csv

    return (
        PROJECT_ROOT
        / output_csv
    )


def write_benchmark_result(
    csv_path: Path,
    result: dict,
) -> str:

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_rows: list[dict] = []

    if csv_path.exists():

        with csv_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file
            )

            if reader.fieldnames != CSV_FIELDS:

                raise ValueError(
                    "El CSV existente tiene una estructura "
                    "diferente a la esperada."
                )

            existing_rows = list(
                reader
            )

    dictionary_name = str(
        result["Dictionary"]
    )

    dictionary_size = str(
        result["DictionarySize"]
    )

    dataset_name = str(
        result["Dataset"]
    )

    updated = False

    for index, row in enumerate(
        existing_rows
    ):

        if (
            row.get("Dictionary")
            == dictionary_name
            and row.get("DictionarySize")
            == dictionary_size
            and row.get("Dataset")
            == dataset_name
        ):

            existing_rows[index] = {
                field: result[field]
                for field in CSV_FIELDS
            }

            updated = True
            break

    if not updated:

        existing_rows.append(
            {
                field: result[field]
                for field in CSV_FIELDS
            }
        )

    existing_rows.sort(
        key=lambda row: (
            row["Dataset"],
            int(
                row["DictionarySize"]
            ),
        )
    )

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CSV_FIELDS,
        )

        writer.writeheader()

        writer.writerows(
            existing_rows
        )

    if updated:
        return "UPDATED"

    return "INSERTED"


# =============================================================================
# PRUEBA HELD-OUT
# =============================================================================

def main() -> int:

    args = parse_arguments()

    dictionary_path = (
        DICTIONARY_DIRECTORY
        / args.dictionary
    )

    heldout_directory = (
        HELDOUT_ROOT
        / args.dataset
    )

    article_directory = (
        heldout_directory
        / "articles"
    )

    manifest_path = (
        heldout_directory
        / "manifest.jsonl"
    )

    output_csv = resolve_output_csv(
        args.output_csv
    )

    expected_dictionary_size = (
        args.size
    )

    expected_documents = (
        args.count
    )

    dataset_name = (
        args.dataset
    )

    if expected_dictionary_size <= 0:

        print(
            "ERROR: --size debe ser mayor que cero."
        )

        return 1

    if expected_documents <= 0:

        print(
            "ERROR: --count debe ser mayor que cero."
        )

        return 1

    expected_first_token = (
        FIRST_LEXICAL_TOKEN
    )

    expected_last_token = (
        FIRST_LEXICAL_TOKEN
        + expected_dictionary_size
        - 1
    )

    if expected_last_token > 65_535:

        print(
            "ERROR: el tamaño solicitado excede "
            "el rango léxico disponible de 16 bits."
        )

        return 1

    print()
    print("=" * 88)

    print(
        "LEXIMAPSP-16 - VALIDACIÓN ROUND-TRIP HELD-OUT"
    )

    print("=" * 88)

    print(
        "Diccionario            : "
        f"{dictionary_path.name}"
    )

    print(
        "Tamaño esperado        : "
        f"{expected_dictionary_size:,}"
    )

    print(
        "Rango esperado         : "
        f"{expected_first_token:,}-"
        f"{expected_last_token:,}"
    )

    print(
        "Dataset                : "
        f"{dataset_name}"
    )

    print(
        "Documentos esperados   : "
        f"{expected_documents:,}"
    )

    if output_csv is not None:

        print(
            "CSV resultados         : "
            f"{output_csv}"
        )

    print("-" * 88)

    # =========================================================================
    # VALIDACIÓN DE ARCHIVOS
    # =========================================================================

    if not dictionary_path.exists():

        print(
            "ERROR: no existe el diccionario:"
        )

        print(
            dictionary_path
        )

        return 1

    if not heldout_directory.exists():

        print(
            "ERROR: no existe el dataset held-out:"
        )

        print(
            heldout_directory
        )

        return 1

    if not manifest_path.exists():

        print(
            "ERROR: no existe el manifiesto:"
        )

        print(
            manifest_path
        )

        return 1

    if not article_directory.exists():

        print(
            "ERROR: no existe el directorio de artículos:"
        )

        print(
            article_directory
        )

        return 1

    # =========================================================================
    # MANIFIESTO
    # =========================================================================

    try:

        records = load_manifest(
            manifest_path
        )

        validate_manifest(
            records,
            expected_documents,
        )

        manifest_metadata = (
            extract_manifest_metadata(
                records
            )
        )

    except Exception as error:

        print(
            "ERROR AL VALIDAR EL MANIFIESTO:"
        )

        print(
            f"{type(error).__name__}: {error}"
        )

        return 1

    dataset_role = (
        manifest_metadata[
            "dataset_role"
        ]
    )

    snapshot_date = (
        manifest_metadata[
            "snapshot_date"
        ]
    )

    segment_name = (
        manifest_metadata[
            "segment"
        ]
    )

    first_valid_order = (
        manifest_metadata[
            "first_valid_order"
        ]
    )

    last_valid_order = (
        manifest_metadata[
            "last_valid_order"
        ]
    )

    print(
        "Rol                    : "
        f"{dataset_role}"
    )

    print(
        "Snapshot               : "
        f"{snapshot_date or 'NO DISPONIBLE'}"
    )

    print(
        "Segmento               : "
        f"{segment_name or 'NO DISPONIBLE'}"
    )

    if (
        first_valid_order is not None
        and last_valid_order is not None
    ):

        print(
            "Posiciones válidas     : "
            f"{first_valid_order:,}-"
            f"{last_valid_order:,}"
        )

    print("-" * 88)

    # =========================================================================
    # DICCIONARIO
    # =========================================================================

    try:

        dictionary = (
            LexiMapDictionary.from_csv(
                dictionary_path
            )
        )

    except Exception as error:

        print(
            "ERROR AL CARGAR EL DICCIONARIO:"
        )

        print(
            f"{type(error).__name__}: {error}"
        )

        return 1

    print(
        "Entradas diccionario   : "
        f"{dictionary.size:,}"
    )

    print(
        "Rango real tokens      : "
        f"{dictionary.first_token:,}-"
        f"{dictionary.last_token:,}"
    )

    if (
        dictionary.size
        != expected_dictionary_size
    ):

        print(
            "ERROR: el tamaño real del diccionario "
            "no coincide con --size."
        )

        print(
            "Esperado              : "
            f"{expected_dictionary_size:,}"
        )

        print(
            "Encontrado            : "
            f"{dictionary.size:,}"
        )

        return 1

    if (
        dictionary.first_token
        != expected_first_token
        or dictionary.last_token
        != expected_last_token
    ):

        print(
            "ERROR: el rango real de tokens "
            "no coincide con el esperado."
        )

        print(
            "Esperado              : "
            f"{expected_first_token:,}-"
            f"{expected_last_token:,}"
        )

        print(
            "Encontrado            : "
            f"{dictionary.first_token:,}-"
            f"{dictionary.last_token:,}"
        )

        return 1

    # =========================================================================
    # CODEC
    # =========================================================================

    encoder = LexiMapEncoder(
        dictionary
    )

    decoder = LexiMapDecoder(
        dictionary
    )

    # =========================================================================
    # CONTADORES
    # =========================================================================

    successful = 0
    roundtrip_failed = 0
    codec_exceptions = 0
    statistics_errors = 0

    total_characters = 0
    total_utf8_bytes = 0
    total_encoded_bytes = 0

    total_lexical_items = 0
    total_direct_items = 0
    total_literal_items = 0
    total_literal_payload_bytes = 0

    total_implicit_spaces = 0
    total_explicit_single_spaces = 0
    total_explicit_whitespace_items = 0
    total_modifier_items = 0

    failure_details: list[dict] = []
    statistics_error_details: list[dict] = []

    timer_start = (
        perf_counter()
    )

    # =========================================================================
    # PROCESAMIENTO DOCUMENTO POR DOCUMENTO
    # =========================================================================

    for document_number, record in enumerate(
        records,
        start=1,
    ):

        page_id = str(
            record["page_id"]
        )

        article_path = (
            article_directory
            / f"{page_id}.txt"
        )

        if not article_path.exists():

            codec_exceptions += 1

            failure_details.append(
                {
                    "document_number": document_number,
                    "page_id": page_id,
                    "error": "FILE_NOT_FOUND",
                    "path": str(article_path),
                }
            )

            continue

        try:

            original_text = (
                article_path.read_text(
                    encoding="utf-8"
                )
            )

            utf8_bytes = len(
                original_text.encode(
                    "utf-8"
                )
            )

            encoding_result = (
                encoder.encode_with_statistics(
                    original_text
                )
            )

            encoded_data = (
                encoding_result.data
            )

            decoding_result = (
                decoder.decode_with_statistics(
                    encoded_data
                )
            )

            reconstructed_text = (
                decoding_result.text
            )

        except Exception as error:

            codec_exceptions += 1

            failure_details.append(
                {
                    "document_number": (
                        document_number
                    ),
                    "page_id": (
                        page_id
                    ),
                    "error": (
                        "CODEC_EXCEPTION"
                    ),
                    "exception_type": (
                        type(error).__name__
                    ),
                    "message": (
                        str(error)
                    ),
                }
            )

            continue

        # =====================================================================
        # COMPARACIÓN EXACTA
        # =====================================================================

        if (
            reconstructed_text
            == original_text
        ):

            successful += 1

        else:

            roundtrip_failed += 1

            difference_detail = (
                find_first_difference(
                    original_text,
                    reconstructed_text,
                )
            )

            detail = {
                "document_number": (
                    document_number
                ),
                "page_id": (
                    page_id
                ),
                "error": (
                    "ROUND_TRIP_MISMATCH"
                ),
                "original_length": (
                    len(original_text)
                ),
                "reconstructed_length": (
                    len(reconstructed_text)
                ),
            }

            if difference_detail is not None:

                (
                    difference_index,
                    original_character,
                    reconstructed_character,
                ) = difference_detail

                detail[
                    "first_difference_index"
                ] = difference_index

                detail[
                    "original_character"
                ] = original_character

                detail[
                    "reconstructed_character"
                ] = reconstructed_character

            failure_details.append(
                detail
            )

        # =====================================================================
        # TAMAÑOS
        # =====================================================================

        total_characters += len(
            original_text
        )

        total_utf8_bytes += (
            utf8_bytes
        )

        total_encoded_bytes += len(
            encoded_data
        )

        # =====================================================================
        # ESTADÍSTICAS
        # =====================================================================

        try:

            statistics = (
                encoding_result.statistics
            )

            total_lexical_items += (
                statistics.lexical_items
            )

            total_direct_items += (
                statistics.direct_items
            )

            total_literal_items += (
                statistics.literal_items
            )

            total_literal_payload_bytes += (
                statistics.literal_payload_bytes
            )

            total_implicit_spaces += (
                statistics.implicit_spaces
            )

            total_explicit_single_spaces += (
                statistics.explicit_single_spaces
            )

            total_explicit_whitespace_items += (
                statistics.explicit_whitespace_items
            )

            total_modifier_items += (
                statistics.modifier_items
            )

        except Exception as error:

            statistics_errors += 1

            statistics_error_details.append(
                {
                    "document_number": (
                        document_number
                    ),
                    "page_id": (
                        page_id
                    ),
                    "exception_type": (
                        type(error).__name__
                    ),
                    "message": (
                        str(error)
                    ),
                }
            )

        # =====================================================================
        # PROGRESO
        # =====================================================================

        if (
            document_number % 100
            == 0
        ):

            elapsed = (
                perf_counter()
                - timer_start
            )

            print(
                "\r"
                "Procesados: "
                f"{document_number:,}"
                "/"
                f"{expected_documents:,}"
                " | OK: "
                f"{successful:,}"
                " | Round-trip err: "
                f"{roundtrip_failed:,}"
                " | Excepciones: "
                f"{codec_exceptions:,}"
                " | Tiempo: "
                f"{elapsed:,.1f}s",
                end="",
                flush=True,
            )

    print()

    elapsed = (
        perf_counter()
        - timer_start
    )

    # =========================================================================
    # CÁLCULOS FINALES
    # =========================================================================

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

    if records:

        roundtrip_success_percentage = (
            successful
            / len(records)
            * 100.0
        )

    else:

        roundtrip_success_percentage = 0.0

    codec_success = (
        successful
        == expected_documents
        and roundtrip_failed == 0
        and codec_exceptions == 0
    )

    statistics_success = (
        statistics_errors == 0
    )

    if codec_success and statistics_success:

        final_result = "OK"

    elif codec_success:

        final_result = "STATISTICS_ERROR"

    else:

        final_result = "CODEC_OR_ROUNDTRIP_ERROR"

    # =========================================================================
    # RESULTADOS
    # =========================================================================

    print()
    print("=" * 88)

    print(
        "RESULTADO ROUND-TRIP HELD-OUT"
    )

    print("-" * 88)

    print(
        "Dataset                : "
        f"{dataset_name}"
    )

    print(
        "Diccionario            : "
        f"{dictionary_path.name}"
    )

    print(
        "Tamaño diccionario     : "
        f"{dictionary.size:,}"
    )

    print(
        "Rango tokens           : "
        f"{dictionary.first_token:,}-"
        f"{dictionary.last_token:,}"
    )

    print(
        "Documentos evaluados   : "
        f"{len(records):,}"
    )

    print(
        "Correctos              : "
        f"{successful:,}"
    )

    print(
        "Errores round-trip     : "
        f"{roundtrip_failed:,}"
    )

    print(
        "Excepciones codec      : "
        f"{codec_exceptions:,}"
    )

    print(
        "Errores estadísticas   : "
        f"{statistics_errors:,}"
    )

    print(
        "Caracteres procesados  : "
        f"{total_characters:,}"
    )

    print(
        "Bytes UTF-8            : "
        f"{total_utf8_bytes:,}"
    )

    print(
        "Bytes LexiMap          : "
        f"{total_encoded_bytes:,}"
    )

    print(
        "Tiempo total           : "
        f"{format_elapsed(elapsed)}"
    )

    print("-" * 88)

    print(
        "Ítems léxicos          : "
        f"{total_lexical_items:,}"
    )

    print(
        "Ítems directos         : "
        f"{total_direct_items:,}"
    )

    print(
        "Ítems literales        : "
        f"{total_literal_items:,}"
    )

    print(
        "Payload literal bytes  : "
        f"{total_literal_payload_bytes:,}"
    )

    print(
        "Espacios implícitos    : "
        f"{total_implicit_spaces:,}"
    )

    print(
        "Espacios simples exp.  : "
        f"{total_explicit_single_spaces:,}"
    )

    print(
        "Whitespace explícito   : "
        f"{total_explicit_whitespace_items:,}"
    )

    print(
        "Modificadores de caso  : "
        f"{total_modifier_items:,}"
    )

    print("-" * 88)

    print(
        "Diferencia bytes       : "
        f"{difference:+,}"
    )

    print(
        "Reducción observada    : "
        f"{reduction_percentage:.4f}%"
    )

    print(
        "Round-trip correcto    : "
        f"{roundtrip_success_percentage:.4f}%"
    )

    print()

    # =========================================================================
    # DETALLE DE ERRORES
    # =========================================================================

    if failure_details:

        print(
            "PRIMEROS ERRORES DEL CODEC / ROUND-TRIP:"
        )

        for detail in (
            failure_details[:10]
        ):

            print(
                json.dumps(
                    detail,
                    ensure_ascii=False,
                )
            )

        print()

    if statistics_error_details:

        print(
            "PRIMEROS ERRORES DE INSTRUMENTACIÓN:"
        )

        for detail in (
            statistics_error_details[:10]
        ):

            print(
                json.dumps(
                    detail,
                    ensure_ascii=False,
                )
            )

        print()

    # =========================================================================
    # PERSISTENCIA CSV
    # =========================================================================

    if output_csv is not None:

        benchmark_result = {
            "Dictionary": dictionary_path.name,
            "DictionarySize": dictionary.size,
            "FirstToken": dictionary.first_token,
            "LastToken": dictionary.last_token,
            "Dataset": dataset_name,
            "DatasetRole": dataset_role,
            "SnapshotDate": snapshot_date,
            "Segment": segment_name,
            "DocumentsEvaluated": len(records),
            "RoundTripCorrect": successful,
            "RoundTripErrors": roundtrip_failed,
            "CodecExceptions": codec_exceptions,
            "StatisticsErrors": statistics_errors,
            "SourceCharacters": total_characters,
            "UTF8Bytes": total_utf8_bytes,
            "LexiMapBytes": total_encoded_bytes,
            "DifferenceBytes": difference,
            "ReductionPercentage": (
                f"{reduction_percentage:.6f}"
            ),
            "LexicalItems": total_lexical_items,
            "DirectItems": total_direct_items,
            "LiteralItems": total_literal_items,
            "LiteralPayloadBytes": (
                total_literal_payload_bytes
            ),
            "ImplicitSpaces": total_implicit_spaces,
            "ExplicitSingleSpaces": (
                total_explicit_single_spaces
            ),
            "ExplicitWhitespaceItems": (
                total_explicit_whitespace_items
            ),
            "ModifierItems": total_modifier_items,
            "ElapsedSeconds": (
                f"{elapsed:.6f}"
            ),
            "RoundTripSuccessPercentage": (
                f"{roundtrip_success_percentage:.6f}"
            ),
            "Result": final_result,
        }

        try:

            csv_operation = (
                write_benchmark_result(
                    output_csv,
                    benchmark_result,
                )
            )

        except Exception as error:

            print(
                "ERROR AL ESCRIBIR CSV DE RESULTADOS:"
            )

            print(
                f"{type(error).__name__}: {error}"
            )

            return 1

        print(
            "CSV resultados         : "
            f"{output_csv}"
        )

        print(
            "Operación CSV          : "
            f"{csv_operation}"
        )

        print()

    # =========================================================================
    # RESULTADO FINAL
    # =========================================================================

    if (
        codec_success
        and statistics_success
    ):

        print(
            "RESULTADO FINAL: "
            "ROUND-TRIP CORRECTO EN LOS "
            f"{expected_documents:,} DOCUMENTOS HELD-OUT."
        )

        print()

        print(
            "Se cumple experimentalmente:"
        )

        print(
            "decode(encode(text)) == text"
        )

        print(
            f"para {successful:,} de "
            f"{expected_documents:,} documentos."
        )

        print()

        print(
            "Instrumentación estadística: OK"
        )

        return 0

    if codec_success:

        print(
            "RESULTADO FINAL: "
            "ROUND-TRIP CORRECTO, "
            "PERO EXISTEN ERRORES DE INSTRUMENTACIÓN."
        )

        return 1

    print(
        "RESULTADO FINAL: "
        "SE DETECTARON ERRORES DEL CODEC "
        "O DEL ROUND-TRIP."
    )

    return 1


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    raise SystemExit(
        main()
    )

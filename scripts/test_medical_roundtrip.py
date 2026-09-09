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

MEDICAL_ROOT = (
    PROJECT_ROOT
    / "data"
    / "medical"
)

FIRST_LEXICAL_TOKEN = 536

DEFAULT_DATASET_ROLE = "MEDICAL_DOMAIN"


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
            "Valida round-trip de LexiMapSp-16 sobre un dominio médico "
            "materializado como texto UTF-8 usando un diccionario LexiCorpus."
        )
    )

    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Directorio del dominio dentro de data/medical, por ejemplo bioquimica.",
    )
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument(
        "--manifest",
        default="extraction_selected_manifest.csv",
        help="Manifest CSV del dominio.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
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

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        records = list(csv.DictReader(file))

    records = [
        record
        for record in records
        if str(record.get("ExtractionStatus", "")).strip().upper()
        == "EXTRACTED"
    ]

    records.sort(
        key=lambda record: int(record["CandidateOrder"])
    )

    return records


def resolve_text_path(
    domain_directory: Path,
    record: dict,
) -> Path:

    raw = str(record.get("ExtractedTextPath", "")).strip()

    if not raw:
        raise ValueError("Registro EXTRACTED sin ExtractedTextPath.")

    path = Path(raw)

    if path.is_absolute():
        return path

    project_relative = PROJECT_ROOT / path
    if project_relative.exists():
        return project_relative

    domain_relative = domain_directory / path
    if domain_relative.exists():
        return domain_relative

    return domain_directory / "extracted_text" / path.name


def validate_manifest(
    records: list[dict],
    expected_documents: int,
    domain_directory: Path,
) -> None:

    if len(records) != expected_documents:
        raise ValueError(
            "Cantidad incorrecta de documentos EXTRACTED. "
            f"Esperados: {expected_documents:,}; "
            f"encontrados: {len(records):,}."
        )

    candidate_orders: list[int] = []

    for record in records:
        order = int(record["CandidateOrder"])
        candidate_orders.append(order)

        if not str(record.get("TextSHA256", "")).strip():
            raise ValueError(
                f"CandidateOrder {order} no tiene TextSHA256."
            )

        text_path = resolve_text_path(domain_directory, record)
        if not text_path.exists():
            raise FileNotFoundError(
                f"No existe el TXT de CandidateOrder {order}: {text_path}"
            )

    if len(candidate_orders) != len(set(candidate_orders)):
        raise ValueError("CandidateOrder duplicado en el manifiesto.")


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

    dictionary_path = DICTIONARY_DIRECTORY / args.dictionary
    domain_directory = MEDICAL_ROOT / args.dataset
    manifest_path = domain_directory / args.manifest
    output_csv = resolve_output_csv(args.output_csv)

    expected_dictionary_size = args.size
    expected_documents = args.count
    dataset_name = args.dataset

    expected_first_token = FIRST_LEXICAL_TOKEN
    expected_last_token = FIRST_LEXICAL_TOKEN + expected_dictionary_size - 1

    if expected_dictionary_size <= 0 or expected_documents <= 0:
        print("ERROR: --size y --count deben ser mayores que cero.")
        return 1

    if expected_last_token > 65_535:
        print("ERROR: el tamaño solicitado excede el rango léxico de 16 bits.")
        return 1

    print()
    print("=" * 88)
    print("LEXIMAPSP-16 - VALIDACIÓN ROUND-TRIP DOMINIO MÉDICO")
    print("=" * 88)
    print(f"Diccionario            : {dictionary_path.name}")
    print(f"Tamaño esperado        : {expected_dictionary_size:,}")
    print(f"Rango esperado         : {expected_first_token:,}-{expected_last_token:,}")
    print(f"Dataset                : {dataset_name}")
    print(f"Documentos esperados   : {expected_documents:,}")
    print(f"Manifest               : {manifest_path}")
    if output_csv is not None:
        print(f"CSV resultados         : {output_csv}")
    print("-" * 88)

    if not dictionary_path.exists():
        print(f"ERROR: no existe el diccionario:\n{dictionary_path}")
        return 1
    if not domain_directory.exists():
        print(f"ERROR: no existe el dominio:\n{domain_directory}")
        return 1
    if not manifest_path.exists():
        print(f"ERROR: no existe el manifiesto:\n{manifest_path}")
        return 1

    try:
        records = load_manifest(manifest_path)
        validate_manifest(records, expected_documents, domain_directory)
    except Exception as error:
        print("ERROR AL VALIDAR EL MANIFIESTO:")
        print(f"{type(error).__name__}: {error}")
        return 1

    try:
        dictionary = LexiMapDictionary.from_csv(dictionary_path)
    except Exception as error:
        print("ERROR AL CARGAR EL DICCIONARIO:")
        print(f"{type(error).__name__}: {error}")
        return 1

    print(f"Entradas diccionario   : {dictionary.size:,}")
    print(f"Rango real tokens      : {dictionary.first_token:,}-{dictionary.last_token:,}")

    if dictionary.size != expected_dictionary_size:
        print("ERROR: el tamaño real del diccionario no coincide con --size.")
        return 1

    if (
        dictionary.first_token != expected_first_token
        or dictionary.last_token != expected_last_token
    ):
        print("ERROR: el rango real de tokens no coincide con el esperado.")
        return 1

    encoder = LexiMapEncoder(dictionary)
    decoder = LexiMapDecoder(dictionary)

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

    timer_start = perf_counter()

    for document_number, record in enumerate(records, start=1):

        candidate_order = int(record["CandidateOrder"])

        try:
            text_path = resolve_text_path(domain_directory, record)

            # Lee bytes exactos para no alterar saltos de línea al abrir el TXT.
            original_bytes = text_path.read_bytes()
            original_text = original_bytes.decode("utf-8")
            utf8_bytes = len(original_bytes)

            manifest_bytes = int(record.get("ExtractedUTF8Bytes", "0") or 0)
            if manifest_bytes and utf8_bytes != manifest_bytes:
                raise ValueError(
                    f"Bytes TXT ({utf8_bytes}) != manifest ({manifest_bytes})."
                )

            encoding_result = encoder.encode_with_statistics(original_text)
            encoded_data = encoding_result.data

            decoding_result = decoder.decode_with_statistics(encoded_data)
            reconstructed_text = decoding_result.text

        except Exception as error:
            codec_exceptions += 1
            failure_details.append({
                "document_number": document_number,
                "candidate_order": candidate_order,
                "relative_path": record.get("RelativePath", ""),
                "error": "CODEC_EXCEPTION",
                "exception_type": type(error).__name__,
                "message": str(error),
            })
            continue

        if reconstructed_text == original_text:
            successful += 1
        else:
            roundtrip_failed += 1
            difference_detail = find_first_difference(
                original_text,
                reconstructed_text,
            )
            detail = {
                "document_number": document_number,
                "candidate_order": candidate_order,
                "relative_path": record.get("RelativePath", ""),
                "error": "ROUND_TRIP_MISMATCH",
                "original_length": len(original_text),
                "reconstructed_length": len(reconstructed_text),
            }
            if difference_detail is not None:
                index, original_character, reconstructed_character = difference_detail
                detail["first_difference_index"] = index
                detail["original_character"] = original_character
                detail["reconstructed_character"] = reconstructed_character
            failure_details.append(detail)

        total_characters += len(original_text)
        total_utf8_bytes += utf8_bytes
        total_encoded_bytes += len(encoded_data)

        try:
            statistics = encoding_result.statistics
            total_lexical_items += statistics.lexical_items
            total_direct_items += statistics.direct_items
            total_literal_items += statistics.literal_items
            total_literal_payload_bytes += statistics.literal_payload_bytes
            total_implicit_spaces += statistics.implicit_spaces
            total_explicit_single_spaces += statistics.explicit_single_spaces
            total_explicit_whitespace_items += statistics.explicit_whitespace_items
            total_modifier_items += statistics.modifier_items
        except Exception as error:
            statistics_errors += 1
            statistics_error_details.append({
                "document_number": document_number,
                "candidate_order": candidate_order,
                "exception_type": type(error).__name__,
                "message": str(error),
            })

        if document_number % 10 == 0 or document_number == expected_documents:
            elapsed_now = perf_counter() - timer_start
            print(
                "\r"
                f"Procesados: {document_number:,}/{expected_documents:,}"
                f" | OK: {successful:,}"
                f" | Round-trip err: {roundtrip_failed:,}"
                f" | Excepciones: {codec_exceptions:,}"
                f" | Tiempo: {elapsed_now:,.1f}s",
                end="",
                flush=True,
            )

    print()
    elapsed = perf_counter() - timer_start

    difference = total_utf8_bytes - total_encoded_bytes
    reduction_percentage = (
        difference / total_utf8_bytes * 100.0
        if total_utf8_bytes
        else 0.0
    )
    roundtrip_success_percentage = (
        successful / len(records) * 100.0
        if records
        else 0.0
    )

    codec_success = (
        successful == expected_documents
        and roundtrip_failed == 0
        and codec_exceptions == 0
    )
    statistics_success = statistics_errors == 0

    if codec_success and statistics_success:
        final_result = "OK"
    elif codec_success:
        final_result = "STATISTICS_ERROR"
    else:
        final_result = "CODEC_OR_ROUNDTRIP_ERROR"

    print()
    print("=" * 88)
    print("RESULTADO ROUND-TRIP DOMINIO MÉDICO")
    print("-" * 88)
    print(f"Dataset                : {dataset_name}")
    print(f"Diccionario            : {dictionary_path.name}")
    print(f"Tamaño diccionario     : {dictionary.size:,}")
    print(f"Rango tokens           : {dictionary.first_token:,}-{dictionary.last_token:,}")
    print(f"Documentos evaluados   : {len(records):,}")
    print(f"Correctos              : {successful:,}")
    print(f"Errores round-trip     : {roundtrip_failed:,}")
    print(f"Excepciones codec      : {codec_exceptions:,}")
    print(f"Errores estadísticas   : {statistics_errors:,}")
    print(f"Caracteres procesados  : {total_characters:,}")
    print(f"Bytes UTF-8            : {total_utf8_bytes:,}")
    print(f"Bytes LexiMap          : {total_encoded_bytes:,}")
    print(f"Tiempo total           : {format_elapsed(elapsed)}")
    print("-" * 88)
    print(f"Ítems léxicos          : {total_lexical_items:,}")
    print(f"Ítems directos         : {total_direct_items:,}")
    print(f"Ítems literales        : {total_literal_items:,}")
    print(f"Payload literal bytes  : {total_literal_payload_bytes:,}")
    print(f"Espacios implícitos    : {total_implicit_spaces:,}")
    print(f"Espacios simples exp.  : {total_explicit_single_spaces:,}")
    print(f"Whitespace explícito   : {total_explicit_whitespace_items:,}")
    print(f"Modificadores de caso  : {total_modifier_items:,}")
    print("-" * 88)
    print(f"Diferencia bytes       : {difference:+,}")
    print(f"Reducción observada    : {reduction_percentage:.4f}%")
    print(f"Round-trip correcto    : {roundtrip_success_percentage:.4f}%")
    print()

    if failure_details:
        print("PRIMEROS ERRORES DEL CODEC / ROUND-TRIP:")
        for detail in failure_details[:10]:
            print(json.dumps(detail, ensure_ascii=False))
        print()

    if statistics_error_details:
        print("PRIMEROS ERRORES DE INSTRUMENTACIÓN:")
        for detail in statistics_error_details[:10]:
            print(json.dumps(detail, ensure_ascii=False))
        print()

    if output_csv is not None:
        benchmark_result = {
            "Dictionary": dictionary_path.name,
            "DictionarySize": dictionary.size,
            "FirstToken": dictionary.first_token,
            "LastToken": dictionary.last_token,
            "Dataset": dataset_name,
            "DatasetRole": DEFAULT_DATASET_ROLE,
            "DocumentsEvaluated": len(records),
            "RoundTripCorrect": successful,
            "RoundTripErrors": roundtrip_failed,
            "CodecExceptions": codec_exceptions,
            "StatisticsErrors": statistics_errors,
            "SourceCharacters": total_characters,
            "UTF8Bytes": total_utf8_bytes,
            "LexiMapBytes": total_encoded_bytes,
            "DifferenceBytes": difference,
            "ReductionPercentage": f"{reduction_percentage:.6f}",
            "LexicalItems": total_lexical_items,
            "DirectItems": total_direct_items,
            "LiteralItems": total_literal_items,
            "LiteralPayloadBytes": total_literal_payload_bytes,
            "ImplicitSpaces": total_implicit_spaces,
            "ExplicitSingleSpaces": total_explicit_single_spaces,
            "ExplicitWhitespaceItems": total_explicit_whitespace_items,
            "ModifierItems": total_modifier_items,
            "ElapsedSeconds": f"{elapsed:.6f}",
            "RoundTripSuccessPercentage": f"{roundtrip_success_percentage:.6f}",
            "Result": final_result,
        }

        try:
            csv_operation = write_benchmark_result(
                output_csv,
                benchmark_result,
            )
        except Exception as error:
            print("ERROR AL ESCRIBIR CSV DE RESULTADOS:")
            print(f"{type(error).__name__}: {error}")
            return 1

        print(f"CSV resultados         : {output_csv}")
        print(f"Operación CSV          : {csv_operation}")
        print()

    if codec_success and statistics_success:
        print(
            "RESULTADO FINAL: ROUND-TRIP CORRECTO EN LOS "
            f"{expected_documents:,} DOCUMENTOS DEL DOMINIO."
        )
        print()
        print("Se cumple experimentalmente:")
        print("decode(encode(text)) == text")
        print(
            f"para {successful:,} de {expected_documents:,} documentos."
        )
        print()
        print("Instrumentación estadística: OK")
        return 0

    if codec_success:
        print(
            "RESULTADO FINAL: ROUND-TRIP CORRECTO, "
            "PERO EXISTEN ERRORES DE INSTRUMENTACIÓN."
        )
        return 1

    print(
        "RESULTADO FINAL: SE DETECTARON ERRORES DEL CODEC "
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

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


from leximapsp16.medical_dataset import (  # noqa: E402
    MedicalDocument,
    calculate_binary_sha256,
    extract_complete_pdf_text,
    calculate_text_sha256,
    discover_pdfs,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara una especialidad medica para la evaluacion "
            "experimental de LexiMapSp-16."
        )
    )

    parser.add_argument(
        "--specialty-path",
        required=True,
        type=Path,
        help="Ruta completa de la carpeta de la especialidad.",
    )

    parser.add_argument(
        "--specialty-name",
        required=True,
        help="Nombre logico de la especialidad.",
    )

    parser.add_argument(
        "--output-csv",
        required=True,
        type=Path,
        help="Archivo CSV donde se guardara el manifiesto.",
    )

    return parser.parse_args()


def write_manifest(
    documents: list[MedicalDocument],
    duplicate_of: dict[str, str],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "Specialty",
        "RelativePath",
        "FileSizeBytes",
        "BinarySHA256",
        "ExtractedCharacters",
        "ExtractedUTF8Bytes",
        "TextSHA256",
        "ExtractionStatus",
        "DuplicateOf",
        "ExtractionError",
    ]

    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for document in documents:
            writer.writerow(
                {
                    "Specialty": document.specialty,
                    "RelativePath": document.relative_path,
                    "FileSizeBytes": document.file_size_bytes,
                    "BinarySHA256": document.binary_sha256,
                    "ExtractedCharacters": (
                        document.extracted_characters
                    ),
                    "ExtractedUTF8Bytes": (
                        document.extracted_utf8_bytes
                    ),
                    "TextSHA256": document.text_sha256,
                    "ExtractionStatus": (
                        document.extraction_status
                    ),
                    "DuplicateOf": duplicate_of.get(
                        document.relative_path,
                        "",
                    ),
                    "ExtractionError": (
                        document.extraction_error
                    ),
                }
            )


def main() -> int:
    args = parse_arguments()

    specialty_path = args.specialty_path.resolve()

    if not specialty_path.exists():
        print(
            f"ERROR: no existe la ruta: "
            f"{specialty_path}"
        )
        return 1

    if not specialty_path.is_dir():
        print(
            f"ERROR: la ruta no es una carpeta: "
            f"{specialty_path}"
        )
        return 1

    documents = discover_pdfs(
        specialty_path=specialty_path,
        specialty_name=args.specialty_name,
    )

    total = len(documents)

    print()
    print("=" * 80)
    print("PREPARACION DE DATASET MEDICO")
    print("=" * 80)
    print(f"Especialidad : {args.specialty_name}")
    print(f"Ruta         : {specialty_path}")
    print(f"PDF          : {total}")
    print("=" * 80)
    print()

    if total == 0:
        print("No se encontraron archivos PDF.")
        return 1

    # ------------------------------------------------------------
    # FASE 1
    # Calculo de SHA-256 binario para todos los PDF.
    # ------------------------------------------------------------

    print("FASE 1 - HUELLAS BINARIAS")
    print("-" * 80)

    binary_representatives: dict[str, MedicalDocument] = {}
    duplicate_of: dict[str, str] = {}

    binary_hash_errors = 0
    binary_duplicate_files = 0

    for index, document in enumerate(
        documents,
        start=1,
    ):
        print(
            f"[{index:04d}/{total:04d}] "
            f"{document.relative_path}"
        )

        try:
            document.binary_sha256 = (
                calculate_binary_sha256(
                    document.path
                )
            )

        except Exception as error:
            document.extraction_status = "ERROR"
            document.extraction_error = (
                "BINARY_HASH_ERROR - "
                f"{type(error).__name__}: {error}"
            )

            binary_hash_errors += 1

            print(
                "    ERROR SHA-256: "
                f"{document.extraction_error}"
            )

            continue

        representative = binary_representatives.get(
            document.binary_sha256
        )

        if representative is None:
            binary_representatives[
                document.binary_sha256
            ] = document

        else:
            document.extraction_status = (
                "BINARY_DUPLICATE"
            )

            duplicate_of[
                document.relative_path
            ] = representative.relative_path

            binary_duplicate_files += 1

    unique_binary_documents = list(
        binary_representatives.values()
    )

    print()
    print(
        "PDF binariamente unicos : "
        f"{len(unique_binary_documents)}"
    )
    print(
        "Copias exactas omitidas  : "
        f"{binary_duplicate_files}"
    )
    print(
        "Errores SHA-256           : "
        f"{binary_hash_errors}"
    )
    print()

    # ------------------------------------------------------------
    # FASE 2
    # Extraccion completa solamente de documentos binariamente
    # unicos.
    # ------------------------------------------------------------

    print("FASE 2 - EXTRACCION DE TEXTO COMPLETO")
    print("-" * 80)

    extracted = 0
    no_text = 0
    extraction_errors = 0

    unique_total = len(unique_binary_documents)

    for index, document in enumerate(
        unique_binary_documents,
        start=1,
    ):
        print(
            f"[{index:04d}/{unique_total:04d}] "
            f"{document.relative_path}"
        )

        try:
            text = extract_complete_pdf_text(
                document.path
            )

            document.extracted_characters = len(text)
            document.extracted_utf8_bytes = len(
                text.encode("utf-8")
            )

            if not text.strip():
                document.extraction_status = "NO_TEXT"
                no_text += 1
                continue

            document.text_sha256 = (
                calculate_text_sha256(text)
            )

            document.extraction_status = "EXTRACTED"
            extracted += 1

        except Exception as error:
            document.extraction_status = "ERROR"
            document.extraction_error = (
                "EXTRACTION_ERROR - "
                f"{type(error).__name__}: {error}"
            )

            extraction_errors += 1

            print(
                "    ERROR EXTRACCION: "
                f"{document.extraction_error}"
            )

    # ------------------------------------------------------------
    # FASE 3
    # Identificacion de duplicados textuales entre documentos
    # binariamente distintos.
    # ------------------------------------------------------------

    print()
    print("FASE 3 - DUPLICADOS TEXTUALES")
    print("-" * 80)

    text_representatives: dict[
        str,
        MedicalDocument
    ] = {}

    text_duplicate_files = 0

    for document in unique_binary_documents:

        if (
            document.extraction_status != "EXTRACTED"
            or not document.text_sha256
        ):
            continue

        representative = text_representatives.get(
            document.text_sha256
        )

        if representative is None:
            text_representatives[
                document.text_sha256
            ] = document

        else:
            document.extraction_status = (
                "TEXT_DUPLICATE"
            )

            duplicate_of[
                document.relative_path
            ] = representative.relative_path

            text_duplicate_files += 1

            print()
            print("Duplicado textual:")
            print(
                f"    ORIGINAL : "
                f"{representative.relative_path}"
            )
            print(
                f"    DUPLICADO: "
                f"{document.relative_path}"
            )

    # ------------------------------------------------------------
    # FASE 4
    # Escritura del manifiesto completo.
    # ------------------------------------------------------------

    write_manifest(
        documents=documents,
        duplicate_of=duplicate_of,
        output_path=args.output_csv,
    )

    eligible_documents = [
        document
        for document in documents
        if document.extraction_status == "EXTRACTED"
    ]

    print()
    print("=" * 80)
    print("RESULTADO")
    print("=" * 80)
    print(
        f"PDF descubiertos                 : {total}"
    )
    print(
        "PDF binariamente unicos         : "
        f"{len(unique_binary_documents)}"
    )
    print(
        "Copias binarias redundantes     : "
        f"{binary_duplicate_files}"
    )
    print(
        f"Texto extraido                   : {extracted}"
    )
    print(
        f"Sin texto nativo                 : {no_text}"
    )
    print(
        "Errores de extraccion            : "
        f"{extraction_errors}"
    )
    print(
        "Errores calculando SHA-256       : "
        f"{binary_hash_errors}"
    )
    print(
        "Copias textuales redundantes     : "
        f"{text_duplicate_files}"
    )
    print(
        "Documentos candidatos unicos     : "
        f"{len(eligible_documents)}"
    )
    print(
        f"Manifest                         : "
        f"{args.output_csv}"
    )
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

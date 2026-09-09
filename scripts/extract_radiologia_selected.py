from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader


@dataclass
class ExtractionResult:
    CandidateOrder: int
    RelativePath: str
    BinarySHA256: str
    SourceFileBytes: int
    TotalPages: int
    PagesWithText: int
    ExtractedCharacters: int
    ExtractedUTF8Bytes: int
    TextSHA256: str
    ExtractionStatus: str
    ExtractionError: str
    ExtractedTextPath: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extrae texto nativo de los PDFs seleccionados de Radiología. "
            "No utiliza OCR. Los textos extraídos se persisten para que "
            "LexiMapSp-16 utilice exactamente la misma representación."
        )
    )

    parser.add_argument(
        "--selection-csv",
        required=True,
        help="CSV de preselección de Radiología."
    )

    parser.add_argument(
        "--specialty-path",
        required=True,
        help="Ruta raíz de la carpeta RADIOLOGÍA."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directorio donde se almacenarán los TXT extraídos."
    )

    parser.add_argument(
        "--output-manifest",
        required=True,
        help="CSV donde se registrarán los resultados de extracción."
    )

    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def load_selection(selection_csv: Path) -> List[Dict[str, str]]:
    with selection_csv.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    selected = [
        row for row in rows
        if row.get("Selected", "").strip().upper() == "YES"
    ]

    selected.sort(
        key=lambda row: int(row["CandidateOrder"])
    )

    return selected


def load_completed(manifest_path: Path) -> Dict[int, Dict[str, str]]:
    if not manifest_path.exists():
        return {}

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    completed: Dict[int, Dict[str, str]] = {}

    for row in rows:
        try:
            order = int(row["CandidateOrder"])
            completed[order] = row
        except Exception:
            continue

    return completed


def write_manifest(
    manifest_path: Path,
    results: List[ExtractionResult]
) -> None:

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(
        ExtractionResult.__dataclass_fields__.keys()
    )

    temp_path = manifest_path.with_suffix(
        manifest_path.suffix + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in sorted(
            results,
            key=lambda x: x.CandidateOrder
        ):
            writer.writerow(asdict(result))

    # En recursos SMB/NAS, os.replace() puede fallar transitoriamente
    # con WinError 5 aunque ambos archivos sean accesibles. Reintentamos
    # primero la sustitución atómica y, si el NAS sigue bloqueándola,
    # hacemos una copia directa del temporal al manifest existente.
    replace_error: Exception | None = None

    for attempt in range(1, 6):
        try:
            os.replace(temp_path, manifest_path)
            return
        except PermissionError as exc:
            replace_error = exc
            if attempt < 5:
                time.sleep(1.0 * attempt)

    # Fallback SMB: conserva el .tmp hasta haber copiado y sincronizado
    # correctamente su contenido al manifest.
    try:
        with temp_path.open("rb") as src_file:
            with manifest_path.open("wb") as dst_file:
                shutil.copyfileobj(src_file, dst_file, length=1024 * 1024)
                dst_file.flush()
                os.fsync(dst_file.fileno())

        try:
            temp_path.unlink()
        except OSError:
            pass

    except Exception as fallback_error:
        raise RuntimeError(
            "No fue posible actualizar el manifest después de 5 reintentos "
            f"de reemplazo ({replace_error}) ni mediante escritura directa "
            f"de respaldo ({fallback_error}). El archivo temporal se conserva "
            f"en: {temp_path}"
        ) from fallback_error


def row_to_result(
    row: Dict[str, str]
) -> ExtractionResult:

    return ExtractionResult(
        CandidateOrder=int(row["CandidateOrder"]),
        RelativePath=row["RelativePath"],
        BinarySHA256=row.get("BinarySHA256", ""),
        SourceFileBytes=int(
            row.get("SourceFileBytes", "0") or 0
        ),
        TotalPages=int(
            row.get("TotalPages", "0") or 0
        ),
        PagesWithText=int(
            row.get("PagesWithText", "0") or 0
        ),
        ExtractedCharacters=int(
            row.get("ExtractedCharacters", "0") or 0
        ),
        ExtractedUTF8Bytes=int(
            row.get("ExtractedUTF8Bytes", "0") or 0
        ),
        TextSHA256=row.get("TextSHA256", ""),
        ExtractionStatus=row.get(
            "ExtractionStatus",
            ""
        ),
        ExtractionError=row.get(
            "ExtractionError",
            ""
        ),
        ExtractedTextPath=row.get(
            "ExtractedTextPath",
            ""
        )
    )


def extract_pdf(
    candidate_order: int,
    relative_path: str,
    binary_sha256: str,
    source_path: Path,
    output_dir: Path
) -> ExtractionResult:

    source_size = 0
    total_pages = 0
    pages_with_text = 0

    try:
        source_size = source_path.stat().st_size

        reader = PdfReader(
            str(source_path),
            strict=False
        )

        total_pages = len(reader.pages)

        page_texts: List[str] = []

        for page in reader.pages:

            extracted = page.extract_text()

            if extracted is None:
                extracted = ""

            if extracted.strip():
                pages_with_text += 1

            page_texts.append(extracted)

        # Separador determinista entre páginas.
        #
        # Este texto persistido será exactamente el que
        # posteriormente deberá consumir LexiMapSp-16.
        full_text = "\n".join(page_texts)

        utf8_data = full_text.encode("utf-8")

        extracted_characters = len(full_text)
        extracted_utf8_bytes = len(utf8_data)

        if not full_text.strip():

            return ExtractionResult(
                CandidateOrder=candidate_order,
                RelativePath=relative_path,
                BinarySHA256=binary_sha256,
                SourceFileBytes=source_size,
                TotalPages=total_pages,
                PagesWithText=0,
                ExtractedCharacters=0,
                ExtractedUTF8Bytes=0,
                TextSHA256="",
                ExtractionStatus="NO_TEXT",
                ExtractionError="",
                ExtractedTextPath=""
            )

        text_hash = sha256_bytes(utf8_data)

        file_name = (
            f"{candidate_order:04d}_"
            f"{text_hash[:16]}.txt"
        )

        output_path = output_dir / file_name

        output_path.write_bytes(utf8_data)

        return ExtractionResult(
            CandidateOrder=candidate_order,
            RelativePath=relative_path,
            BinarySHA256=binary_sha256,
            SourceFileBytes=source_size,
            TotalPages=total_pages,
            PagesWithText=pages_with_text,
            ExtractedCharacters=extracted_characters,
            ExtractedUTF8Bytes=extracted_utf8_bytes,
            TextSHA256=text_hash,
            ExtractionStatus="EXTRACTED",
            ExtractionError="",
            ExtractedTextPath=str(output_path)
        )

    except Exception as exc:

        return ExtractionResult(
            CandidateOrder=candidate_order,
            RelativePath=relative_path,
            BinarySHA256=binary_sha256,
            SourceFileBytes=source_size,
            TotalPages=total_pages,
            PagesWithText=pages_with_text,
            ExtractedCharacters=0,
            ExtractedUTF8Bytes=0,
            TextSHA256="",
            ExtractionStatus="ERROR",
            ExtractionError=(
                f"{type(exc).__name__}: {exc}"
            ),
            ExtractedTextPath=""
        )


def print_summary(
    results: List[ExtractionResult],
    expected: int,
    manifest_path: Path,
    output_dir: Path
) -> None:

    extracted = sum(
        1 for r in results
        if r.ExtractionStatus == "EXTRACTED"
    )

    no_text = sum(
        1 for r in results
        if r.ExtractionStatus == "NO_TEXT"
    )

    errors = sum(
        1 for r in results
        if r.ExtractionStatus == "ERROR"
    )

    total_chars = sum(
        r.ExtractedCharacters
        for r in results
        if r.ExtractionStatus == "EXTRACTED"
    )

    total_utf8 = sum(
        r.ExtractedUTF8Bytes
        for r in results
        if r.ExtractionStatus == "EXTRACTED"
    )

    print()
    print("=" * 80)
    print("EXTRACCION RADIOLOGIA - RESULTADO FINAL")
    print("=" * 80)

    print(
        f"PDF seleccionados       : {expected}"
    )

    print(
        f"Procesados registrados  : {len(results)}"
    )

    print(
        f"Texto extraido          : {extracted}"
    )

    print(
        f"Sin texto nativo        : {no_text}"
    )

    print(
        f"Errores de extraccion   : {errors}"
    )

    print(
        f"Caracteres extraidos    : {total_chars:,}"
    )

    print(
        f"Bytes UTF-8 extraidos   : {total_utf8:,}"
    )

    print(
        f"Directorio textos       : {output_dir}"
    )

    print(
        f"Manifest                : {manifest_path}"
    )

    print("=" * 80)


def main() -> int:

    args = parse_args()

    selection_csv = Path(
        args.selection_csv
    )

    specialty_path = Path(
        args.specialty_path
    )

    output_dir = Path(
        args.output_dir
    )

    manifest_path = Path(
        args.output_manifest
    )

    if not selection_csv.exists():
        print(
            f"ERROR: no existe el CSV de selección:\n"
            f"{selection_csv}"
        )
        return 1

    if not specialty_path.exists():
        print(
            f"ERROR: no existe la carpeta de Radiología:\n"
            f"{specialty_path}"
        )
        return 1

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    selected = load_selection(
        selection_csv
    )

    print()
    print("=" * 80)
    print("EXTRACCION NATIVA DE RADIOLOGIA")
    print("=" * 80)

    print(
        f"Seleccionados            : {len(selected)}"
    )

    print(
        f"Ruta origen              : {specialty_path}"
    )

    print(
        f"Directorio de salida     : {output_dir}"
    )

    print(
        f"Manifest                 : {manifest_path}"
    )

    print(
        "OCR                      : NO"
    )

    print("=" * 80)
    print()

    completed_rows = load_completed(
        manifest_path
    )

    results: List[ExtractionResult] = [
        row_to_result(row)
        for row in completed_rows.values()
    ]

    completed_orders = {
        result.CandidateOrder
        for result in results
    }

    if completed_orders:

        print(
            f"Reanudación detectada: "
            f"{len(completed_orders)} documentos "
            f"ya registrados."
        )

        print()

    total = len(selected)

    for position, row in enumerate(
        selected,
        start=1
    ):

        candidate_order = int(
            row["CandidateOrder"]
        )

        if candidate_order in completed_orders:

            print(
                f"[{position:4d}/{total}] "
                f"SKIP {candidate_order:4d} "
                f"(ya procesado)"
            )

            continue

        relative_path = row[
            "RelativePath"
        ]

        binary_sha256 = (
            row.get("SHA256", "")
            or row.get("BinarySHA256", "")
        )

        source_path = (
            specialty_path /
            Path(relative_path)
        )

        print(
            f"[{position:4d}/{total}] "
            f"{candidate_order:4d} "
            f"{relative_path}"
        )

        if not source_path.exists():

            result = ExtractionResult(
                CandidateOrder=candidate_order,
                RelativePath=relative_path,
                BinarySHA256=binary_sha256,
                SourceFileBytes=0,
                TotalPages=0,
                PagesWithText=0,
                ExtractedCharacters=0,
                ExtractedUTF8Bytes=0,
                TextSHA256="",
                ExtractionStatus="ERROR",
                ExtractionError=(
                    "FileNotFoundError: "
                    "archivo fuente no encontrado"
                ),
                ExtractedTextPath=""
            )

        else:

            result = extract_pdf(
                candidate_order=candidate_order,
                relative_path=relative_path,
                binary_sha256=binary_sha256,
                source_path=source_path,
                output_dir=output_dir
            )

        results.append(result)

        completed_orders.add(
            candidate_order
        )

        # Checkpoint después de cada PDF.
        #
        # Si la ejecución se interrumpe, el script puede
        # reiniciarse sin volver a procesar lo terminado.
        write_manifest(
            manifest_path,
            results
        )

        if result.ExtractionStatus == "EXTRACTED":

            print(
                f"             -> EXTRACTED | "
                f"{result.TotalPages} páginas | "
                f"{result.ExtractedUTF8Bytes:,} bytes UTF-8"
            )

        elif result.ExtractionStatus == "NO_TEXT":

            print(
                "             -> NO_TEXT"
            )

        else:

            print(
                f"             -> ERROR | "
                f"{result.ExtractionError}"
            )

    print_summary(
        results=results,
        expected=len(selected),
        manifest_path=manifest_path,
        output_dir=output_dir
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
    
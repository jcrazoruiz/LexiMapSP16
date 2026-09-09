from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class MedicalDocument:
    specialty: str
    path: Path
    relative_path: str
    file_size_bytes: int
    binary_sha256: str = ""
    extracted_characters: int = 0
    extracted_utf8_bytes: int = 0
    text_sha256: str = ""
    extraction_status: str = "PENDING"
    extraction_error: str = ""


def discover_pdfs(
    specialty_path: Path,
    specialty_name: str,
) -> list[MedicalDocument]:
    """
    Descubre todos los PDF de una especialidad de forma deterministica.
    """

    pdfs = sorted(
        (
            path
            for path in specialty_path.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".pdf"
        ),
        key=lambda path: str(path).casefold(),
    )

    return [
        MedicalDocument(
            specialty=specialty_name,
            path=path,
            relative_path=str(path.relative_to(specialty_path)),
            file_size_bytes=path.stat().st_size,
        )
        for path in pdfs
    ]


def calculate_binary_sha256(path: Path) -> str:
    """
    Calcula SHA-256 sobre los bytes originales del archivo.
    """

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)

    return digest.hexdigest()


def extract_complete_pdf_text(path: Path) -> str:
    """
    Extrae el texto nativo completo de todas las paginas del PDF.

    No realiza OCR.
    """

    reader = PdfReader(str(path))

    pages: list[str] = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def normalize_text_for_duplicate_detection(text: str) -> str:
    """
    Normalizacion utilizada exclusivamente para detectar documentos
    textualmente identicos.

    Esta representacion NO es el texto que posteriormente recibira LexiMap.
    """

    text = unicodedata.normalize("NFC", text)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


def calculate_text_sha256(text: str) -> str:
    """
    Calcula SHA-256 sobre la representacion normalizada utilizada
    exclusivamente para deteccion de duplicados textuales.
    """

    normalized = normalize_text_for_duplicate_detection(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def analyze_document(document: MedicalDocument) -> tuple[MedicalDocument, str]:
    """
    Analiza un PDF completo.

    Devuelve:
        MedicalDocument actualizado
        texto original extraido

    Los errores de lectura/extraccion se registran como problemas de la
    fuente documental, nunca como fallos de LexiMapSp-16.
    """

    try:
        document.binary_sha256 = calculate_binary_sha256(document.path)

        text = extract_complete_pdf_text(document.path)

        document.extracted_characters = len(text)
        document.extracted_utf8_bytes = len(text.encode("utf-8"))

        if not text.strip():
            document.extraction_status = "NO_TEXT"
            return document, text

        document.text_sha256 = calculate_text_sha256(text)
        document.extraction_status = "EXTRACTED"

        return document, text

    except Exception as error:
        document.extraction_status = "ERROR"
        document.extraction_error = (
            f"{type(error).__name__}: {error}"
        )

        return document, ""


def group_by_binary_hash(
    documents: list[MedicalDocument],
) -> dict[str, list[MedicalDocument]]:
    """
    Agrupa documentos con contenido binario exactamente identico.
    """

    groups: dict[str, list[MedicalDocument]] = {}

    for document in documents:
        if document.binary_sha256:
            groups.setdefault(
                document.binary_sha256,
                [],
            ).append(document)

    return groups


def group_by_text_hash(
    documents: list[MedicalDocument],
) -> dict[str, list[MedicalDocument]]:
    """
    Agrupa documentos cuyo texto normalizado es exactamente identico.
    """

    groups: dict[str, list[MedicalDocument]] = {}

    for document in documents:
        if document.text_sha256:
            groups.setdefault(
                document.text_sha256,
                [],
            ).append(document)

    return groups

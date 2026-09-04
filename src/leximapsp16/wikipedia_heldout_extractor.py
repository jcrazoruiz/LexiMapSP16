from __future__ import annotations

import bz2
import json
import xml.etree.ElementTree as ET

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from leximapsp16.wikipedia_wikitext_cleaner import (
    WikipediaWikitextCleaner,
)


@dataclass(slots=True)
class WikipediaHeldoutExtractionResult:
    pages_inspected: int = 0
    namespace_zero: int = 0
    redirects: int = 0
    empty_wikitext: int = 0
    cleaned_empty: int = 0
    extracted: int = 0
    errors: int = 0


class WikipediaHeldoutExtractor:
    """
    Extrae un conjunto held-out determinista de artículos de Wikipedia.

    El extractor:

    - lee directamente un segmento XML BZ2 de Wikipedia;
    - considera únicamente páginas del namespace 0;
    - excluye redirecciones;
    - descarta páginas sin wikitext;
    - aplica WikipediaWikitextCleaner;
    - descarta textos vacíos después de la limpieza;
    - conserva page_id, título y metadatos básicos;
    - se detiene exactamente al alcanzar target_articles.

    El orden de selección corresponde al orden de aparición de las
    páginas en el segmento del dump. Por tanto, dados el mismo dump,
    cleaner y target_articles, el conjunto resultante es reproducible.
    """

    def __init__(
        self,
        cleaner: WikipediaWikitextCleaner,
        output_directory: Path,
        manifest_path: Path,
        snapshot_date: str,
        target_articles: int = 1_000,
        source_code: str = "wikipedia_es",
    ) -> None:

        if target_articles <= 0:
            raise ValueError(
                "target_articles debe ser mayor que cero."
            )

        self.cleaner = cleaner
        self.output_directory = output_directory
        self.manifest_path = manifest_path
        self.snapshot_date = snapshot_date
        self.target_articles = target_articles
        self.source_code = source_code

    @staticmethod
    def _local_name(
        tag: str,
    ) -> str:

        if "}" in tag:
            return tag.rsplit(
                "}",
                1,
            )[-1]

        return tag

    @classmethod
    def _child_text(
        cls,
        element: ET.Element,
        name: str,
    ) -> str | None:

        for child in element:

            if cls._local_name(
                child.tag
            ) == name:

                return child.text

        return None

    @classmethod
    def _revision_text(
        cls,
        page: ET.Element,
    ) -> str:

        for child in page:

            if cls._local_name(
                child.tag
            ) != "revision":
                continue

            for item in child:

                if cls._local_name(
                    item.tag
                ) == "text":

                    return (
                        item.text
                        or ""
                    )

        return ""

    @classmethod
    def _is_redirect(
        cls,
        page: ET.Element,
    ) -> bool:

        for child in page:

            if cls._local_name(
                child.tag
            ) == "redirect":

                return True

        return False

    @classmethod
    def _page_id(
        cls,
        page: ET.Element,
    ) -> str:

        for child in page:

            if cls._local_name(
                child.tag
            ) == "id":

                return (
                    child.text
                    or ""
                ).strip()

        return ""

    @staticmethod
    def _build_source_url(
        title: str,
    ) -> str:

        normalized_title = (
            title.replace(
                " ",
                "_",
            )
        )

        return (
            "https://es.wikipedia.org/wiki/"
            + normalized_title
        )

    def _prepare_output(
        self,
    ) -> None:
        """
        Prepara el directorio de salida.

        Para proteger la reproducibilidad del conjunto held-out,
        no se permite mezclar una ejecución nueva con resultados
        anteriores.
        """

        self.output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.manifest_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        existing_articles = list(
            self.output_directory.glob(
                "*.txt"
            )
        )

        if existing_articles:
            raise FileExistsError(
                "El directorio held-out ya contiene "
                f"{len(existing_articles):,} archivos TXT. "
                "No se sobrescribirá ni mezclará el conjunto."
            )

        if self.manifest_path.exists():
            raise FileExistsError(
                "El manifiesto held-out ya existe: "
                f"{self.manifest_path}"
            )

    def extract(
        self,
        segment_path: Path,
    ) -> WikipediaHeldoutExtractionResult:

        if not segment_path.exists():
            raise FileNotFoundError(
                f"No existe el segmento: {segment_path}"
            )

        self._prepare_output()

        result = (
            WikipediaHeldoutExtractionResult()
        )

        segment_filename = (
            segment_path.name
        )

        timer_start = (
            perf_counter()
        )

        with (
            bz2.open(
                segment_path,
                "rb",
            ) as input_file,
            self.manifest_path.open(
                "x",
                encoding="utf-8",
            ) as manifest_file,
        ):

            context = ET.iterparse(
                input_file,
                events=("end",),
            )

            for _, element in context:

                if self._local_name(
                    element.tag
                ) != "page":
                    continue

                result.pages_inspected += 1

                namespace = (
                    self._child_text(
                        element,
                        "ns",
                    )
                    or ""
                )

                if namespace != "0":
                    element.clear()
                    continue

                result.namespace_zero += 1

                if self._is_redirect(
                    element
                ):
                    result.redirects += 1
                    element.clear()
                    continue

                page_id = (
                    self._page_id(
                        element
                    )
                )

                title = (
                    self._child_text(
                        element,
                        "title",
                    )
                    or ""
                ).strip()

                wikitext = (
                    self._revision_text(
                        element
                    )
                )

                if not wikitext.strip():
                    result.empty_wikitext += 1
                    element.clear()
                    continue

                try:
                    cleaning_result = (
                        self.cleaner.clean(
                            wikitext
                        )
                    )

                except Exception:
                    result.errors += 1
                    element.clear()
                    continue

                cleaned_text = (
                    cleaning_result.text
                )

                if not cleaned_text.strip():
                    result.cleaned_empty += 1
                    element.clear()
                    continue

                if not page_id:
                    result.errors += 1
                    element.clear()
                    continue

                output_path = (
                    self.output_directory
                    / f"{page_id}.txt"
                )

                output_path.write_text(
                    cleaned_text + "\n",
                    encoding="utf-8",
                )

                metadata = {
                    "selection_order": (
                        result.extracted + 1
                    ),
                    "identifier": (
                        f"wikipedia_es:{page_id}"
                    ),
                    "source_code": (
                        self.source_code
                    ),
                    "page_id": (
                        page_id
                    ),
                    "title": (
                        title
                    ),
                    "snapshot_date": (
                        self.snapshot_date
                    ),
                    "segment": (
                        segment_filename
                    ),
                    "source_url": (
                        self._build_source_url(
                            title
                        )
                    ),
                    "original_wikitext_characters": (
                        cleaning_result
                        .original_characters
                    ),
                    "cleaned_characters": (
                        cleaning_result
                        .cleaned_characters
                    ),
                    "removed_characters": (
                        cleaning_result
                        .removed_characters
                    ),
                    "reduction_ratio": (
                        cleaning_result
                        .reduction_ratio
                    ),
                    "local_path": str(
                        output_path
                    ),
                    "evaluation_role": (
                        "HELD_OUT"
                    ),
                    "acquisition_status": (
                        "EXTRACTED"
                    ),
                }

                manifest_file.write(
                    json.dumps(
                        metadata,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                result.extracted += 1

                element.clear()

                if (
                    result.extracted % 100
                    == 0
                ):
                    elapsed = (
                        perf_counter()
                        - timer_start
                    )

                    print(
                        "\r"
                        "Extraídos: "
                        f"{result.extracted:,}"
                        "/"
                        f"{self.target_articles:,}"
                        " | Páginas inspeccionadas: "
                        f"{result.pages_inspected:,}"
                        " | Redirects: "
                        f"{result.redirects:,}"
                        " | Errores: "
                        f"{result.errors:,}"
                        " | Tiempo: "
                        f"{elapsed:,.1f}s",
                        end="",
                        flush=True,
                    )

                if (
                    result.extracted
                    >= self.target_articles
                ):
                    break

        print()

        if (
            result.extracted
            != self.target_articles
        ):
            raise RuntimeError(
                "No fue posible completar el conjunto held-out. "
                f"Solicitados: {self.target_articles:,}; "
                f"extraídos: {result.extracted:,}."
            )

        return result
from __future__ import annotations

import html
import re
from dataclasses import dataclass

import mwparserfromhell


@dataclass(slots=True)
class WikipediaCleaningResult:
    original_characters: int
    cleaned_characters: int
    removed_characters: int
    reduction_ratio: float
    text: str


class WikipediaWikitextCleaner:
    """
    Limpia wikitext procedente de Wikipedia preservando,
    en la medida de lo posible, el contenido lingüístico
    visible.

    Responsabilidades:
    - eliminar estructuras propias de MediaWiki;
    - preservar texto visible de enlaces y determinadas
      plantillas lingüísticamente relevantes;
    - retirar referencias, archivos, categorías, tablas,
      galerías y metadatos auxiliares;
    - entregar texto pre-normalizado al pipeline común
      de LexiCorpus.

    La normalización lingüística definitiva no pertenece
    a este componente.
    """

    def clean(
        self,
        wikitext: str,
    ) -> WikipediaCleaningResult:

        original = (
            wikitext
            or ""
        )

        if not original:
            return WikipediaCleaningResult(
                original_characters=0,
                cleaned_characters=0,
                removed_characters=0,
                reduction_ratio=0.0,
                text="",
            )

        # -----------------------------------------------------
        # Las tablas MediaWiki {| ... |} no son interpretadas
        # como nodos independientes por mwparserfromhell de la
        # misma manera que tags/templates.
        #
        # Se eliminan antes del parseo mediante balanceo.
        # -----------------------------------------------------

        preprocessed = (
            self._remove_mediawiki_tables(
                original
            )
        )

        try:
            code = (
                mwparserfromhell.parse(
                    preprocessed
                )
            )

            self._remove_comments(
                code
            )

            self._remove_references(
                code
            )

            self._remove_non_text_tags(
                code
            )

            self._process_templates(
                code
            )

            self._process_wikilinks(
                code
            )

            self._process_external_links(
                code
            )

            text = code.strip_code(
                normalize=True,
                collapse=True,
            )

        except Exception:
            # -------------------------------------------------
            # Fallback conservador:
            # nunca destruir silenciosamente un artículo.
            # -------------------------------------------------

            text = preprocessed

        text = html.unescape(
            text
        )

        text = self._postprocess_text(
            text
        )

        original_characters = len(
            original
        )

        cleaned_characters = len(
            text
        )

        removed_characters = max(
            0,
            original_characters
            - cleaned_characters,
        )

        reduction_ratio = (
            removed_characters
            / original_characters
            if original_characters
            else 0.0
        )

        return WikipediaCleaningResult(
            original_characters=(
                original_characters
            ),
            cleaned_characters=(
                cleaned_characters
            ),
            removed_characters=(
                removed_characters
            ),
            reduction_ratio=(
                reduction_ratio
            ),
            text=text,
        )

    # =========================================================
    # Comentarios
    # =========================================================

    @staticmethod
    def _remove_comments(
        code,
    ) -> None:

        for node in list(
            code.filter_comments(
                recursive=True
            )
        ):
            try:
                code.remove(
                    node,
                    recursive=True,
                )

            except ValueError:
                pass

    # =========================================================
    # Referencias
    # =========================================================

    @staticmethod
    def _remove_references(
        code,
    ) -> None:

        for tag in list(
            code.filter_tags(
                recursive=True
            )
        ):
            tag_name = (
                str(
                    tag.tag
                )
                .strip()
                .lower()
            )

            if tag_name in {
                "ref",
                "references",
            }:
                try:
                    code.remove(
                        tag,
                        recursive=True,
                    )

                except ValueError:
                    pass

    # =========================================================
    # Tags no lingüísticos
    # =========================================================

    @staticmethod
    def _remove_non_text_tags(
        code,
    ) -> None:

        removable_tags = {
            "gallery",
            "math",
            "source",
            "syntaxhighlight",
            "timeline",
            "imagemap",
            "score",
            "mapframe",
            "maplink",
        }

        for tag in list(
            code.filter_tags(
                recursive=True
            )
        ):
            tag_name = (
                str(
                    tag.tag
                )
                .strip()
                .lower()
            )

            if (
                tag_name
                in removable_tags
            ):
                try:
                    code.remove(
                        tag,
                        recursive=True,
                    )

                except ValueError:
                    pass

    # =========================================================
    # Plantillas
    # =========================================================

    def _process_templates(
        self,
        code,
    ) -> None:

        # Se procesan de forma recursiva.
        #
        # Repetimos varias pasadas porque reemplazar una
        # plantilla puede exponer otra estructura que estaba
        # contenida dentro de ella.

        for _ in range(5):

            templates = list(
                code.filter_templates(
                    recursive=True
                )
            )

            if not templates:
                break

            changed = False

            # Procesar primero las más internas reduce
            # interacciones entre plantillas anidadas.

            for template in reversed(
                templates
            ):
                replacement = (
                    self._template_to_text(
                        template
                    )
                )

                try:
                    code.replace(
                        template,
                        replacement,
                        recursive=True,
                    )

                    changed = True

                except ValueError:
                    pass

            if not changed:
                break

    def _template_to_text(
        self,
        template,
    ) -> str:

        name = (
            str(
                template.name
            )
            .strip()
            .lower()
        )

        # -----------------------------------------------------
        # Plantillas auxiliares que NO deben aportar texto.
        # -----------------------------------------------------

        auxiliary_templates = {
            "rp",
            "refn",
            "harvnp",
            "harvtxt",
            "sfn",
            "sfnp",
            "sfnref",
            "listaref",
            "referencias",
            "control de autoridades",
            "commonscat",
            "commons category",
            "portal",
            "portal asociado",
            "otros usos",
            "otro uso",
            "redirige aquí",
            "redirige aqui",
            "distinguir",
            "desambiguación",
            "desambiguacion",
            "wikcionario",
            "wikisource",
            "wikiquote",
            "wikilibros",
            "wikiviajes",
        }

        if (
            name
            in auxiliary_templates
        ):
            return ""

        # -----------------------------------------------------
        # Fichas e infoboxes.
        # -----------------------------------------------------

        if (
            name.startswith(
                "ficha"
            )
            or name.startswith(
                "infobox"
            )
        ):
            return ""

        # -----------------------------------------------------
        # Citas bibliográficas.
        # -----------------------------------------------------

        if (
            name.startswith(
                "cita "
            )
            or name.startswith(
                "cita_"
            )
            or name.startswith(
                "cite "
            )
            or name.startswith(
                "cite_"
            )
        ):
            return ""

        # -----------------------------------------------------
        # Plantillas de idioma / transliteración.
        #
        # Ejemplos:
        # {{lang|la|agri}}
        # {{nombre original|ca|Principat d'Andorra}}
        # -----------------------------------------------------

        if name in {
            "lang",
            "lang-es",
            "lang-en",
            "lang-la",
            "nombre original",
            "transl",
            "transliteration",
        }:
            values = (
                self._parameter_values(
                    template
                )
            )

            if values:
                return values[-1]

            return ""

        # -----------------------------------------------------
        # Pronunciación.
        # -----------------------------------------------------

        if name in {
            "ipa",
            "afi",
        }:
            values = (
                self._parameter_values(
                    template
                )
            )

            return " ".join(
                values
            )

        # -----------------------------------------------------
        # Siglos.
        # -----------------------------------------------------

        if name in {
            "siglo",
            "siglo2",
        }:
            values = (
                self._parameter_values(
                    template
                )
            )

            if values:
                return (
                    f"siglo {values[0]}"
                )

            return ""

        # -----------------------------------------------------
        # Números y unidades.
        # -----------------------------------------------------

        if name in {
            "unidad",
            "convertir",
            "formatnum",
            "esd",
        }:
            values = (
                self._parameter_values(
                    template
                )
            )

            return " ".join(
                values
            )

        # -----------------------------------------------------
        # Fechas.
        # -----------------------------------------------------

        if name in {
            "fecha",
            "fecha de inicio",
            "fecha de fin",
        }:
            values = (
                self._parameter_values(
                    template
                )
            )

            return " ".join(
                values
            )

        # -----------------------------------------------------
        # Plantillas simples con un solo valor.
        #
        # Regla conservadora:
        # si existe únicamente un valor textual, se conserva.
        # -----------------------------------------------------

        values = (
            self._parameter_values(
                template
            )
        )

        if len(values) == 1:
            return values[0]

        return ""

    @staticmethod
    def _parameter_values(
        template,
    ) -> list[str]:

        values: list[str] = []

        for parameter in (
            template.params
        ):
            value = (
                str(
                    parameter.value
                )
                .strip()
            )

            if value:
                values.append(
                    value
                )

        return values

    # =========================================================
    # Wikilinks
    # =========================================================

    @staticmethod
    def _process_wikilinks(
        code,
    ) -> None:

        for link in list(
            code.filter_wikilinks(
                recursive=True
            )
        ):
            title = (
                str(
                    link.title
                )
                .strip()
            )

            normalized_title = (
                title.lower()
            )

            # -------------------------------------------------
            # Archivos y categorías no forman parte
            # del cuerpo lingüístico.
            # -------------------------------------------------

            if normalized_title.startswith(
                (
                    "archivo:",
                    "file:",
                    "imagen:",
                    "image:",
                    "categoría:",
                    "categoria:",
                    "category:",
                )
            ):
                replacement = ""

            else:
                if link.text:
                    replacement = (
                        str(
                            link.text
                        )
                        .strip()
                    )

                else:
                    replacement = (
                        title
                        .split(
                            "#",
                            1,
                        )[0]
                        .strip()
                    )

            try:
                code.replace(
                    link,
                    replacement,
                    recursive=True,
                )

            except ValueError:
                pass

    # =========================================================
    # Enlaces externos
    # =========================================================

    @staticmethod
    def _process_external_links(
        code,
    ) -> None:

        for link in list(
            code.filter_external_links(
                recursive=True
            )
        ):
            if link.title:
                replacement = (
                    str(
                        link.title
                    )
                    .strip()
                )

            else:
                replacement = ""

            try:
                code.replace(
                    link,
                    replacement,
                    recursive=True,
                )

            except ValueError:
                pass

    # =========================================================
    # Tablas MediaWiki
    # =========================================================

    @staticmethod
    def _remove_mediawiki_tables(
        text: str,
    ) -> str:
        """
        Elimina estructuras:

            {| ...
               ...
            |}

        soportando tablas anidadas.

        Si una tabla queda sin cerrar, la eliminación se
        limita hasta el final del documento. El fallback
        posterior evita que un error destruya silenciosamente
        el artículo completo.
        """

        output: list[str] = []

        index = 0

        length = len(
            text
        )

        depth = 0

        while index < length:

            if text.startswith(
                "{|",
                index,
            ):
                depth += 1

                index += 2

                continue

            if (
                depth > 0
                and text.startswith(
                    "|}",
                    index,
                )
            ):
                depth -= 1

                index += 2

                if depth == 0:
                    output.append(
                        "\n"
                    )

                continue

            if depth == 0:
                output.append(
                    text[index]
                )

            index += 1

        # -----------------------------------------------------
        # Protección frente a tabla aparentemente desbalanceada.
        #
        # Si quedó profundidad > 0, volvemos al contenido
        # original. Es preferible conservar ruido antes que
        # destruir el artículo.
        # -----------------------------------------------------

        if depth != 0:
            return text

        return "".join(
            output
        )

    # =========================================================
    # Postprocesamiento
    # =========================================================

    @staticmethod
    def _postprocess_text(
        text: str,
    ) -> str:

        text = (
            text
            .replace(
                "\r\n",
                "\n",
            )
            .replace(
                "\r",
                "\n",
            )
        )

        # -----------------------------------------------------
        # Residuos defensivos.
        # -----------------------------------------------------

        text = re.sub(
            r"\[\[|\]\]",
            " ",
            text,
        )

        text = re.sub(
            r"'{2,5}",
            "",
            text,
        )

        # -----------------------------------------------------
        # Viñetas y numeraciones MediaWiki.
        #
        # Conservamos el contenido textual pero quitamos
        # los marcadores.
        # -----------------------------------------------------

        text = re.sub(
            r"(?m)^\s*[*#:;]+\s*",
            "",
            text,
        )

        # -----------------------------------------------------
        # Encabezados residuales.
        # -----------------------------------------------------

        text = re.sub(
            r"(?m)^\s*={2,6}\s*",
            "",
            text,
        )

        text = re.sub(
            r"\s*={2,6}\s*$",
            "",
            text,
            flags=re.MULTILINE,
        )

        # -----------------------------------------------------
        # Espacios.
        # -----------------------------------------------------

        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        text = re.sub(
            r"\n[ \t]+",
            "\n",
            text,
        )

        text = re.sub(
            r"[ \t]+\n",
            "\n",
            text,
        )

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        lines: list[str] = []

        for line in (
            text.splitlines()
        ):
            line = line.strip()

            if line:
                lines.append(
                    line
                )

        return "\n".join(
            lines
        ).strip()
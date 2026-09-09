from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Revisa un selection.csv medico y agrupa candidatos "
            "potencialmente pertenecientes a la misma obra."
        )
    )

    parser.add_argument(
        "--selection",
        required=True,
        type=Path,
        help="Archivo selection.csv generado previamente.",
    )

    return parser.parse_args()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )

    value = value.upper()

    value = re.sub(
        r"\.PDF$",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"[^A-Z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def remove_edition_and_part_markers(value: str) -> str:
    patterns = [
        r"\b\d+\s*(?:A|RA|TA|NA|VA)?\s*EDICION\b",
        r"\b\d+\s*ED\b",
        r"\bPARTE\s*\d+\b",
        r"\bPART\s*\d+\b",
        r"\bPART_\d+\b",
        r"\bPTE\s*\d+\b",
        r"\b\d+A\b",
        r"\b\d+VA\b",
        r"\b\d+RA\b",
        r"\b\d+TA\b",
        r"\b\d+NA\b",
    ]

    result = value

    for pattern in patterns:
        result = re.sub(
            pattern,
            " ",
            result,
            flags=re.IGNORECASE,
        )

    result = re.sub(
        r"\s+",
        " ",
        result,
    )

    return result.strip()


def build_family_key(relative_path: str) -> str:
    filename = Path(relative_path).name

    normalized = normalize_text(filename)

    normalized = remove_edition_and_part_markers(
        normalized
    )

    tokens = normalized.split()

    stop_tokens = {
        "PDF",
        "ED",
        "EDICION",
        "EDICIÓN",
        "ILUSTRADA",
        "MEDICA",
        "MEDICA",
    }

    tokens = [
        token
        for token in tokens
        if token not in stop_tokens
    ]

    if len(tokens) > 6:
        tokens = tokens[:6]

    return " ".join(tokens)


def read_selection(
    path: Path,
) -> list[dict[str, str]]:

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        return list(
            csv.DictReader(file)
        )


def print_group(
    key: str,
    rows: list[dict[str, str]],
) -> None:

    print()
    print("-" * 80)
    print(f"POSIBLE FAMILIA: {key}")
    print("-" * 80)

    for index, row in enumerate(
        rows,
        start=1,
    ):
        print(
            f"[{index}] "
            f"{row['RelativePath']}"
        )

        print(
            f"    CandidateOrder       : "
            f"{row['CandidateOrder']}"
        )

        print(
            f"    ExtractedCharacters  : "
            f"{row['ExtractedCharacters']}"
        )

        print(
            f"    ExtractedUTF8Bytes   : "
            f"{row['ExtractedUTF8Bytes']}"
        )


def main() -> int:
    args = parse_arguments()

    if not args.selection.exists():
        print(
            f"ERROR: no existe el archivo: "
            f"{args.selection}"
        )
        return 1

    rows = read_selection(
        args.selection
    )

    groups: dict[
        str,
        list[dict[str, str]]
    ] = defaultdict(list)

    for row in rows:
        key = build_family_key(
            row["RelativePath"]
        )

        groups[key].append(row)

    suspicious_groups = [
        (key, group)
        for key, group in groups.items()
        if len(group) > 1
    ]

    suspicious_groups.sort(
        key=lambda item: item[0]
    )

    print()
    print("=" * 80)
    print("REVISION DE POSIBLES OBRAS DUPLICADAS")
    print("=" * 80)
    print(
        f"Candidatos totales          : {len(rows)}"
    )
    print(
        f"Familias sospechosas        : "
        f"{len(suspicious_groups)}"
    )

    for key, group in suspicious_groups:
        print_group(
            key,
            group,
        )

    print()
    print("=" * 80)
    print(
        "Este reporte NO modifica selection.csv."
    )
    print(
        "Los grupos son solamente candidatos para revision."
    )
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
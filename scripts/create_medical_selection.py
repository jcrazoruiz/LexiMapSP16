from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera el manifiesto de seleccion de obras "
            "para un dataset medico."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Manifest CSV generado por prepare_medical_dataset.py.",
    )

    parser.add_argument(
        "--output-csv",
        required=True,
        type=Path,
        help="Archivo selection.csv de salida.",
    )

    parser.add_argument(
        "--exclude",
        nargs="*",
        type=int,
        default=[],
        help=(
            "CandidateOrder que deben quedar excluidos. "
            "Ejemplo: --exclude 4 6 8 46 47"
        ),
    )

    parser.add_argument(
        "--reason",
        default="SAME_WORK_ALTERNATIVE",
        help=(
            "Razon registrada para los candidatos excluidos. "
            "Default: SAME_WORK_ALTERNATIVE"
        ),
    )

    return parser.parse_args()


def normalize_identifier(value: str) -> str:
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
        "_",
        value,
    )

    value = re.sub(
        r"_+",
        "_",
        value,
    )

    return value.strip("_")


def read_manifest(
    manifest_path: Path,
) -> list[dict[str, str]]:

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        return list(csv.DictReader(file))


def create_selection_rows(
    manifest_rows: list[dict[str, str]],
    excluded_orders: set[int],
    exclusion_reason: str,
) -> list[dict[str, str]]:

    selection_rows: list[dict[str, str]] = []

    sequence = 0

    for row in manifest_rows:

        if row.get("ExtractionStatus") != "EXTRACTED":
            continue

        sequence += 1

        relative_path = row["RelativePath"]
        filename = Path(relative_path).name

        proposed_identifier = normalize_identifier(
            filename
        )

        work_id = (
            f"{row['Specialty'].upper()}_"
            f"{sequence:03d}_"
            f"{proposed_identifier[:60]}"
        )

        if sequence in excluded_orders:
            selected = "NO"
            selection_reason = exclusion_reason
            notes = (
                "Excluded during work-level curation; "
                "another representation of the same or "
                "potentially duplicated intellectual work "
                "was retained."
            )
        else:
            selected = "YES"
            selection_reason = "SELECTED"
            notes = ""

        selection_rows.append(
            {
                "Specialty": row["Specialty"],
                "CandidateOrder": str(sequence),
                "WorkID": work_id,
                "RelativePath": relative_path,
                "WorkPart": "1",
                "Selected": selected,
                "SelectionReason": selection_reason,
                "ExtractedCharacters": row[
                    "ExtractedCharacters"
                ],
                "ExtractedUTF8Bytes": row[
                    "ExtractedUTF8Bytes"
                ],
                "BinarySHA256": row[
                    "BinarySHA256"
                ],
                "TextSHA256": row[
                    "TextSHA256"
                ],
                "Notes": notes,
            }
        )

    return selection_rows


def validate_exclusions(
    rows: list[dict[str, str]],
    excluded_orders: set[int],
) -> None:

    valid_orders = {
        int(row["CandidateOrder"])
        for row in rows
    }

    invalid_orders = (
        excluded_orders - valid_orders
    )

    if invalid_orders:
        values = ", ".join(
            str(value)
            for value in sorted(invalid_orders)
        )

        raise ValueError(
            "CandidateOrder inexistentes en la seleccion: "
            f"{values}"
        )


def write_selection(
    rows: list[dict[str, str]],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "Specialty",
        "CandidateOrder",
        "WorkID",
        "RelativePath",
        "WorkPart",
        "Selected",
        "SelectionReason",
        "ExtractedCharacters",
        "ExtractedUTF8Bytes",
        "BinarySHA256",
        "TextSHA256",
        "Notes",
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
        writer.writerows(rows)


def main() -> int:
    args = parse_arguments()

    if not args.manifest.exists():
        print(
            f"ERROR: no existe el manifest: "
            f"{args.manifest}"
        )
        return 1

    excluded_orders = set(args.exclude)

    manifest_rows = read_manifest(
        args.manifest
    )

    selection_rows = create_selection_rows(
        manifest_rows=manifest_rows,
        excluded_orders=excluded_orders,
        exclusion_reason=args.reason,
    )

    try:
        validate_exclusions(
            rows=selection_rows,
            excluded_orders=excluded_orders,
        )
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1

    write_selection(
        rows=selection_rows,
        output_path=args.output_csv,
    )

    selected_count = sum(
        1
        for row in selection_rows
        if row["Selected"] == "YES"
    )

    excluded_count = sum(
        1
        for row in selection_rows
        if row["Selected"] == "NO"
    )

    print()
    print("=" * 80)
    print("MANIFIESTO DE SELECCION MEDICA")
    print("=" * 80)
    print(
        f"Manifest origen       : {args.manifest}"
    )
    print(
        f"Candidatos extraibles : {len(selection_rows)}"
    )
    print(
        f"Seleccionados         : {selected_count}"
    )
    print(
        f"Excluidos por curacion: {excluded_count}"
    )
    print(
        f"Selection CSV         : {args.output_csv}"
    )
    print("=" * 80)

    if excluded_orders:
        print()
        print(
            "CandidateOrder excluidos: "
            + ", ".join(
                str(value)
                for value in sorted(excluded_orders)
            )
        )
        print(
            f"Razon de exclusion    : {args.reason}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
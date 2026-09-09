from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_PATH),
)

from leximapsp16.decoder import LexiMapDecoder  # noqa: E402
from leximapsp16.dictionary import LexiMapDictionary  # noqa: E402
from leximapsp16.encoder import LexiMapEncoder  # noqa: E402


DICTIONARY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "input"
    / "LexiCorpus_65000_Completo.csv"
)

RANDOM_SEED = 16
RANDOM_CASES = 1_000
RANDOM_LENGTH = 64

REAL_FRAGMENT = (
    "ro?\n(O\x03SURIHVRU\x03%XWWV\x03FD\\y"
    "\x03SRU\x03HO\x03I"
)


def first_difference(
    original: str,
    reconstructed: str,
) -> tuple[int, str | None, str | None] | None:

    maximum = max(
        len(original),
        len(reconstructed),
    )

    for index in range(maximum):
        left = (
            original[index]
            if index < len(original)
            else None
        )
        right = (
            reconstructed[index]
            if index < len(reconstructed)
            else None
        )

        if left != right:
            return index, left, right

    return None


def assert_roundtrip(
    encoder: LexiMapEncoder,
    decoder: LexiMapDecoder,
    text: str,
    label: str,
) -> None:

    encoded = encoder.encode(
        text
    )

    try:
        reconstructed = decoder.decode(
            encoded
        )
    except Exception as error:
        raise AssertionError(
            f"{label}: excepción durante decode; "
            f"original={text!r}; "
            f"HEX={encoded.hex(' ')}; "
            f"error={type(error).__name__}: {error}"
        ) from error

    if reconstructed != text:
        difference = first_difference(
            text,
            reconstructed,
        )

        raise AssertionError(
            f"{label}: round-trip incorrecto. "
            f"Primera diferencia={difference!r}; "
            f"HEX={encoded.hex(' ')}"
        )


def main() -> int:

    print("=" * 78)
    print("VALIDACIÓN EXHAUSTIVA DEL PROTOCOLO LEXIMAPSP-16")
    print("=" * 78)
    print()

    if not DICTIONARY_PATH.exists():
        print(
            f"ERROR: no existe el diccionario: {DICTIONARY_PATH}"
        )
        return 1

    dictionary = LexiMapDictionary.from_csv(
        DICTIONARY_PATH
    )

    encoder = LexiMapEncoder(
        dictionary
    )

    decoder = LexiMapDecoder(
        dictionary
    )

    # -----------------------------------------------------------------
    # 1. Los 256 valores del alfabeto Latin-1 individualmente.
    # -----------------------------------------------------------------
    print("1/4 Valores U+0000..U+00FF individuales...")

    for value in range(256):
        assert_roundtrip(
            encoder,
            decoder,
            chr(value),
            f"U+{value:04X}",
        )

    print("    OK: 256/256")

    print("    Verificando espacio de control 00-0F...")

    assigned_character_controls = {
        0x09: b"\x00\x09",
        0x0A: b"\x00\x0A",
        0x0D: b"\x00\x0D",
    }
    future_reserved = {
        0x04, 0x05, 0x06, 0x07, 0x08,
        0x0B, 0x0C, 0x0E, 0x0F,
    }

    for value in range(0x00, 0x04):
        encoded = encoder.encode(chr(value))
        if not encoded.startswith(b"\x00\x00"):
            raise AssertionError(
                f"U+{value:04X} no fue enrutado a literal: {encoded.hex(' ')}"
            )

    for value, expected in assigned_character_controls.items():
        encoded = encoder.encode(chr(value))
        if encoded != expected:
            raise AssertionError(
                f"U+{value:04X} debía codificarse como {expected.hex(' ')}, "
                f"pero produjo {encoded.hex(' ')}"
            )

    for value in future_reserved:
        encoded = encoder.encode(chr(value))
        if not encoded.startswith(b"\x00\x00"):
            raise AssertionError(
                f"U+{value:04X} invade un código reservado futuro: {encoded.hex(' ')}"
            )

    print("    OK: 09/0A/0D activos y 9 códigos futuros protegidos")

    # -----------------------------------------------------------------
    # 2. Todas las 65,536 parejas posibles.
    # -----------------------------------------------------------------
    print("2/4 Todas las parejas U+0000..U+00FF...")

    pair_count = 0

    for first in range(256):
        for second in range(256):
            text = (
                chr(first)
                + chr(second)
            )

            assert_roundtrip(
                encoder,
                decoder,
                text,
                (
                    f"U+{first:04X} "
                    f"U+{second:04X}"
                ),
            )

            pair_count += 1

    print(
        f"    OK: {pair_count:,}/{pair_count:,}"
    )

    # -----------------------------------------------------------------
    # 3. Patrones adversariales del framing y fragmento real.
    # -----------------------------------------------------------------
    print("3/4 Patrones adversariales...")

    adversarial = [
        "\x00",
        "\x01",
        "\x02",
        "\x03",
        "\x00\x00",
        "\x00\x01",
        "\x00\x02",
        "\x00\x03",
        "\x00\x01\x02\x03",
        "".join(chr(value) for value in range(0x00, 0x10)),
        "".join(chr(value) for value in (0x04,0x05,0x06,0x07,0x08,0x0B,0x0C,0x0E,0x0F)),
        "\x03\x00\x01\x03",
        "A\x00\x01B",
        "A\x02B\x03C",
        REAL_FRAGMENT,
    ]

    for index, text in enumerate(
        adversarial,
        start=1,
    ):
        assert_roundtrip(
            encoder,
            decoder,
            text,
            f"adversarial-{index}",
        )

    print(
        f"    OK: {len(adversarial)}/{len(adversarial)}"
    )

    # -----------------------------------------------------------------
    # 4. Secuencias pseudoaleatorias reproducibles.
    # -----------------------------------------------------------------
    print("4/4 Secuencias pseudoaleatorias reproducibles...")

    random_generator = random.Random(
        RANDOM_SEED
    )

    for case in range(
        1,
        RANDOM_CASES + 1,
    ):
        text = "".join(
            chr(
                random_generator.randrange(
                    256
                )
            )
            for _ in range(
                RANDOM_LENGTH
            )
        )

        assert_roundtrip(
            encoder,
            decoder,
            text,
            f"random-{case}",
        )

    print(
        f"    OK: {RANDOM_CASES:,}/{RANDOM_CASES:,}"
    )

    print()
    print("=" * 78)
    print("RESULTADO FINAL: PROTOCOLO REVERSIBLE EN TODAS LAS PRUEBAS EJECUTADAS.")
    print("=" * 78)
    print()
    print("Se cumple:")
    print("decode(encode(text)) == text")
    print()
    print(
        "Cobertura mínima demostrada: 256 valores individuales, "
        "65,536 parejas, patrones adversariales y secuencias aleatorias."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )

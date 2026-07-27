"""Freeze the final NMSQA scale dataset into a reproducible ZIP bundle."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import soundfile as sf


DEFAULT_MANIFEST = Path("data/manifests/scale_nmsqa_final.jsonl")
DEFAULT_OUTPUT = Path("scale_nmsqa_final_dataset.zip")
DEFAULT_INCLUDE = [
    Path("data/scale_nmsqa_manual_review_report.json"),
    Path("data/scale_nmsqa_manual_review_decisions.csv"),
    Path("data/asr_answer_check_report.json"),
    Path("data/asr_answer_check_suspects.csv"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_audio(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing audio: {path}")
    info = sf.info(str(path))
    if info.format != "WAV":
        raise ValueError(f"Expected WAV: {path}")
    if info.samplerate != 16000:
        raise ValueError(f"Expected 16 kHz audio: {path}")
    if info.channels != 1:
        raise ValueError(f"Expected mono audio: {path}")
    if not 0 < info.duration <= 30.000001:
        raise ValueError(f"Expected <=30s audio: {path} ({info.duration:.3f}s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-a", type=int, default=141)
    parser.add_argument("--expected-c", type=int, default=235)
    parser.add_argument("--expected-audio", type=int, default=51)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.manifest)
    counts = Counter(row["category"] for row in rows)
    if counts["A"] != args.expected_a or counts["C"] != args.expected_c:
        raise ValueError(
            "Unexpected manifest counts: "
            f"A={counts['A']}, C={counts['C']} "
            f"(expected A={args.expected_a}, C={args.expected_c})"
        )

    audio_paths = sorted({Path(row["audio_path"]) for row in rows})
    if len(audio_paths) != args.expected_audio:
        raise ValueError(
            f"Expected {args.expected_audio} audio files, found {len(audio_paths)}"
        )
    for path in audio_paths:
        validate_audio(path)

    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.manifest, args.manifest.as_posix())
        for path in DEFAULT_INCLUDE:
            if path.is_file():
                zf.write(path, path.as_posix())
        for path in audio_paths:
            zf.write(path, path.as_posix())

    with zipfile.ZipFile(args.output) as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            raise ValueError(f"Corrupt ZIP member: {bad_member}")

    print(f"Items:       {len(rows)}")
    print(f"A/C:         {counts['A']} / {counts['C']}")
    print(f"Audio files: {len(audio_paths)}")
    print(f"ZIP:         {args.output}")


if __name__ == "__main__":
    main()

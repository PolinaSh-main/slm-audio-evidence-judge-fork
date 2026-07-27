"""Merge NMSQA context segments for the 51 SQuAD 2.0 dev recordings.

Source segment files are not modified. Full recordings are written as
``<recording-prefix>-full.wav``. This script does not apply the 30-second crop.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import soundfile as sf

from select_nmsqa_squad_audio import select_audio_names


DEFAULT_OVERLAP_DIR = Path("data/raw/nmsqa_overlap")
DEFAULT_SOURCE_DIR = Path("data/raw/nmsqa/NMSQA_audio/test_audios")
DEFAULT_OUTPUT_DIR = Path("data/raw/nmsqa_squad_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge NMSQA c-0, c-1, ... segments into 51 full recordings."
    )
    parser.add_argument(
        "--nmsqa-parquet",
        type=Path,
        default=DEFAULT_OVERLAP_DIR / "nmsqa_test.parquet",
    )
    parser.add_argument(
        "--squad-dev",
        type=Path,
        default=DEFAULT_OVERLAP_DIR / "dev-v2.0.json",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def recording_prefix(audio_name: str) -> str:
    stem = Path(audio_name).stem
    prefix, separator, segment_number = stem.rpartition("-c-")
    if not separator or not prefix or not segment_number.isdigit():
        raise ValueError(f"Unexpected NMSQA context filename: {audio_name}")
    return prefix


def segment_number(path: Path) -> int:
    suffix = path.stem.rpartition("-c-")[2]
    if not suffix.isdigit():
        raise ValueError(f"Unexpected NMSQA segment filename: {path.name}")
    return int(suffix)


def find_segments(source_dir: Path, prefix: str) -> list[Path]:
    segments = sorted(
        source_dir.glob(f"{prefix}-c-*.wav"),
        key=segment_number,
    )
    if not segments:
        raise FileNotFoundError(f"No context segments found for {prefix}")

    numbers = [segment_number(path) for path in segments]
    expected = list(range(len(segments)))
    if numbers != expected:
        raise ValueError(
            f"Non-contiguous segments for {prefix}: "
            f"found {numbers}, expected {expected}"
        )
    return segments


def merge_segments(segments: list[Path], destination: Path) -> dict:
    first = sf.info(str(segments[0]))
    expected_frames = 0

    for path in segments:
        info = sf.info(str(path))
        if info.samplerate != first.samplerate:
            raise ValueError(f"Sampling-rate mismatch: {path}")
        if info.channels != first.channels:
            raise ValueError(f"Channel-count mismatch: {path}")
        expected_frames += info.frames

    destination.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(
        str(destination),
        mode="w",
        samplerate=first.samplerate,
        channels=first.channels,
        format="WAV",
        subtype=first.subtype,
    ) as output:
        for path in segments:
            with sf.SoundFile(str(path), mode="r") as source:
                while True:
                    block = source.read(
                        frames=65536,
                        dtype="float32",
                        always_2d=True,
                    )
                    if len(block) == 0:
                        break
                    output.write(block)

    merged = sf.info(str(destination))
    if merged.frames != expected_frames:
        raise ValueError(
            f"Frame-count mismatch for {destination}: "
            f"{merged.frames} != {expected_frames}"
        )

    return {
        "audio_path": destination.as_posix(),
        "segments": [path.as_posix() for path in segments],
        "segment_count": len(segments),
        "duration_seconds": round(merged.duration, 3),
        "sampling_rate": merged.samplerate,
        "channels": merged.channels,
    }


def main() -> None:
    args = parse_args()
    selected_names = select_audio_names(args.nmsqa_parquet, args.squad_dev)
    prefixes = sorted({recording_prefix(name) for name in selected_names})
    if len(prefixes) != 51:
        raise ValueError(f"Expected 51 recordings, found {len(prefixes)}")

    records = []
    for index, prefix in enumerate(prefixes, start=1):
        segments = find_segments(args.source_dir, prefix)
        destination = args.output_dir / f"{prefix}-full.wav"
        record = {
            "recording_id": prefix,
            **merge_segments(segments, destination),
        }
        records.append(record)
        print(
            f"[{index:02d}/51] {destination.name}: "
            f"{record['segment_count']} segments, "
            f"{record['duration_seconds']:.3f}s"
        )

    manifest_path = args.output_dir / "merged_audio_manifest.json"
    manifest_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    durations = [record["duration_seconds"] for record in records]
    print(f"\nMerged recordings: {len(records)}")
    print(f"Minimum duration:  {min(durations):.3f}s")
    print(f"Median duration:   {statistics.median(durations):.3f}s")
    print(f"Maximum duration:  {max(durations):.3f}s")
    print(f"Output:            {args.output_dir}")
    print(f"Manifest:          {manifest_path}")
    print("30-second cropping: not applied")


if __name__ == "__main__":
    main()

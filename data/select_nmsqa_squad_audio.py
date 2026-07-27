"""Select and validate the 51 NMSQA files matched to SQuAD 2.0 dev."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from explore_nmsqa_overlap import load_nmsqa_test, load_squad2


DEFAULT_OVERLAP_DIR = Path("data/raw/nmsqa_overlap")
DEFAULT_SOURCE_DIR = Path("data/raw/nmsqa/NMSQA_audio/test_audios")
DEFAULT_OUTPUT_DIR = Path("data/raw/nmsqa_squad_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy and validate the 51 NMSQA/SQuAD overlap files."
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
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def select_audio_names(
    nmsqa_parquet: Path,
    squad_dev: Path,
) -> set[str]:
    _, nmsqa_contexts = load_nmsqa_test(nmsqa_parquet)
    squad_contexts, _, _ = load_squad2(squad_dev)
    matched_keys = set(nmsqa_contexts) & set(squad_contexts)
    names = {
        Path(audio_path).name
        for key in matched_keys
        for audio_path in nmsqa_contexts[key]["audio_paths"]
    }
    if len(matched_keys) != 48:
        raise ValueError(
            f"Expected 48 matched contexts, found {len(matched_keys)}"
        )
    if len(names) != 51:
        raise ValueError(f"Expected 51 audio files, found {len(names)}")
    return names


def validate_audio(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    if info.frames <= 0:
        raise ValueError(f"Audio has no frames: {path}")
    if info.samplerate <= 0:
        raise ValueError(f"Invalid sampling rate: {path}")
    if info.channels <= 0:
        raise ValueError(f"Invalid channel count: {path}")
    if info.duration <= 0:
        raise ValueError(f"Invalid duration: {path}")
    return info.duration


def main() -> None:
    args = parse_args()
    names = select_audio_names(
        args.nmsqa_parquet,
        args.squad_dev,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        name
        for name in sorted(names)
        if not (args.source_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} source files: {missing[:5]}"
        )

    copied = 0
    reused = 0
    durations: dict[str, float] = {}
    for name in sorted(names):
        source = args.source_dir / name
        destination = args.output_dir / name
        if (
            destination.is_file()
            and destination.stat().st_size == source.stat().st_size
        ):
            reused += 1
        else:
            shutil.copy2(source, destination)
            copied += 1
        durations[name] = validate_audio(destination)

    output_names = {
        path.name
        for path in args.output_dir.iterdir()
        if path.is_file()
    }
    unexpected = sorted(output_names - names)
    if unexpected:
        raise ValueError(
            f"Unexpected files in output directory: {unexpected[:5]}"
        )

    short_files = {
        name: round(duration, 3)
        for name, duration in durations.items()
        if duration < 10
    }
    print(f"Selected contexts: 48")
    print(f"Validated audio files: {len(durations)}")
    print(f"Copied: {copied}")
    print(f"Reused: {reused}")
    print(f"Files under 10 seconds: {len(short_files)}")
    if short_files:
        for name, duration in short_files.items():
            print(f"- {name}: {duration:.3f}s")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()

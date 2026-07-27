"""Package the scale manifest and its 51 WAV files for DataSphere."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import soundfile as sf


DEFAULT_MANIFEST = Path("data/manifests/scale_nmsqa.jsonl")
DEFAULT_OUTPUT = Path("scale_nmsqa_datasphere.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a validated DataSphere ZIP for scale runs."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 380:
        raise ValueError(f"Expected 380 manifest rows, found {len(rows)}")

    audio_paths = sorted({Path(row["audio_path"]) for row in rows})
    if len(audio_paths) != 51:
        raise ValueError(f"Expected 51 unique WAV files, found {len(audio_paths)}")

    for path in audio_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {path}")
        info = sf.info(str(path))
        if (
            info.format != "WAV"
            or info.samplerate != 16000
            or info.channels != 1
            or not 0 < info.duration <= 30.000001
        ):
            raise ValueError(
                f"Invalid audio profile: {path} "
                f"({info.format}, {info.samplerate} Hz, "
                f"{info.channels} ch, {info.duration:.3f}s)"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        archive.write(args.manifest, args.manifest.as_posix())
        for path in audio_paths:
            archive.write(path, path.as_posix())

    with zipfile.ZipFile(args.output) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"Corrupt ZIP member: {bad_member}")
        members = archive.namelist()
    if len(members) != 52:
        raise ValueError(f"Expected 52 ZIP members, found {len(members)}")

    print(f"Manifest rows:  {len(rows)}")
    print(f"Audio files:    {len(audio_paths)}")
    print(f"ZIP members:    {len(members)}")
    print(f"Archive size:   {args.output.stat().st_size / 1024**2:.1f} MiB")
    print(f"Output:         {args.output.resolve()}")


if __name__ == "__main__":
    main()

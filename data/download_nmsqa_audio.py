"""Download and extract the complete NMSQA audio archive."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path


ARCHIVE_URL = (
    "https://huggingface.co/datasets/voidful/NMSQA/resolve/main/"
    "nmsqa_audio.tar.gz"
)
DEFAULT_OUTPUT_DIR = Path("data/raw/nmsqa")
EXPECTED_ARCHIVE_BYTES = 27_208_266_995


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the complete NMSQA audio archive."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download the archive without extracting it.",
    )
    return parser.parse_args()


def download_with_resume(url: str, destination: Path) -> None:
    if (
        destination.is_file()
        and destination.stat().st_size == EXPECTED_ARCHIVE_BYTES
    ):
        print(f"Using complete archive: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "slm-audio-evidence/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
        print(
            "Resuming download at "
            f"{offset / 1024**3:.2f} GiB",
            flush=True,
        )

    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        resumed = response.status == 206
        if offset and not resumed:
            offset = 0
        mode = "ab" if resumed else "wb"
        downloaded = offset
        next_report = (downloaded // 1024**3 + 1) * 1024**3
        with partial.open(mode) as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    print(
                        "Downloaded: "
                        f"{downloaded / 1024**3:.1f} / "
                        f"{EXPECTED_ARCHIVE_BYTES / 1024**3:.1f} GiB",
                        flush=True,
                    )
                    next_report += 1024**3

    actual_size = partial.stat().st_size
    if actual_size != EXPECTED_ARCHIVE_BYTES:
        raise RuntimeError(
            f"Incomplete archive: {actual_size:,} of "
            f"{EXPECTED_ARCHIVE_BYTES:,} bytes. Run the command again "
            "to resume."
        )
    partial.replace(destination)
    print(f"Downloaded archive: {destination}")


def safe_destination(root: Path, member_name: str) -> Path:
    destination = (root / member_name).resolve()
    root_resolved = root.resolve()
    if destination != root_resolved and root_resolved not in destination.parents:
        raise ValueError(f"Unsafe archive path: {member_name}")
    return destination


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    marker = output_dir / ".nmsqa_audio_extract_complete"
    if marker.is_file():
        print(f"Using extracted audio: {output_dir / 'NMSQA_audio'}")
        return

    print(f"Extracting into: {output_dir}", flush=True)
    files_extracted = 0
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            destination = safe_destination(output_dir, member.name)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(
                    f"Unsupported archive member: {member.name}"
                )
            if (
                destination.is_file()
                and destination.stat().st_size == member.size
            ):
                files_extracted += 1
                if files_extracted % 10_000 == 0:
                    print(
                        f"Verified existing files: {files_extracted:,}",
                        flush=True,
                    )
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(
                    f"Cannot read archive member: {member.name}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_suffix(destination.suffix + ".part")
            with partial.open("wb") as output:
                shutil.copyfileobj(source, output)
            partial.replace(destination)
            files_extracted += 1
            if files_extracted % 10_000 == 0:
                print(
                    f"Extracted files: {files_extracted:,}",
                    flush=True,
                )

    marker.write_text("ok\n", encoding="utf-8")
    print(f"Extraction complete: {files_extracted:,} files")


def main() -> None:
    args = parse_args()
    archive_path = args.output_dir / "nmsqa_audio.tar.gz"
    download_with_resume(ARCHIVE_URL, archive_path)
    if not args.download_only:
        extract_archive(archive_path, args.output_dir)
    print(f"Archive: {archive_path}")
    print(f"Audio root: {args.output_dir / 'NMSQA_audio'}")


if __name__ == "__main__":
    main()

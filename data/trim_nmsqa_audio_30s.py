"""Trim the 51 merged NMSQA/SQuAD recordings to at most 30 seconds.

The transcript is truncated proportionally to the retained audio. Category-A
eligibility is checked with SQuAD 2.0 ``answers[0].answer_start``. The script
also reports the stricter condition where the complete answer text is retained.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from explore_nmsqa_overlap import (
    load_nmsqa_test,
    merged_audio_name,
    normalize_context,
)


DEFAULT_OVERLAP_DIR = Path("data/raw/nmsqa_overlap")
DEFAULT_INPUT_DIR = Path("data/raw/nmsqa_squad_test")
DEFAULT_OUTPUT_DIR = Path("data/raw/nmsqa_squad_30s")
DEFAULT_REPORT = Path("data/nmsqa_30s_report.json")
MAX_SECONDS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trim merged NMSQA/SQuAD recordings and audit A coverage."
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
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    return parser.parse_args()


def load_squad_paragraphs(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    paragraphs = {}
    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            key = normalize_context(paragraph["context"])
            if key in paragraphs:
                raise ValueError(f"Duplicate normalized SQuAD context: {key[:80]}")
            paragraphs[key] = paragraph
    return paragraphs


def trim_audio(source: Path, destination: Path, max_seconds: float) -> dict:
    info = sf.info(str(source))
    keep_frames = min(info.frames, math.floor(max_seconds * info.samplerate))
    destination.parent.mkdir(parents=True, exist_ok=True)

    with sf.SoundFile(str(source), mode="r") as input_audio:
        with sf.SoundFile(
            str(destination),
            mode="w",
            samplerate=info.samplerate,
            channels=info.channels,
            format="WAV",
            subtype=info.subtype,
        ) as output_audio:
            remaining = keep_frames
            while remaining:
                block = input_audio.read(
                    frames=min(65536, remaining),
                    dtype="float32",
                    always_2d=True,
                )
                if len(block) == 0:
                    raise ValueError(f"Unexpected end of audio: {source}")
                output_audio.write(block)
                remaining -= len(block)

    trimmed = sf.info(str(destination))
    if trimmed.frames != keep_frames:
        raise ValueError(
            f"Frame-count mismatch for {destination}: "
            f"{trimmed.frames} != {keep_frames}"
        )
    return {
        "full_frames": info.frames,
        "kept_frames": keep_frames,
        "sampling_rate": info.samplerate,
        "full_duration_seconds": info.duration,
        "kept_duration_seconds": trimmed.duration,
    }


def main() -> None:
    args = parse_args()
    if args.max_seconds <= 0:
        raise ValueError("--max-seconds must be positive")

    _, nmsqa_contexts = load_nmsqa_test(args.nmsqa_parquet)
    squad_contexts = load_squad_paragraphs(args.squad_dev)
    matched_keys = sorted(set(nmsqa_contexts) & set(squad_contexts))
    if len(matched_keys) != 48:
        raise ValueError(f"Expected 48 matched contexts, found {len(matched_keys)}")

    recordings_by_context: dict[str, list[dict]] = defaultdict(list)
    recording_rows = []
    for context_key in matched_keys:
        context = squad_contexts[context_key]["context"]
        for source_audio_path in sorted(
            nmsqa_contexts[context_key]["audio_paths"]
        ):
            full_name = merged_audio_name(source_audio_path)
            source = args.input_dir / full_name
            if not source.is_file():
                raise FileNotFoundError(
                    f"Merged recording not found: {source}\n"
                    "Run: python data/merge_nmsqa_squad_audio.py"
                )

            destination = args.output_dir / full_name
            audio = trim_audio(source, destination, args.max_seconds)
            retained_ratio = audio["kept_frames"] / audio["full_frames"]
            transcript_end = min(
                len(context),
                math.floor(len(context) * retained_ratio),
            )
            row = {
                "recording_id": Path(full_name).stem.removesuffix("-full"),
                "audio_path": destination.as_posix(),
                "full_audio_path": source.as_posix(),
                "full_duration_seconds": round(
                    audio["full_duration_seconds"], 3
                ),
                "duration_seconds": round(
                    audio["kept_duration_seconds"], 3
                ),
                "transcript_end": transcript_end,
                "transcript": context[:transcript_end],
            }
            recordings_by_context[context_key].append(row)
            recording_rows.append(row)

    if len(recording_rows) != 51:
        raise ValueError(f"Expected 51 recordings, found {len(recording_rows)}")

    questions = []
    A_total = 0
    A_start_inside = 0
    A_fully_inside = 0
    C_total = 0
    for context_key in matched_keys:
        recordings = recordings_by_context[context_key]
        for qa in squad_contexts[context_key]["qas"]:
            if qa.get("is_impossible", False):
                C_total += 1
                questions.append(
                    {
                        "id": qa["id"],
                        "category": "C",
                        "question": qa["question"],
                        "eligible": True,
                        "compatible_recordings": [
                            row["recording_id"] for row in recordings
                        ],
                    }
                )
                continue

            A_total += 1
            answer = qa["answers"][0]
            answer_start = answer["answer_start"]
            answer_end = answer_start + len(answer["text"])
            start_compatible = [
                row["recording_id"]
                for row in recordings
                if answer_start < row["transcript_end"]
            ]
            full_compatible = [
                row["recording_id"]
                for row in recordings
                if answer_end <= row["transcript_end"]
            ]
            start_inside = bool(start_compatible)
            fully_inside = bool(full_compatible)
            A_start_inside += start_inside
            A_fully_inside += fully_inside
            questions.append(
                {
                    "id": qa["id"],
                    "category": "A",
                    "question": qa["question"],
                    "gold_answer": answer["text"],
                    "answer_start": answer_start,
                    "answer_end": answer_end,
                    "eligible_by_answer_start": start_inside,
                    "eligible_with_full_answer": fully_inside,
                    "compatible_recordings_by_answer_start": start_compatible,
                    "compatible_recordings_with_full_answer": full_compatible,
                }
            )

    report = {
        "parameters": {
            "max_seconds": args.max_seconds,
            "transcript_truncation": "proportional_by_audio_frames",
            "primary_A_rule": "answers[0].answer_start < transcript_end",
        },
        "summary": {
            "contexts": len(matched_keys),
            "recordings": len(recording_rows),
            "A_before_filter": A_total,
            "A_after_answer_start_filter": A_start_inside,
            "A_with_complete_answer_retained": A_fully_inside,
            "C_retained": C_total,
        },
        "recordings": recording_rows,
        "questions": questions,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Trimmed recordings:              {len(recording_rows)}")
    print(f"Maximum duration:                {args.max_seconds:g}s")
    print(f"A before filter:                 {A_total}")
    print(f"A after answer_start filter:     {A_start_inside}")
    print(f"A with complete answer retained: {A_fully_inside}")
    print(f"C retained:                      {C_total}")
    print(f"Output:                          {args.output_dir}")
    print(f"Report:                          {args.report}")


if __name__ == "__main__":
    main()

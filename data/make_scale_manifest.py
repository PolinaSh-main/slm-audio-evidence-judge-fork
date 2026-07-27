"""Build the NMSQA scale manifest from the audited 30-second recordings."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import soundfile as sf


DEFAULT_REPORT = Path("data/nmsqa_30s_report.json")
DEFAULT_OUTPUT = Path("data/manifests/scale_nmsqa.jsonl")
REQUIRED_FIELDS = [
    "id",
    "audio_path",
    "transcript",
    "question",
    "category",
    "subtype",
    "gold_answer",
    "source",
    "generator",
    "verified_by",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build data/manifests/scale_nmsqa.jsonl."
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def validate_audio(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    info = sf.info(str(path))
    if info.format != "WAV":
        raise ValueError(f"Expected WAV audio: {path}")
    if info.samplerate != 16000:
        raise ValueError(f"Expected 16 kHz audio: {path}")
    if info.channels != 1:
        raise ValueError(f"Expected mono audio: {path}")
    if not 0 < info.duration <= 30.000001:
        raise ValueError(f"Expected 0-30 s audio: {path} ({info.duration:.3f}s)")


def choose_recording(
    compatible_ids: list[str],
    usage: Counter[str],
) -> str:
    if not compatible_ids:
        raise ValueError("Question has no compatible 30-second recording")
    return min(compatible_ids, key=lambda item: (usage[item], item))


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    recordings = {
        row["recording_id"]: row
        for row in report["recordings"]
    }
    if len(recordings) != 51:
        raise ValueError(f"Expected 51 recordings, found {len(recordings)}")

    for recording in recordings.values():
        validate_audio(Path(recording["audio_path"]))

    usage: Counter[str] = Counter()
    manifest = []
    skipped_partial_A = 0

    for question in report["questions"]:
        category = question["category"]
        if category == "A":
            if not question["eligible_with_full_answer"]:
                if question["eligible_by_answer_start"]:
                    skipped_partial_A += 1
                continue
            compatible_ids = question[
                "compatible_recordings_with_full_answer"
            ]
            subtype = "stated"
            gold_answer = question["gold_answer"]
        elif category == "C":
            compatible_ids = question["compatible_recordings"]
            subtype = "native-unanswerable"
            gold_answer = "UNANSWERABLE"
        else:
            raise ValueError(f"Unexpected category: {category}")

        recording_id = choose_recording(compatible_ids, usage)
        recording = recordings[recording_id]
        usage[recording_id] += 1

        if category == "A":
            answer_start = question["answer_start"]
            answer_end = question["answer_end"]
            audible_answer = recording["transcript"][
                answer_start:answer_end
            ]
            if audible_answer != gold_answer:
                raise ValueError(
                    f"Gold answer mismatch for {question['id']}: "
                    f"{audible_answer!r} != {gold_answer!r}"
                )

        item = {
            "id": f"sc-nat-{question['id']}",
            "audio_path": recording["audio_path"],
            "transcript": recording["transcript"],
            "question": question["question"],
            "category": category,
            "subtype": subtype,
            "gold_answer": gold_answer,
            "source": "nmsqa",
            "generator": "native-squad2",
            "verified_by": "M2-spotcheck",
        }
        if list(item) != REQUIRED_FIELDS:
            raise ValueError(f"Manifest schema mismatch for {item['id']}")
        manifest.append(item)

    manifest.sort(key=lambda item: item["id"])
    ids = [item["id"] for item in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate manifest IDs")

    A_count = sum(item["category"] == "A" for item in manifest)
    C_count = sum(item["category"] == "C" for item in manifest)
    if A_count != report["summary"]["A_with_complete_answer_retained"]:
        raise ValueError(
            "A count differs from the 30-second report: "
            f"{A_count} != "
            f"{report['summary']['A_with_complete_answer_retained']}"
        )
    if C_count != report["summary"]["C_retained"]:
        raise ValueError(
            "C count differs from the 30-second report: "
            f"{C_count} != {report['summary']['C_retained']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for item in manifest:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Manifest items:              {len(manifest)}")
    print(f"Category A:                 {A_count}")
    print(f"Category C:                 {C_count}")
    print(f"Skipped partial A answers:  {skipped_partial_A}")
    print(f"Audio files used:           {len(usage)}")
    print(f"Output:                     {args.output}")


if __name__ == "__main__":
    main()

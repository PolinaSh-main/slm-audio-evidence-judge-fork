"""Build the 30-second Spoken-SQuAD twin of the NMSQA scale manifest."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from explore_nmsqa_overlap import normalize_context
from trim_nmsqa_audio_30s import trim_audio


DEFAULT_SCALE_MANIFEST = Path("data/manifests/scale_nmsqa.jsonl")
DEFAULT_DATA_CSV = Path("data/generation/data.csv")
DEFAULT_SQUAD_DEV = Path("data/raw/nmsqa_overlap/dev-v2.0.json")
DEFAULT_AUDIO_DIR = Path("data/audio/spoken_squad_twin_30s")
DEFAULT_OUTPUT = Path("data/manifests/scale_tts_twin.jsonl")
MAX_SECONDS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the paired 30-second Spoken-SQuAD twin manifest."
    )
    parser.add_argument(
        "--scale-manifest",
        type=Path,
        default=DEFAULT_SCALE_MANIFEST,
    )
    parser.add_argument("--data-csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--squad-dev", type=Path, default=DEFAULT_SQUAD_DEV)
    parser.add_argument("--audio-dir", type=Path, default=DEFAULT_AUDIO_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_squad_questions(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = {}
    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            context_key = normalize_context(context)
            for qa in paragraph["qas"]:
                questions[qa["id"]] = {
                    "context": context,
                    "context_key": context_key,
                    "qa": qa,
                }
    return questions


def load_pool(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    by_context: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_context[normalize_context(row["transcript"])].append(row)
    return dict(by_context)


def main() -> None:
    args = parse_args()
    scale_rows = load_jsonl(args.scale_manifest)
    squad_questions = load_squad_questions(args.squad_dev)
    pool = load_pool(args.data_csv)

    scale_contexts = {
        squad_questions[row["id"].removeprefix("sc-nat-")]["context_key"]
        for row in scale_rows
    }
    twin_contexts = scale_contexts & set(pool)
    if len(twin_contexts) != 26:
        raise ValueError(f"Expected 26 twin contexts, found {len(twin_contexts)}")

    twin_audio = {}
    for context_key in sorted(twin_contexts):
        candidates = sorted(pool[context_key], key=lambda row: row["id"])
        source_row = candidates[0]
        source_audio = Path(source_row["audio_path"])
        if not source_audio.is_file():
            raise FileNotFoundError(f"TTS audio not found: {source_audio}")
        source_info = sf.info(str(source_audio))
        if (
            source_info.format != "WAV"
            or source_info.samplerate != 16000
            or source_info.channels != 1
        ):
            raise ValueError(
                f"Expected WAV, 16 kHz, mono audio: {source_audio}"
            )

        destination = args.audio_dir / f"{source_row['id']}.wav"
        audio = trim_audio(source_audio, destination, MAX_SECONDS)
        context = source_row["transcript"]
        retained_ratio = audio["kept_frames"] / audio["full_frames"]
        transcript_end = min(
            len(context),
            math.floor(len(context) * retained_ratio),
        )
        twin_audio[context_key] = {
            "audio_path": destination.as_posix(),
            "transcript": context[:transcript_end],
            "transcript_end": transcript_end,
        }

    manifest = []
    skipped_A = []
    for natural_item in scale_rows:
        squad_qid = natural_item["id"].removeprefix("sc-nat-")
        squad = squad_questions.get(squad_qid)
        if squad is None:
            raise ValueError(f"SQuAD question not found: {squad_qid}")
        context_key = squad["context_key"]
        if context_key not in twin_contexts:
            continue

        audio = twin_audio[context_key]
        if natural_item["category"] == "A":
            answer = squad["qa"]["answers"][0]
            answer_start = answer["answer_start"]
            answer_end = answer_start + len(answer["text"])
            if answer_end > audio["transcript_end"]:
                skipped_A.append(squad_qid)
                continue
            audible_answer = audio["transcript"][answer_start:answer_end]
            if audible_answer != natural_item["gold_answer"]:
                raise ValueError(
                    f"Gold answer mismatch for {squad_qid}: "
                    f"{audible_answer!r} != "
                    f"{natural_item['gold_answer']!r}"
                )

        item = dict(natural_item)
        item.update(
            {
                "id": f"sc-tts-{squad_qid}",
                "audio_path": audio["audio_path"],
                "transcript": audio["transcript"],
                "source": "spoken-squad",
            }
        )
        manifest.append(item)

    manifest.sort(key=lambda item: item["id"])
    ids = [item["id"] for item in manifest]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate twin manifest IDs")

    natural_ids = {row["id"] for row in scale_rows}
    missing_pairs = [
        item["id"]
        for item in manifest
        if item["id"].replace("sc-tts-", "sc-nat-", 1) not in natural_ids
    ]
    if missing_pairs:
        raise ValueError(f"Missing natural pairs: {missing_pairs[:5]}")

    A_count = sum(item["category"] == "A" for item in manifest)
    C_count = sum(item["category"] == "C" for item in manifest)
    audio_paths = {item["audio_path"] for item in manifest}
    if len(audio_paths) != 26:
        raise ValueError(f"Expected 26 TTS audio files, used {len(audio_paths)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for item in manifest:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Twin contexts:                 {len(twin_contexts)}")
    print(f"Twin manifest items:           {len(manifest)}")
    print(f"Category A:                    {A_count}")
    print(f"Category C:                    {C_count}")
    print(f"A excluded beyond TTS 30s:     {len(skipped_A)}")
    print(f"TTS audio files used:          {len(audio_paths)}")
    print(f"Audio output:                  {args.audio_dir}")
    print(f"Manifest output:               {args.output}")


if __name__ == "__main__":
    main()

"""Measure how many SQuAD 2.0 dev A/C questions have natural NMSQA audio.

The unit of overlap is a normalized full paragraph. If an NMSQA test paragraph
matches a SQuAD 2.0 dev paragraph, every SQuAD question attached to that
paragraph can reuse the corresponding natural full-context recording.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path


NMSQA_TEST_URL = (
    "https://huggingface.co/datasets/voidful/NMSQA/resolve/main/data/"
    "test-00000-of-00001-e59cc4b2d3e13fe2.parquet?download=true"
)
SQUAD2_DEV_URL = (
    "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/"
    "master/dataset/dev-v2.0.json"
)
SQUAD2_TRAIN_URL = (
    "https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/"
    "master/dataset/train-v2.0.json"
)
DEFAULT_RAW_DIR = Path("data/raw/nmsqa_overlap")
DEFAULT_REPORT = Path("data/nmsqa_overlap_report.json")
DEFAULT_DATA_CSV = Path("data/generation/data.csv")


def merged_audio_name(audio_path: str) -> str:
    stem = Path(audio_path).stem
    prefix, separator, segment_number = stem.rpartition("-c-")
    if not separator or not prefix or not segment_number.isdigit():
        raise ValueError(f"Unexpected NMSQA context filename: {audio_path}")
    return f"{prefix}-full.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count SQuAD 2.0 dev A/C questions covered by NMSQA test audio."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--data-csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument(
        "--ignore-pool",
        action="store_true",
        help="Do not restrict the primary overlap to contexts from data.csv.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/raw/nmsqa"),
        help="Directory containing extracted NMSQA full-context audio.",
    )
    parser.add_argument(
        "--duration-histogram",
        type=Path,
        default=Path("data/nmsqa_duration_histogram.png"),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download source metadata again even when local files exist.",
    )
    return parser.parse_args()


def download(url: str, destination: Path, force: bool, attempts: int = 4) -> None:
    if destination.is_file() and not force:
        print(f"Using cached file: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "slm-audio-evidence/1.0"})

    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with partial.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            partial.replace(destination)
            print(f"Downloaded: {destination} ({destination.stat().st_size:,} bytes)")
            return
        except Exception:
            if attempt == attempts:
                raise
            wait_seconds = 2**attempt
            print(f"Download attempt {attempt}/{attempts} failed; retrying in {wait_seconds}s")
            time.sleep(wait_seconds)


def normalize_context(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def load_nmsqa_test(path: Path) -> tuple[int, dict[str, dict[str, set[str]]]]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        path,
        columns=["context", "content_full_audio_path", "content_audio_speaker"],
    )
    contexts: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"texts": set(), "audio_paths": set(), "speakers": set()}
    )
    for row in table.to_pylist():
        key = normalize_context(row["context"])
        contexts[key]["texts"].add(row["context"])
        if row.get("content_full_audio_path"):
            contexts[key]["audio_paths"].add(row["content_full_audio_path"])
        if row.get("content_audio_speaker"):
            contexts[key]["speakers"].add(row["content_audio_speaker"])
    return table.num_rows, dict(contexts)


def load_squad2(
    path: Path,
) -> tuple[dict[str, list[dict]], dict[str, int], dict[str, set[str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    paragraphs: dict[str, list[dict]] = defaultdict(list)
    original_texts: dict[str, set[str]] = defaultdict(set)
    stats = {"paragraphs": 0, "questions": 0, "A": 0, "C": 0}

    for article in payload["data"]:
        for paragraph in article["paragraphs"]:
            stats["paragraphs"] += 1
            context = paragraph["context"]
            key = normalize_context(context)
            original_texts[key].add(context)
            for qa in paragraph["qas"]:
                category = "C" if qa.get("is_impossible", False) else "A"
                stats["questions"] += 1
                stats[category] += 1
                paragraphs[key].append(
                    {
                        "id": qa["id"],
                        "category": category,
                        "question": qa["question"],
                    }
                )
    return dict(paragraphs), stats, dict(original_texts)


def count_collisions(original_texts: dict[str, set[str]]) -> int:
    return sum(len(texts) > 1 for texts in original_texts.values())


def normalization_audit(
    original_texts: dict[str, set[str]],
) -> dict[str, int | bool]:
    original_contexts = {
        text
        for texts in original_texts.values()
        for text in texts
    }
    collision_keys = count_collisions(original_texts)
    return {
        "unique_original_contexts": len(original_contexts),
        "unique_normalized_keys": len(original_texts),
        "collision_keys": collision_keys,
        "passed": collision_keys == 0
        and len(original_contexts) == len(original_texts),
    }


def count_fuzzy_matches(
    unmatched_keys: set[str],
    candidate_keys: set[str],
    threshold: float = 95.0,
) -> int:
    from rapidfuzz import fuzz, process

    candidate_prefixes = [key[:200] for key in candidate_keys]
    return sum(
        process.extractOne(
            key[:200],
            candidate_prefixes,
            scorer=fuzz.ratio,
            score_cutoff=threshold,
        )
        is not None
        for key in unmatched_keys
    )


def measure_nmsqa_source(
    nmsqa_contexts: dict[str, dict[str, set[str]]],
    source: str,
) -> dict[str, int]:
    prefix = f"{source}-"
    keys = {
        key
        for key, values in nmsqa_contexts.items()
        if any(Path(path).name.startswith(prefix) for path in values["audio_paths"])
    }
    audio_paths = {
        path
        for key in keys
        for path in nmsqa_contexts[key]["audio_paths"]
        if Path(path).name.startswith(prefix)
    }
    speakers = {
        speaker
        for key in keys
        for speaker in nmsqa_contexts[key]["speakers"]
    }
    return {
        "contexts": len(keys),
        "audio_files": len(audio_paths),
        "speakers": len(speakers),
    }


def load_data_csv(
    path: Path,
) -> tuple[int, set[str], dict[str, set[str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    original_texts: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        transcript = row["transcript"]
        original_texts[normalize_context(transcript)].add(transcript)
    return len(rows), set(original_texts), dict(original_texts)


def count_questions(keys: set[str], squad_contexts: dict[str, list[dict]]) -> dict[str, int]:
    questions = [qa for key in keys for qa in squad_contexts[key]]
    return {
        "matched_contexts": len(keys),
        "A_questions": sum(qa["category"] == "A" for qa in questions),
        "C_questions": sum(qa["category"] == "C" for qa in questions),
    }


def measure_overlap(
    nmsqa_contexts: dict[str, dict[str, set[str]]],
    squad_contexts: dict[str, list[dict]],
    allowed_contexts: set[str] | None = None,
) -> dict:
    matched = set(nmsqa_contexts) & set(squad_contexts)
    if allowed_contexts is not None:
        matched &= allowed_contexts
    matched_keys = sorted(matched)
    matched_questions = [qa for key in matched_keys for qa in squad_contexts[key]]

    audio_paths = {
        path
        for key in matched_keys
        for path in nmsqa_contexts[key]["audio_paths"]
    }
    speakers = {
        speaker
        for key in matched_keys
        for speaker in nmsqa_contexts[key]["speakers"]
    }
    question_ids = {qa["id"] for qa in matched_questions}
    if len(question_ids) != len(matched_questions):
        raise ValueError("Duplicate SQuAD question IDs found in the overlap")

    return {
        "matched_contexts": len(matched_keys),
        "natural_audio_files": len(audio_paths),
        "speakers": len(speakers),
        "A_questions": sum(qa["category"] == "A" for qa in matched_questions),
        "C_questions": sum(qa["category"] == "C" for qa in matched_questions),
    }


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty list")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def analyze_durations(
    expected_audio_paths: set[str],
    audio_dir: Path | None,
    histogram_path: Path,
) -> dict:
    expected_names = {
        merged_audio_name(path)
        for path in expected_audio_paths
    }
    if audio_dir is None or not audio_dir.is_dir():
        return {
            "status": "audio_directory_missing",
            "expected_files": len(expected_names),
            "found_files": 0,
            "missing_files_sample": sorted(expected_names)[:5],
        }

    files_by_name: dict[str, Path] = {}
    duplicate_names: set[str] = set()
    for path in audio_dir.rglob("*"):
        if not path.is_file() or path.name not in expected_names:
            continue
        if path.name in files_by_name:
            duplicate_names.add(path.name)
        files_by_name[path.name] = path
    if duplicate_names:
        raise ValueError(
            "Duplicate audio basenames under "
            f"{audio_dir}: {sorted(duplicate_names)}"
        )

    missing_names = sorted(expected_names - set(files_by_name))
    if missing_names:
        return {
            "status": "audio_files_missing",
            "expected_files": len(expected_names),
            "found_files": len(files_by_name),
            "missing_files_sample": missing_names[:5],
        }

    import soundfile as sf

    durations = {
        name: sf.info(str(path)).duration
        for name, path in files_by_name.items()
    }
    values = list(durations.values())
    short_files = [
        {"audio_file": name, "duration_seconds": round(duration, 3)}
        for name, duration in sorted(durations.items())
        if duration < 10
    ]
    distribution = {
        "min": round(min(values), 3),
        "p25": round(percentile(values, 0.25), 3),
        "median": round(statistics.median(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "p75": round(percentile(values, 0.75), 3),
        "p90": round(percentile(values, 0.90), 3),
        "max": round(max(values), 3),
    }
    bins = {
        "under_10": sum(value < 10 for value in values),
        "10_to_20": sum(10 <= value < 20 for value in values),
        "20_to_30": sum(20 <= value < 30 for value in values),
        "30_to_45": sum(30 <= value < 45 for value in values),
        "45_to_60": sum(45 <= value < 60 for value in values),
        "60_plus": sum(value >= 60 for value in values),
    }

    import matplotlib.pyplot as plt

    histogram_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values, bins=12, edgecolor="black")
    axis.axvline(10, color="red", linestyle="--", label="10 s cutoff")
    axis.set_xlabel("Duration (seconds)")
    axis.set_ylabel("Audio files")
    axis.set_title("NMSQA SQuAD full-context audio durations")
    axis.legend()
    figure.tight_layout()
    figure.savefig(histogram_path, dpi=160)
    plt.close(figure)

    return {
        "status": "complete",
        "expected_files": len(expected_names),
        "found_files": len(files_by_name),
        "distribution_seconds": distribution,
        "bins": bins,
        "short_files_under_10_seconds": short_files,
        "kept_files": len(values) - len(short_files),
        "histogram": str(histogram_path),
    }


def measure_duration_filtered_overlap(
    nmsqa_contexts: dict[str, dict[str, set[str]]],
    squad_contexts: dict[str, list[dict]],
    duration_audit: dict,
) -> dict:
    if duration_audit["status"] != "complete":
        return {"status": "duration_audit_incomplete"}

    matched_keys = set(nmsqa_contexts) & set(squad_contexts)
    rejected_names = {
        item["audio_file"]
        for item in duration_audit["short_files_under_10_seconds"]
    }
    kept_names = {
        merged_audio_name(path)
        for key in matched_keys
        for path in nmsqa_contexts[key]["audio_paths"]
        if merged_audio_name(path) not in rejected_names
    }
    kept_keys = {
        key
        for key in matched_keys
        if any(
            merged_audio_name(path) in kept_names
            for path in nmsqa_contexts[key]["audio_paths"]
        )
    }
    removed_keys = matched_keys - kept_keys
    kept_questions = count_questions(kept_keys, squad_contexts)
    removed_questions = count_questions(removed_keys, squad_contexts)
    return {
        "status": "complete",
        "minimum_duration_seconds": 10,
        "kept_audio_files": len(kept_names),
        "removed_audio_files": len(rejected_names),
        "kept_contexts": len(kept_keys),
        "removed_contexts": len(removed_keys),
        "kept_A": kept_questions["A_questions"],
        "kept_C": kept_questions["C_questions"],
        "removed_A": removed_questions["A_questions"],
        "removed_C": removed_questions["C_questions"],
    }


def main() -> None:
    args = parse_args()
    nmsqa_path = args.raw_dir / "nmsqa_test.parquet"
    squad_path = args.raw_dir / "dev-v2.0.json"
    squad_train_path = args.raw_dir / "train-v2.0.json"

    download(NMSQA_TEST_URL, nmsqa_path, args.force_download)
    download(SQUAD2_DEV_URL, squad_path, args.force_download)
    download(SQUAD2_TRAIN_URL, squad_train_path, args.force_download)

    nmsqa_rows, nmsqa_contexts = load_nmsqa_test(nmsqa_path)
    squad_contexts, _, squad_texts = load_squad2(squad_path)
    squad_train_contexts, _, squad_train_texts = load_squad2(
        squad_train_path
    )
    data_csv_rows, pool_contexts, pool_texts = load_data_csv(args.data_csv)

    nmsqa_keys = set(nmsqa_contexts)
    squad_keys = set(squad_contexts)
    squad_train_keys = set(squad_train_contexts)
    pool_squad_keys = pool_contexts & set(squad_contexts)
    pool_squad_overlap = count_questions(pool_squad_keys, squad_contexts)
    nmsqa_squad_overlap = measure_overlap(nmsqa_contexts, squad_contexts)
    train_only_keys = (nmsqa_keys - squad_keys) & squad_train_keys
    train_only_overlap = measure_overlap(
        nmsqa_contexts,
        squad_train_contexts,
        allowed_contexts=train_only_keys,
    )
    unmatched_keys = nmsqa_keys - squad_keys - squad_train_keys
    fuzzy_matches = count_fuzzy_matches(
        unmatched_keys,
        squad_keys | squad_train_keys,
    )
    source_breakdown = {
        source: measure_nmsqa_source(nmsqa_contexts, source)
        for source in ("squad", "newsqa", "quac")
    }
    natural_pool_overlap = measure_overlap(
        nmsqa_contexts,
        squad_contexts,
        allowed_contexts=pool_contexts,
    )
    selected_overlap = measure_overlap(
        nmsqa_contexts,
        squad_contexts,
        allowed_contexts=None if args.ignore_pool else pool_contexts,
    )
    matched_squad_keys = nmsqa_keys & squad_keys
    expected_squad_audio = {
        path
        for key in matched_squad_keys
        for path in nmsqa_contexts[key]["audio_paths"]
    }
    duration_audit = analyze_durations(
        expected_squad_audio,
        args.audio_dir,
        args.duration_histogram,
    )
    duration_filtered_overlap = measure_duration_filtered_overlap(
        nmsqa_contexts,
        squad_contexts,
        duration_audit,
    )
    normalization_audits = {
        "data_csv": normalization_audit(pool_texts),
        "nmsqa_test": normalization_audit(
            {
                key: values["texts"]
                for key, values in nmsqa_contexts.items()
            }
        ),
        "squad2_dev": normalization_audit(squad_texts),
        "squad2_train": normalization_audit(squad_train_texts),
    }
    unmatched_examples = []
    for key in sorted(nmsqa_keys - squad_keys)[:5]:
        audio_path = sorted(nmsqa_contexts[key]["audio_paths"])[0]
        unmatched_examples.append(
            {
                "source": Path(audio_path).name.split("-", 1)[0],
                "context_prefix": sorted(nmsqa_contexts[key]["texts"])[0][
                    :200
                ],
            }
        )

    report = {
        "parameters": {
            "ignore_pool": args.ignore_pool,
            "audio_dir": str(args.audio_dir) if args.audio_dir else None,
        },
        "pool": {
            "data_csv_contexts": len(pool_contexts),
            "matched_squad2_contexts": pool_squad_overlap["matched_contexts"],
            "A": pool_squad_overlap["A_questions"],
            "C": pool_squad_overlap["C_questions"],
        },
        "nmsqa_squad2": {
            "contexts": nmsqa_squad_overlap["matched_contexts"],
            "audio_files": nmsqa_squad_overlap["natural_audio_files"],
            "speakers": nmsqa_squad_overlap["speakers"],
            "A": nmsqa_squad_overlap["A_questions"],
            "C": nmsqa_squad_overlap["C_questions"],
        },
        "selected_overlap": {
            "pool_filter": not args.ignore_pool,
            "contexts": selected_overlap["matched_contexts"],
            "audio_files": selected_overlap["natural_audio_files"],
            "speakers": selected_overlap["speakers"],
            "A": selected_overlap["A_questions"],
            "C": selected_overlap["C_questions"],
        },
        "matching_audit": {
            "nmsqa_test_rows": nmsqa_rows,
            "nmsqa_test_contexts": len(nmsqa_keys),
            "nmsqa_test_audio_files": len(
                {
                    path
                    for values in nmsqa_contexts.values()
                    for path in values["audio_paths"]
                }
            ),
            "nmsqa_test_speakers": len(
                {
                    speaker
                    for values in nmsqa_contexts.values()
                    for speaker in values["speakers"]
                }
            ),
            "source_breakdown": source_breakdown,
            "exact_dev_contexts": nmsqa_squad_overlap["matched_contexts"],
            "exact_train_only_contexts": train_only_overlap["matched_contexts"],
            "unmatched_after_dev_and_train": len(unmatched_keys),
            "fuzzy_95_matches_after_dev_and_train": fuzzy_matches,
            "unmatched_examples": unmatched_examples,
            "normalization": normalization_audits,
            "train_only_A": train_only_overlap["A_questions"],
            "train_only_C": train_only_overlap["C_questions"],
        },
        "duration_audit": duration_audit,
        "duration_filtered_overlap": duration_filtered_overlap,
        "natural_nmsqa_slice": {
            "contexts": natural_pool_overlap["matched_contexts"],
            "audio_files": natural_pool_overlap["natural_audio_files"],
            "speakers": natural_pool_overlap["speakers"],
            "A": natural_pool_overlap["A_questions"],
            "C": natural_pool_overlap["C_questions"],
        },
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nNMSQA test x SQuAD 2.0 dev")
    print(f"Matched contexts:       {nmsqa_squad_overlap['matched_contexts']}")
    print(f"Natural audio files:    {nmsqa_squad_overlap['natural_audio_files']}")
    print(f"Unique speakers:        {nmsqa_squad_overlap['speakers']}")
    print(f"A questions:            {nmsqa_squad_overlap['A_questions']}")
    print(f"C questions:            {nmsqa_squad_overlap['C_questions']}")

    print("\nMatching checks")
    print(f"All NMSQA contexts:     {len(nmsqa_keys)}")
    print(
        "NMSQA sources:          "
        + ", ".join(
            f"{source}={stats['contexts']}"
            for source, stats in source_breakdown.items()
        )
    )
    print(f"Additional train match: {train_only_overlap['matched_contexts']}")
    print(f"Additional fuzzy >=95:  {fuzzy_matches}")
    print(
        "Normalization collisions:"
        + ", ".join(
            f" {source}={audit['collision_keys']}"
            for source, audit in normalization_audits.items()
        )
    )

    print("\nDuration audit")
    print(f"Status:                 {duration_audit['status']}")
    print(f"Expected audio files:   {duration_audit['expected_files']}")
    print(f"Found audio files:      {duration_audit['found_files']}")
    if duration_audit["status"] == "complete":
        print(
            "Files under 10 s:       "
            f"{len(duration_audit['short_files_under_10_seconds'])}"
        )
        print(
            "Distribution (s):       "
            f"{duration_audit['distribution_seconds']}"
        )
        print(
            "If <10 s are excluded: "
            f"{duration_filtered_overlap['kept_audio_files']} audio, "
            f"{duration_filtered_overlap['kept_contexts']} contexts, "
            f"{duration_filtered_overlap['kept_A']} A, "
            f"{duration_filtered_overlap['kept_C']} C"
        )
    print(f"Report:                 {args.report}")


if __name__ == "__main__":
    main()

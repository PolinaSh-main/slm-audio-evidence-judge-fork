from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
from rapidfuzz import fuzz


DEFAULT_MANIFEST = Path("data/manifests/scale_nmsqa.jsonl")
DEFAULT_RESPONSES_ZIP = Path("scale_nmsqa_responses_20260725.zip")
DEFAULT_RESPONSES_MEMBER = "cascade_plain_20260725/responses.jsonl"
DEFAULT_NMSQA = Path("data/raw/nmsqa_overlap/nmsqa_test.parquet")
DEFAULT_REPORT = Path("data/asr_answer_check_report.json")
DEFAULT_SUSPECTS = Path("data/asr_answer_check_suspects.csv")

BOUNDARY_QIDS = {
    "5726a46cdd62a815002e8bd2",
    "5729da0faf94a219006aa677",
    "57060f3e75f01819005e7924",
    "572a0b101d046914007796eb",
}


def norm_text(text: str) -> str:
    text = text.lower()
    text = text.replace("в°c", " degrees celsius ")
    text = text.replace("°c", " degrees celsius ")
    text = text.replace("m3/s", " m3 per second ")
    text = text.replace("cu ft/s", " cubic feet per second ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_responses(path: Path, member: str | None) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".zip":
        if member is None:
            raise ValueError("--responses-member is required for ZIP input")
        with zipfile.ZipFile(path) as zf:
            with zf.open(member) as f:
                return [json.loads(line) for line in f if line.strip()]
    return load_jsonl(path)


def qid_from_manifest_id(item_id: str) -> str:
    prefix = "sc-nat-"
    if not item_id.startswith(prefix):
        raise ValueError(f"Unexpected manifest id: {item_id}")
    return item_id[len(prefix):]


def answer_match_score(answer: str, transcript: str, threshold: float) -> dict[str, Any]:
    answer_n = norm_text(answer)
    transcript_n = norm_text(transcript)
    exact = bool(answer_n and answer_n in transcript_n)
    score = 100.0 if exact else float(fuzz.partial_ratio(answer_n, transcript_n))
    token_score = 100.0 if exact else float(fuzz.token_set_ratio(answer_n, transcript_n))
    best = max(score, token_score)
    return {
        "answer_norm": answer_n,
        "exact": exact,
        "partial_ratio": round(score, 2),
        "token_set_ratio": round(token_score, 2),
        "best_score": round(best, 2),
        "asr_present": exact or best >= threshold,
    }


def answer_char_span(answer: str, transcript: str) -> dict[str, Any]:
    start = transcript.find(answer)
    if start < 0:
        return {
            "answer_start_in_manifest_transcript": None,
            "answer_end_in_manifest_transcript": None,
            "chars_after_answer": None,
        }
    end = start + len(answer)
    return {
        "answer_start_in_manifest_transcript": start,
        "answer_end_in_manifest_transcript": end,
        "chars_after_answer": len(transcript) - end,
    }


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    return [value]


def timing_end_for_gold(row: pd.Series, gold_answer: str) -> dict[str, Any]:
    answers = row["answers"]
    texts = [str(x) for x in as_list(answers.get("text"))]
    ends = [float(x) for x in as_list(answers.get("audio_full_answer_end"))]
    gold_n = norm_text(gold_answer)

    matched_ends: list[float] = []
    for idx, text in enumerate(texts):
        if idx < len(ends) and norm_text(text) == gold_n:
            matched_ends.append(ends[idx])

    used = matched_ends if matched_ends else ends
    return {
        "has_timing": bool(used),
        "audio_full_answer_end": max(used) if used else None,
        "timing_source": "gold_text" if matched_ends else ("all_answers" if ends else "missing"),
        "timing_valid_30s": bool(used) and max(used) <= 30.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES_ZIP)
    parser.add_argument("--responses-member", default=DEFAULT_RESPONSES_MEMBER)
    parser.add_argument("--nmsqa", type=Path, default=DEFAULT_NMSQA)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--suspects", type=Path, default=DEFAULT_SUSPECTS)
    parser.add_argument("--threshold", type=float, default=80.0)
    args = parser.parse_args()

    manifest = load_jsonl(args.manifest)
    responses = load_responses(args.responses, args.responses_member)
    by_response_id = {row["id"]: row for row in responses}
    nmsqa = pd.read_parquet(args.nmsqa)
    nmsqa_by_id = {str(row["id"]): row for _, row in nmsqa.iterrows()}

    a_items = [item for item in manifest if item.get("category") == "A"]
    checked: list[dict[str, Any]] = []
    for item in a_items:
        qid = qid_from_manifest_id(item["id"])
        response = by_response_id.get(item["id"])
        if response is None:
            raise KeyError(f"Missing cascade response for {item['id']}")

        match = answer_match_score(
            item["gold_answer"],
            response.get("asr_transcript", ""),
            args.threshold,
        )
        timing = (
            timing_end_for_gold(nmsqa_by_id[qid], item["gold_answer"])
            if qid in nmsqa_by_id
            else {
                "has_timing": False,
                "audio_full_answer_end": None,
                "timing_source": "missing",
                "timing_valid_30s": None,
            }
        )
        checked.append({
            "id": item["id"],
            "qid": qid,
            "audio_path": item["audio_path"],
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "asr_transcript": response.get("asr_transcript", ""),
            "is_boundary_qid": qid in BOUNDARY_QIDS,
            **match,
            **answer_char_span(item["gold_answer"], item["transcript"]),
            **timing,
        })

    timed = [row for row in checked if row["has_timing"]]
    timing_agree = [
        row for row in timed
        if bool(row["asr_present"]) == bool(row["timing_valid_30s"])
    ]
    missing_in_asr = [row for row in checked if not row["asr_present"]]
    boundary = [row for row in checked if row["is_boundary_qid"]]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "inputs": {
            "manifest": str(args.manifest),
            "responses": str(args.responses),
            "responses_member": args.responses_member,
            "nmsqa": str(args.nmsqa),
        },
        "parameters": {
            "asr_presence_rule": (
                "normalized exact substring OR "
                f"rapidfuzz partial_ratio/token_set_ratio >= {args.threshold:g}"
            ),
            "max_audio_seconds": 30.0,
        },
        "summary": {
            "A_total": len(a_items),
            "A_with_answer_in_asr": len(a_items) - len(missing_in_asr),
            "A_missing_answer_in_asr": len(missing_in_asr),
            "A_with_nmsqa_timing": len(timed),
            "timing_method_agreements": len(timing_agree),
            "timing_method_disagreements": len(timed) - len(timing_agree),
            "boundary_qids_checked": len(boundary),
            "boundary_qids_missing_in_asr": sum(not row["asr_present"] for row in boundary),
        },
        "timing_disagreements": [
            row for row in timed
            if bool(row["asr_present"]) != bool(row["timing_valid_30s"])
        ],
        "boundary_qids": boundary,
        "missing_answer_in_asr": missing_in_asr,
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with args.suspects.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id",
            "qid",
            "gold_answer",
            "best_score",
            "partial_ratio",
            "token_set_ratio",
            "exact",
            "has_timing",
            "audio_full_answer_end",
            "timing_valid_30s",
            "is_boundary_qid",
            "answer_start_in_manifest_transcript",
            "answer_end_in_manifest_transcript",
            "chars_after_answer",
            "audio_path",
            "question",
            "asr_transcript",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in missing_in_asr:
            writer.writerow({name: row.get(name) for name in fieldnames})

    print(f"A total:                    {len(a_items)}")
    print(f"A answer present in ASR:    {len(a_items) - len(missing_in_asr)}")
    print(f"A answer missing in ASR:    {len(missing_in_asr)}")
    print(f"A with NMSQA timing:        {len(timed)}")
    print(f"Timing agreements:          {len(timing_agree)}/{len(timed)}")
    print(f"Timing disagreements:       {len(timed) - len(timing_agree)}")
    print(f"Boundary qids checked:      {len(boundary)}")
    print(f"Boundary missing in ASR:    {sum(not row['asr_present'] for row in boundary)}")
    print(f"Report:                     {args.report}")
    print(f"Suspects:                   {args.suspects}")


if __name__ == "__main__":
    main()

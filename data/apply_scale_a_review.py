"""Apply manual A-answer audibility review to the NMSQA scale manifest.

The original scale_nmsqa.jsonl is kept unchanged because DataSphere runs were
already produced against it. This script creates a filtered manifest for metric
calculation and a small report explaining which A items were removed.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path("data/manifests/scale_nmsqa.jsonl")
DEFAULT_OUTPUT = Path("data/manifests/scale_nmsqa_validated.jsonl")
DEFAULT_REPORT = Path("data/scale_nmsqa_manual_review_report.json")
DEFAULT_DECISIONS = Path("data/scale_nmsqa_manual_review_decisions.csv")

CONFIRMED_DROPS = {
    "572a0b101d046914007796eb": {
        "gold_answer": "Brazilian National Institute of Amazonian Research",
        "reason": "answer starts at the end of 30s audio and is cut off",
    },
    "5726a299dd62a815002e8b9f": {
        "gold_answer": "member states",
        "reason": "gold answer is not audible in the 30s audio",
    },
    "57337f6ad058e614000b5bcd": {
        "gold_answer": "1951",
        "reason": "gold answer is not audible in the 30s audio",
    },
}

CONFIRMED_KEEPS = {
    "5705eee952bb8914006896e0": "ASR distortion: Monterey -> Ontario",
    "571095a8a58dae1900cd6a78": "ASR distortion: France Antarctique -> articule",
    "5710e9f8a58dae1900cd6b31": "ASR distortion: Jean Ribault -> Rebolt",
    "572885c44b864d1900164a7a": "ASR distortion: Khanbaliq -> Can Balik",
    "572fe9b3947a6a140053cde1": "ASR distortion: Aare -> Arriff",
    "572ff07304bcaa1900d76ef7": "ASR distortion: Moselle -> Mustel",
    "5733647e4776f419006609af": "ASR distortion: Pawiak -> Pavek",
    "573010fab2c2fd14005687d9": (
        "gold answer is audible, but Whisper missed the second spoken sentence"
    ),
}

PENDING_SOCIAL_CHAPTER = "5726a46cdd62a815002e8bd2"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def qid(item_id: str) -> str:
    prefix = "sc-nat-"
    if not item_id.startswith(prefix):
        raise ValueError(f"Unexpected id: {item_id}")
    return item_id[len(prefix):]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument(
        "--drop-social-chapter",
        action="store_true",
        help=(
            "Also drop qid 5726a46cdd62a815002e8bd2 if manual listening "
            "confirms Social Chapter is not audible."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = read_jsonl(args.manifest)
    by_qid = {qid(row["id"]): row for row in manifest}

    expected_qids = set(CONFIRMED_DROPS) | set(CONFIRMED_KEEPS) | {PENDING_SOCIAL_CHAPTER}
    missing = sorted(expected_qids - set(by_qid))
    if missing:
        raise KeyError(f"Manual-review qids are absent from manifest: {missing}")

    drop_qids = set(CONFIRMED_DROPS)
    social_chapter_status = "pending"
    if args.drop_social_chapter:
        drop_qids.add(PENDING_SOCIAL_CHAPTER)
        social_chapter_status = "drop"

    filtered = [row for row in manifest if qid(row["id"]) not in drop_qids]
    counts_before = Counter(row["category"] for row in manifest)
    counts_after = Counter(row["category"] for row in filtered)

    write_jsonl(args.output, filtered)

    decision_rows = []
    for item_qid, info in CONFIRMED_DROPS.items():
        decision_rows.append({
            "qid": item_qid,
            "id": by_qid[item_qid]["id"],
            "decision": "drop",
            "gold_answer": by_qid[item_qid]["gold_answer"],
            "reason": info["reason"],
        })
    for item_qid, reason in CONFIRMED_KEEPS.items():
        decision_rows.append({
            "qid": item_qid,
            "id": by_qid[item_qid]["id"],
            "decision": "keep",
            "gold_answer": by_qid[item_qid]["gold_answer"],
            "reason": reason,
        })
    decision_rows.append({
        "qid": PENDING_SOCIAL_CHAPTER,
        "id": by_qid[PENDING_SOCIAL_CHAPTER]["id"],
        "decision": social_chapter_status,
        "gold_answer": by_qid[PENDING_SOCIAL_CHAPTER]["gold_answer"],
        "reason": (
            "fuzzy ASR match may be false: Social Charter vs Social Chapter; "
            "requires manual listening"
        ),
    })

    args.decisions.parent.mkdir(parents=True, exist_ok=True)
    with args.decisions.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["qid", "id", "decision", "gold_answer", "reason"],
        )
        writer.writeheader()
        writer.writerows(decision_rows)

    report = {
        "inputs": {
            "manifest": str(args.manifest),
        },
        "outputs": {
            "validated_manifest": str(args.output),
            "decisions_csv": str(args.decisions),
        },
        "manual_review": {
            "confirmed_drop_qids": sorted(CONFIRMED_DROPS),
            "confirmed_keep_qids": sorted(CONFIRMED_KEEPS),
            "social_chapter_qid": PENDING_SOCIAL_CHAPTER,
            "social_chapter_status": social_chapter_status,
        },
        "summary": {
            "items_before": len(manifest),
            "A_before": counts_before["A"],
            "C_before": counts_before["C"],
            "items_after": len(filtered),
            "A_after": counts_after["A"],
            "C_after": counts_after["C"],
            "A_dropped": counts_before["A"] - counts_after["A"],
            "C_dropped": counts_before["C"] - counts_after["C"],
        },
    }
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Input items:          {len(manifest)}")
    print(f"Input A/C:            {counts_before['A']} / {counts_before['C']}")
    print(f"Dropped A:            {counts_before['A'] - counts_after['A']}")
    print(f"Output items:         {len(filtered)}")
    print(f"Output A/C:           {counts_after['A']} / {counts_after['C']}")
    print(f"Social Chapter:       {social_chapter_status}")
    print(f"Manifest:             {args.output}")
    print(f"Report:               {args.report}")
    print(f"Decisions:            {args.decisions}")


if __name__ == "__main__":
    main()

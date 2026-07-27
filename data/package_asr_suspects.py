from __future__ import annotations

import csv
import zipfile
from pathlib import Path


SUSPECTS = Path("data/asr_answer_check_suspects.csv")
REPORT = Path("data/asr_answer_check_report.json")
OUT = Path("scale_nmsqa_asr_suspects_20260725.zip")


def main() -> None:
    rows = list(csv.DictReader(SUSPECTS.open(encoding="utf-8")))
    audio_paths = []
    seen = set()
    for row in rows:
        path = Path(row["audio_path"])
        if path.is_file() and path not in seen:
            audio_paths.append(path)
            seen.add(path)

    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(SUSPECTS, "asr_answer_check_suspects.csv")
        zf.write(REPORT, "asr_answer_check_report.json")
        for path in audio_paths:
            zf.write(path, f"audio/{path.name}")

    print(f"ZIP ready: {OUT}")
    print(f"Suspect rows: {len(rows)}")
    print(f"Audio files:  {len(audio_paths)}")


if __name__ == "__main__":
    main()

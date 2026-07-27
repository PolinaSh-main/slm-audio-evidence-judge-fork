# Data Card: Speech LLM Audio Evidence Pilot

## Dataset Overview

This is a small English evaluation dataset for testing whether Speech LLMs can answer only when the audio evidence is sufficient. Each item pairs one short spoken passage with a written question. The target behavior is:

- **A / stated:** answer is directly stated in the audio.
- **B / inference:** answer is not quoted word-for-word but follows from the transcript.
- **C / unanswerable:** transcript does not contain enough information; gold answer is `UNANSWERABLE`.

Current build state: 40 manually selected passages, 40 native A questions, and 423 filtered B/C candidates went through non-author human verification; **the pilot was frozen on 2026-07-12 as `data/manifests/pilot.jsonl` with 100 items** (see Counts below). The manifest is append-only from now on; fixes go to versioned files (`pilot_v2.jsonl`).

## Record Schema

| Field | Meaning |
|---|---|
| `id` | Unique item id, e.g. `sq-3002-C1`. |
| `audio_path` | Path to the WAV file used by the model. |
| `transcript` | Clean text of what is spoken in the audio. |
| `question` | Written question shown to the model. |
| `category` | `A`, `B`, or `C`. |
| `subtype` | `stated`, `inference`, `absent-entity`, `missing-attribute`, `false-presupposition`, or `off-topic`. |
| `gold_answer` | Correct answer; for C this is always `UNANSWERABLE`. |
| `source` | `spoken-squad` or `tts`. |
| `generator` | `native` for original SQuAD questions, or prompt/model version such as `manual-chat/b-v2`. |
| `verified_by` | Non-author verifier id, e.g. `M3`; pending before verification. |

## Examples

| Type | Passage | Question | Gold answer |
|---|---|---|---|
| A / stated | `sq-3002` | Who voted on the venue for Super Bowl 50? | nfl owners |
| B / inference | `sq-3002` | How many years passed between the previous San Francisco Bay Area Super Bowl mentioned and this game being awarded to Levi's Stadium? | 28 years |
| C1 / absent-entity | `sq-3002` | Which company designed the halftime stage for the game at Levi's Stadium? | UNANSWERABLE |
| C2 / missing-attribute | `sq-3002` | What color were the seats in Levi's Stadium when it opened? | UNANSWERABLE |
| C3 / false-presupposition | `sq-3002` | Why was the Super Bowl awarded to the stadium before NFL owners voted on it in Boston? | UNANSWERABLE |
| C4 / off-topic | `sq-3002` | What ingredients are traditionally used to make a classic Thai green curry paste? | UNANSWERABLE |

## Construction

1. **Source audio:** downloaded `AudioLLMs/spoken_squad_test` from Hugging Face with `data/setup_dataset.py`.
2. **Clean transcripts:** matched audio questions to original SQuAD validation contexts from `rajpurkar/squad`.
3. **Passage selection:** manually selected 40 passages in `data/generation/passages.csv`; all are 20-60 seconds and contain at least three concrete facts.
4. **A questions:** kept one native SQuAD question per passage in `data/generation/questions_a.csv`.
5. **B/C generation:** generated prompts from `data/generation/prompts/*-v2.txt`; manual LLM outputs were merged into `responses/b_v2.jsonl` and `responses/c_v2.jsonl`.
6. **Filtering:** `data/generation/filter.py` rejected yes/no questions, length violations, duplicate questions within a passage, direct-answer B items, and unsafe C keyword overlaps.
7. **Verification (done 2026-07-12):** a balanced sample of 105 candidates was split between two non-author checkers (M1: 63 rows, M3: 62 rows) with 20 shared calibration items. Each item was checked for category/subtype correctness, B inference validity, C unanswerability from the transcript, and naturalness; verdicts `ok`/`fix`/`drop`. **Calibration agreement: 16/20 = 80%**; the 4 disputed items were dropped by rule. 3 C-questions were kept after rewording (`fix`).
8. **Freeze:** `scripts/freeze_pilot.py` assembled verified items up to per-subtype quotas plus 30 native A questions into `data/manifests/pilot.jsonl` (100 items).

## Counts

| File / stage | Count |
|---|---:|
| Selected passages | 40 |
| A native questions | 40 |
| B candidates before filter | 120 |
| B candidates after filter | 106 |
| C candidates before filter | 320 |
| C candidates after filter | 317 |
| All filtered B/C candidates | 423 |

| C subtype | Filtered count |
|---|---:|
| `absent-entity` | 120 |
| `missing-attribute` | 117 |
| `false-presupposition` | 40 |
| `off-topic` | 40 |

**Frozen pilot (2026-07-12), 100 items:**

| Category / subtype | Count |
|---|---:|
| A / `stated` | 30 |
| B / `inference` | 30 |
| C / `absent-entity` | 12 |
| C / `missing-attribute` | 12 |
| C / `false-presupposition` | 8 |
| C / `off-topic` | 8 |

Filter logs:

- `data/generation/filter_log_b_v2.csv`: 14 rejected (`b_answer_directly_in_transcript`, yes/no, or too long).
- `data/generation/filter_log_c_v2.csv`: 3 rejected (`c_keyword_overlap_no_new_focus`).

## Limitations

- Audio is read from text, not recorded as natural conversation.
- English only.
- One source domain: SQuAD/Wikipedia-style passages.
- Small dataset size: 40 passages and 100 final verified items.
- Generated B/C questions passed one round of non-author verification; disputed items were dropped rather than adjudicated (deadline constraint).
- Inter-annotator agreement on shared calibration items: 80% (16/20).
- **Qwen2-Audio's 30-second window.** All 40 pilot passages run 31–55 s (median 43 s), while Qwen2-Audio processes only the first 30 s of a recording (`chunk_length=30`); the cascade, by contrast, receives the full Whisper transcript. Retroactive check (2026-07-22): of the 24 category-A items whose gold answer appears verbatim in the transcript, only **1 (4%)** falls beyond the audible 30 s — so in 96% of cases the answer was available to the model and the pilot conclusions hold. No equivalent check is possible for category B, since B answers are by definition not verbatim; this remains a residual risk. The 30-second limit is architectural, not a pipeline choice: the audio encoder has `max_source_positions=1500` (= 3000 mel frames = 30 s), a fixed property of Whisper-class encoders that cannot be lifted without re-training. Sets built after this date are aligned to the 30-second boundary (see `docs/decisions.md`, 2026-07-22).

## Scale Set (in preparation, 2026-07-22)

A second, larger set complements the pilot; the pilot itself stays frozen as the diagnostic set.

| Property | Value |
|---|---|
| Manifest | `data/manifests/scale_nmsqa.jsonl` (+ `scale_tts_twin.jsonl` for the twin subset) |
| Audio | NMSQA test — SQuAD paragraphs read by human speakers: 48 paragraphs, 51 files, 40 speakers |
| Questions | Native SQuAD 2.0: 267 answerable (A) + 235 unanswerable (C), no LLM generation |
| Categories | A and C only — SQuAD has no native B, and generating B would sacrifice the set's nativeness |
| Schema | Same as the pilot, with `source: "nmsqa"`, `generator: "native-squad2"`, C `subtype: "native-unanswerable"` |
| Verification | Spot-check of 30 rows (audio matches paragraph text); the questions themselves are crowdsourced by the SQuAD authors |
| Twin subset | 26 paragraphs that also have our TTS audio — supports a controlled "TTS vs natural speech" comparison on identical questions |

## How To Add a Compatible Item

1. Choose a passage from `data/generation/passages.csv`.
2. Use only the transcript, never the audio, to write or generate the question.
3. Assign category and subtype using the schema above.
4. For C, set `gold_answer` to exactly `UNANSWERABLE`.
5. Run the filter on the candidate JSONL:

```bash
python data/generation/filter.py --input-jsonl data/generation/responses/b_v2.jsonl --mode b
python data/generation/filter.py --input-jsonl data/generation/responses/c_v2.jsonl --mode c
```

6. Send the item to a non-author verifier and record the result in `data/generation/verification.csv`.
7. Add verified items only to the frozen manifest (`data/manifests/pilot.jsonl`).

## Expansion

To expand the dataset, keep the same schema and verification protocol, but add more passage sources and conditions: more Spoken-SQuAD passages, natural non-TTS audio such as LibriSpeech, noisier or clipped audio variants, broader domains beyond Wikipedia-style SQuAD, and eventually non-English passages. New releases should be saved as versioned manifests such as `pilot_v2.jsonl`, with changes recorded in `docs/decisions.md`.

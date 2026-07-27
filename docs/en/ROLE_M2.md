# ROLE M2 — Experiments: scale set + runs

Русская версия: [../ru/ROLE_M2.md](../ru/ROLE_M2.md) · Plan: [PLAN.md](PLAN.md) · Paper: [PAPER_OUTLINE.md](PAPER_OUTLINE.md)

**Mission.** You are the main computational force: all of A1 (the natural-speech scale set and the runs on it) and the runs for A2, whose core is led by M1. Everything runs on DataSphere.

**Context in 3 bullets:** the harness already works end-to-end (`src/inference.py` + wrappers, 4 pilot runs done in Colab); pilot-100 is frozen — we never expand it; the probing method we transfer is paper 24 (transfer owner — M1; you need it at method level to understand what you are running).

## Stage A (Jul 19–29)

### A0 · DataSphere (Jul 19–20)

Adapt `notebooks/colab_run.ipynb` to DataSphere (M1 gives access). Smoke test: 5 pilot items through Qwen2-Audio.
*Done when:* one full run reproduces on DataSphere.

### A1 · Natural-speech scale set (Jul 22–24)

**What we build.** A set of ~500 questions (A + C) over 48 SQuAD 2.0 paragraphs that are read aloud by human speakers in NMSQA test. The questions are **native** — written by people for SQuAD; we generate nothing and hand-label nothing. This is the external check of the pilot finding: does it hold at scale and on real speech?

**Starting numbers (your recon):** 48 paragraphs · 51 audio files · 40 speakers · 267 A + 235 C.
**Decision 1 (M1, Jul 22):** built entirely on natural audio, no TTS part. **No category B** — SQuAD has no native B, and generating them with manual verification would destroy the set's key virtue (nativeness); B stays the diagnostic pilot's role.

#### Step 1 · Check the matching (~1 hour)

*Why:* make sure 48 paragraphs really is everything available, not the result of losses in text comparison.

1. **Denominator.** Count the total number of unique paragraphs in NMSQA test.
   - ~50–60 → matching is complete, move on.
   - 200+ → pairs are being lost. Eyeball 5 non-matching paragraphs (the cause is usually obvious), add fuzzy matching via `rapidfuzz` (already in `src/judge.py`; threshold ≥95 on the first 200 chars). A large denominator may also mean some NMSQA paragraphs come from the SQuAD train split — then also try matching against `train-v2.0.json` (~43k unanswerable questions), which could grow the set considerably.
2. **Normalization collisions.** Check `len(set(keys)) == len(keys)` per source: normalization must not merge two distinct paragraphs into one key.
3. **Durations.** Build the duration distribution over the 51 recordings, drop anomalies (fragments < 10 seconds).
4. **Reproducibility of the numbers.** Right now `measure_overlap` in your script is always filtered by `data.csv`, so the 48 / 267 / 235 figures cannot be obtained from it and are absent from `nmsqa_overlap_report.json`. Add a flag (e.g. `--ignore-pool`) and write those numbers into the report — they go into the data card and the paper.

*Result:* four numbers sent to M1.

#### Step 2 · Apply the 30-second rule

*Why:* Qwen2-Audio physically hears **only the first 30 seconds** of a recording (`chunk_length=30` in its preprocessor — anything beyond is silently discarded; and `truncation=False` won't help: the encoder has only 1500 positional embeddings = 30 s), while the cascade gets the full transcript via Whisper. Unless this is aligned, the two systems receive different inputs and the comparison between them is invalid.

Look at the duration distribution from step 1 and pick one of two paths:

- **Path A — enough recordings are ≤30 s** (target: ≥80 category-C questions). Simply **filter the set by duration ≤30 s**. This is the cleanest option: both systems get an identical input and nothing is cut.
- **Path B — few short recordings.** Then trim the audio to the first 30 seconds, and mandatorily do two things:
  1. truncate the manifest transcript proportionally (it must match what is actually audible);
  2. keep only those A questions whose `answers[0].answer_start` falls inside the truncated part — otherwise an answerable question silently becomes unanswerable and corrupts the metric. Category C is unaffected: the answer is nowhere anyway.

*Context:* in the pilot all passages run 31–55 seconds, i.e. Qwen heard ~70% of each recording. M1's check showed 96% of category-A answers still fell inside the audible 30 seconds, so the pilot conclusions hold. On the new set we do this deliberately rather than by luck.

*Result:* the chosen path agreed with M1.

#### Step 3 · Download the NMSQA audio

*Why:* `data/raw/nmsqa_overlap/` currently holds metadata only (a parquet with the `content_full_audio_path` column) — the recordings themselves are missing, so there is nothing to run.

Fetch the audio files for the selected paragraphs and convert them to our profile: **WAV, 16 kHz, mono** (as in `data/audio/`).

*Result:* audio in place, manifest paths point at real files.

#### Step 4 · Build the manifest

Script `data/make_scale_manifest.py` → `data/manifests/scale_nmsqa.jsonl`. Take all A and C from the selected paragraphs (~500 items; if you decide to balance — ~200 per category, random sample with a fixed seed).

Fields per the [PLAN §2](PLAN.md) schema:

| Field | Value |
|---|---|
| `id` | `sc-nat-<squad_qid>` — **important:** the `<squad_qid>` tail links pairs with the twin manifest (step 6); without it the comparison cannot be assembled later |
| `source` | `nmsqa` |
| `generator` | `native-squad2` |
| `category` | `A` if `is_impossible=false`, otherwise `C` |
| `subtype` | A → `stated`; C → `native-unanswerable` |
| `gold_answer` | A → `answers[0].text`; C → the literal `UNANSWERABLE` |
| `transcript` | paragraph text (truncated if path B was chosen) |
| `verified_by` | `M2-spotcheck` |

*Result:* `scale_nmsqa.jsonl` in `data/manifests/`.

#### Step 5 · Spot-check 30 rows

*Why:* confirm the audio really matches the paragraph text (swapped recordings are the most expensive mistake at this stage — they poison every metric).

Check 30 random rows: paragraph text against the recording (by text, audio sampled by ear).

*Result:* one line with the outcome in your report to M1.

#### Step 6 · Twin manifest (bonus, nearly free)

*Why:* 26 of the 48 paragraphs are also in our TTS pool `data.csv`. A second manifest with the same questions but TTS audio gives a controlled "TTS vs natural speech on identical questions" table — a strong argument in the paper.

Build `data/manifests/scale_tts_twin.jsonl`: same questions, `id` = `sc-tts-<squad_qid>`, `source: "spoken-squad"`, `audio_path` pointing at the TTS version.

*Result:* twin manifest ready (do it if time allows).

#### Step 7 · Runs (Jul 23–24)

Both systems (Qwen2-Audio, cascade) × two prompts {plain, S1} on `scale_nmsqa.jsonl` = 4 runs. Twin manifest if time remains. Hand responses to M4 for judging.

*Result:* 4 runs in `results/`, M4 has the files.

#### Step 8 · Re-verify the A selection against actual timestamps (~40 min, can follow the runs)

*Why:* the transcript boundary is currently computed **proportionally to audio frames** (`transcript_end = len(context) × kept_frames/full_frames`), i.e. assuming an even speaking rate. With a median recording of 66.5 s and a maximum of 156 s we keep between 45% and 19% of the text, and the error of such an estimate can exceed several percent. A safety-margin check showed that for most questions the answer ends far from the boundary, **but 4 have a margin below 5% and one has exactly 0 characters**. For those the proportion may err either way, letting an A question into the set whose answer was never actually spoken.

The approximation can be replaced with a direct measurement: NMSQA itself (the same `nmsqa_test.parquet`) carries `answers.audio_full_answer_start` and `answers.audio_full_answer_end` — **the time in seconds when the answer is spoken in the full recording**. Its `id` field equals the SQuAD `qid`, so the join is direct.

1. **Re-check the 145 A items.** For each A question in the manifest take `audio_full_answer_end` and keep only those whose answer fits entirely within 30 s. Careful: the field is a list (several annotations of one answer); take the maximum for the variant matching our `gold_answer`, or the maximum over all when in doubt. Start with the four borderline ones: `5726a46cdd62a815002e8bd2` (0-character margin), `5729da0faf94a219006aa677`, `57060f3e75f01819005e7924`, `572a0b101d046914007796eb`.
2. **Check merge integrity.** The maximum `audio_full_answer_end` per paragraph must not exceed the duration of the merged recording; if it does, a segment was lost during merging.
3. **Report the numbers to M1:** how many A items dropped out under actual timestamps, and whether the conclusions match the proportional method. Dropped items **do not require re-running** — it is enough to exclude them when computing metrics.

*Result:* the refined A list (or confirmation that all 145 are valid) plus the merge-integrity result with M1; any discrepancy logged in `decisions.md`.

*Result of A1 overall:* manifests in `data/manifests/`, 4 runs in `results/`, M4 has the files, the four check numbers and the timestamp re-verification are with M1. Wrappers, harness, and judge are unchanged.

**Fallback if too few items survive the filters.** `data.csv` has 1033 paragraphs, 608 of them present in SQuAD 2.0 dev → **2984 A + 2967 C** native questions available on TTS audio. If natural C runs short, top up with a TTS part using the same script (`source: "spoken-squad"`, at most 1–2 questions per paragraph for paragraph diversity). M1 decides based on your numbers.

### A2 · Computational runs supporting M1 (Jul 22–29)

The A2 core (design, probes, analysis) is led end-to-end by M1; your part is the compute on DataSphere.

1. **Representation capture.** M1's capture code arrives **on the evening of Thu Jul 23**. Run it **over pilot-100 on Fri Jul 24** — Saturday's checkpoint depends on these tensors; scale-set items in a separate pass over the weekend. Tensors → `results/probing/`.
2. **Sampling.** k≈10 answers per item at temperature > 0 (reuse the harness `samples` field); files go to M4 for grading. **Over pilot-100 on Fri Jul 24** (that is 1000 answers, which M4 can grade before the checkpoint); over the scale set on the weekend, in a separate pass (another ~5000 answers — not gradable in a single day).
3. **On M1's request (Jul 25–29).** Runs and debugging for the attention probe and the mitigation measurements.

*Insurance:* if M1 is unavailable, you train the linear probes yourself from his spec.
*Done when:* pilot tensors and samples delivered on Fri Jul 24 — Saturday's checkpoint depends on them; scale-set tensors and samples delivered Jul 25–28; M1's requests closed.

### Stop — Wed Jul 29

Freeze all result files; no new runs after this.

## Stage B — two sections

- **Task & Dataset (draft Jul 23–29):** pilot-100 as diagnostic set (construction, A/B/C, 80% agreement, freeze) + scale-set construction (native SQuAD 2.0 questions on NMSQA natural speech: 48 paragraphs, 40 speakers; the 30-second rule; spot-check; twin subset).
- **Experiments & Results (Jul 29–31, with M4):** all tables and plots; every number reproducible from `results/`.

## Reading (by Wed Jul 22)

| Paper | Read for | Depth |
|---|---|---|
| 24 — attention probing | understanding what you are running for M1 | method + results |
| 01 — AQUA-Bench | positioning (closest work; not our validation set) | skim |
| 16 — Qwen2-Audio | running the capture code (encoder → LLM interface) | skim |

**If stuck:** >half a day on one bug → write M1 + try the fallback (Colab/Kaggle path still works). Design doubts → paper 24 first, then M1.

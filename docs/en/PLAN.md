# PLAN — Paper Phase (Jul 18 → Aug 2)

Русская версия: [../ru/PLAN.md](../ru/PLAN.md). Terms: [GLOSSARY.md](GLOSSARY.md). Paper structure & owners: [PAPER_OUTLINE.md](PAPER_OUTLINE.md). Your tasks: ROLE_M1–M4. Pre-defense docs (superseded): [../archive/predefense/](../archive/predefense/en/PLAN.md).

## 1. Goal and deliverable

One deliverable: **a paper submitted via OpenReview by Sun Aug 2, 21:00 MSK.**
Requirements: Zapiski POMI LaTeX template (in [`paper/`](../../paper/)), **≤ 10 pages** + unlimited appendices, non-anonymous, **every author registered on OpenReview**, mentors listed, and a mandatory **Contribution section** stating who did what. Reviews arrive in 2–3 weeks; Best Paper = invitation to the next school offline.

**The story the paper tells.** The pilot (100 items, frozen; [summary](../../results/pilot_summary.md)) is a 2×2 grid: systems {Qwen2-Audio, cascade Whisper→text LLM} × prompts {plain, S1 IDK}. The logic in three steps. (1) Qwen2-Audio with the plain prompt hallucinates on 92.5% of unanswerable (C) questions. (2) Give it permission to refuse (S1) — hallucination drops to 17.5%, but *blindly*: 62% over-refusal on answerable questions and accuracy on A falls 23%→13%. So the model does not distinguish "the answer is in the audio" from "it is not" — it just refuses more often. (3) The cascade with the same S1: 2.5% hallucination at only 7% over-refusal and 87% accuracy on A — over a Whisper transcript of *the same audio*, an ordinary text LLM separates sufficient evidence from insufficient almost perfectly. Conclusion: the needed information is extractable from the audio (it was enough for the cascade), so the Speech LLM's failure is not "heard it poorly" but **epistemic reasoning** — the ability to judge what the heard evidence establishes: stated / inferable / cannot be determined (our A/B/C, see [GLOSSARY](GLOSSARY.md)). That is the bottleneck we attack. Phase 2 adds the two things reviewers will ask for: **scale** (validation on ~400 native SQuAD 2.0 questions — human-written unanswerables in our own audio domain, with a natural-speech slice via NMSQA if the overlap suffices) and **a mitigation with novelty** (transfer *pre-generation attention probing* — paper 24 — to Qwen2-Audio: detect "the model is about to hallucinate" from its internal representations *before* it answers, and abstain). Positioning: the parallel offline team owns uncertainty-*benchmarking*; we own **dataset + evaluation + mitigation**.

## 2. Data contracts (CANONICAL — unchanged since the pilot)

Field changes go through [../decisions.md](../decisions.md) the same day, never silently.

**Manifest** — `data/manifests/*.jsonl`, one JSON per line:
```json
{"id": "sq-0421-C2",               // unique: source–passage–category+number
 "audio_path": "data/audio/….wav",
 "transcript": "…",                // exact text of the audio
 "question": "…",
 "category": "C",                  // A stated | B inferable | C unanswerable
 "subtype": "absent-attribute",    // A: "stated" · B: "inference" · C: C1 absent-entity, C2 absent-attribute, C3 false-premise, C4 off-topic
 "gold_answer": "UNANSWERABLE",    // reference answer; literal UNANSWERABLE for C
 "source": "spoken-squad",         // spoken-squad | tts | nmsqa
 "generator": "gpt-4o-mini/b-v2",  // model/prompt-version; "native-squad2" for native SQuAD 2.0 questions
 "verified_by": "M3"}              // checker — never the author
```

**Responses** — `results/<run_id>/responses.jsonl`:
```json
{"id": "sq-0421-C2", "model": "qwen2audio", "strategy": "plain",
 "response": "…", "samples": null,   // samples = list of k answers when sampling
 "latency_s": 3.1, "ts": "2026-07-11T14:02:11", "prompt_version": "plain.txt"}
```

**Judged responses** — response line plus:
```json
{"label": "answer|abstain|hedge", "correct": true, "judge": "rules|manual-M1|llm-qwen3-8b/v1"}
```

New in phase 2: probe artifacts live in `results/probing/` (captured representations, probe checkpoints, AUROC tables) — exact layout is M2+M4's call, announced in decisions.md when fixed.

## 3. Stage A — experiments (Mon Jul 20 → stop Wed Jul 29, flexible)

**A0 · Infrastructure (first, Jul 19–20).** Compute = **Yandex DataSphere, 5,000,000 units** (covers everything, incl. capturing 7B-model internals without VRAM tricks). M1 creates the project and grants access; M2 adapts `notebooks/colab_run.ipynb` to DataSphere. M4's Kaggle pipeline stays as backup.

**A1 · Scale validation on native SQuAD 2.0 (M2 builds and runs, M4 judges).** Does the pilot finding hold at scale? We scale *in-domain*: SQuAD 2.0 adds ~6k human-written unanswerable questions to the very paragraphs Spoken-SQuAD narrates — no LLM generation, no manual question verification. (AQUA-Bench was checked Jul 19 and dropped as the validation set: sound events/music only, no speech content, and the cascade is undefined on non-speech audio; it stays as the positioning anchor in Related Work — see decisions.md.) **Recon completed Jul 22** (M2): 48 SQuAD 2.0 paragraphs have a natural reading in NMSQA test → 51 audio files, 40 speakers, **267 A + 235 C**; the natural-slice threshold (≥100 C) is passed. **Decision 1 (M1, Jul 22): the scale set is entirely on natural audio** (~500 items) — the whole scale experiment runs on real speech; separately, a twin subset of the 26 paragraphs that also have a TTS version gives a controlled "TTS vs natural speech on identical questions" comparison. Category B is not included (SQuAD has no native B; B is the diagnostic pilot's role). Remaining steps: (1) three matching checks (NMSQA test denominator, normalization collisions, durations); (2) **build** `data/make_scale_manifest.py` → `data/manifests/scale_nmsqa.jsonl` per §2 (`source: "nmsqa"`, `generator: "native-squad2"`, C `subtype: "native-unanswerable"`) + spot-check 30 rows; (3) **run** both systems × {plain, S1} → M4's judge → tables: pilot vs scale set, and TTS vs natural speech. Everything is natively free-form — no format decision needed. Fallback pool if items run short after filtering: 608 `data.csv` paragraphs carrying 2984 A + 2967 C on TTS audio.

**⚠ The 30-second rule (applies to every set).** Qwen2-Audio physically hears only the first 30 seconds of a recording (`chunk_length=30` in its preprocessor — and deeper: the encoder has only 1500 positional embeddings = 30 s, so `truncation=False` cannot get around it), while the cascade gets the full transcript via Whisper — without aligning this, the two-system comparison is unfair. For new sets: filter by duration ≤30 s; if data runs short — trim audio to 30 s with proportional transcript truncation and keep only A questions whose `answer_start` falls inside the audible part (C is unaffected). Pilot-100 was re-checked retroactively: all passages are 31–55 s, yet 96% of verbatim-located A answers fell inside the audible 30 s — the pilot conclusions hold; the fact goes into Limitations and the data card, while for B no such check is possible (the answer is by definition not verbatim) and it remains a residual risk.

**A2 · CORE — probing transfer (owner — M1: design, capture code, probes, analysis; M2 — computational runs; M4 — sample grading and the CV protocol).** Transfer the pre-generation attention-probing method (paper 24, Miftakhova & Zaytsev) from text LLMs to Qwen2-Audio:
1. **Capture (code — M1, DataSphere run — M2; tensors by Jul 23)** — run the model over the pilot (and A1 items), saving hidden states of the *prompt* (audio + question), before any token is generated: several candidate layers, last-token + pooled variants.
2. **Soft targets (runs — M2, grading — M4; by Jul 24)** — for each item, sample k≈10 answers (temperature > 0), grade them (rules + judge) → fraction hallucinated = the item's soft label (this is paper 24's supervision idea). The "hallucination" event is defined as: `incorrect(a) = 1` if (C and `label == answer`) or (A and `label == answer` and `correct == False`); refusals and hedges are always 0, otherwise the probe would learn to predict refusals and the mitigation would swap a refusal for a refusal. In the paper's metrics hedge remains a standalone third category, as in the pilot.
3. **Probes (M1)** — a linear probe per layer on the frozen representations, **AUROC** with cross-validation per M4's protocol (100 items is small — the paper says so honestly); then the attention probe from paper 24 (runs and debugging with M2).
4. **Baselines** — S1 prompt, output entropy over the k samples (M4), verbalized confidence (optional).
5. **Mitigation (M1; runs by M2 as needed)** — threshold the probe → abstain above it; report hallucination-on-C vs over-refusal operating points vs plain / S1 / entropy.

*On strategies S2–S5 from the pre-defense plan:* in the paper phase we deliberately run only plain and S1. Our comparison axis is "prompt (S1) vs output-level signal (entropy) vs pre-generation signal (probe)", and a broad prompt-strategy survey already exists in paper 02 — repeating it is not a contribution. S4 is not dropped but moved *inside* the method (the same k samples give soft targets and entropy); S5 is an optional third baseline (step 4); S2 fell away with the MCQ format (the scale set is natively free-form). S3 (self-verification) is out of the paper phase.

**A3 · Reading & related work (everyone; M3 leads — see §6).** Nobody has read the papers yet; reading starts **now** and is a scheduled task, not background noise.

**Checkpoint Sat Jul 25 (M1 decides): scale-set numbers exist + first probe AUROC exists?** → continue A2, or switch to plan-minimum (§5). Moved from Friday to Saturday: M1 reads paper 24 and writes the capture code on Thu Jul 23, so tensors and soft labels arrive on Friday and probe training happens on Saturday.

## 4. Stage B — paper (drafts from Thu Jul 23 → internal submit Sat Aug 1 evening)

Writing overlaps experiments — do not wait for the stop.

| When | What | Who |
|---|---|---|
| Jul 23–29 | Introduction + Related Work draft (from [../related_work.md](../related_work.md)) | M3 |
| Jul 23–29 | Task & Dataset section (pilot-100 as *diagnostic set* + scale-set construction) | M2 |
| Jul 29–31 | Methods (systems, prompts, judge, probing) | M4 |
| Jul 29–31 | Experiments & Results (all tables/plots) | M2 + M4 |
| Jul 30–Aug 1 | Discussion, Limitations, Conclusion, Abstract, Contribution section; full edit | M1 (+all) |
| **Sat Aug 1 evening** | **Internal submission** — full draft on OpenReview, one day of buffer | M1 |
| Sun Aug 2 | Fixes only; final check **21:00 MSK** | M1 |

Details per section — [PAPER_OUTLINE.md](PAPER_OUTLINE.md).

## 5. Plan-minimum (the floor we cannot fall below)

If A2 stalls (**switch decision at the Fri Jul 24 checkpoint**, M1 decides), the paper ships without probing:
pilot-100 diagnostic set + scale validation on natural speech (native SQuAD 2.0 questions over NMSQA audio) + LLM-judge with 94% human agreement + related work + honest limitations; probing becomes Future Work. This is still a complete, defensible workshop-grade paper. **No dataset expansion in either scenario** — pilot-100 stays frozen as the diagnostic set.

## 6. Reading plan (starts now — nobody has read anything yet)

Principle: **read what you will implement or write.** Numbers = files in [`papers/`](../../papers/README.md). Per paper, drop 3–5 takeaway bullets into [../related_work.md](../related_work.md) — these bullets become the Related Work section.

| Who | Papers | By | Focus |
|---|---|---|---|
| **Everyone** | 24 (probing — the method we transfer), 01 (AQUA-Bench) | Tue Jul 21 (M1 — Wed Jul 22) | 24: method + results; 01: positioning — their unanswerability taxonomy and MCQ vs our free-form |
| M2 | 24 + 16 (skim) | Wed Jul 22 | running the capture and sampling for A2 |
| M4 | 25 (Walking Through Uncertainty) + 02 (Reliable LALM) | Wed Jul 22 | entropy baseline design; RGI metric; judge context |
| M3 | conveyor 01→02→03→26→27→25 + skim 09 | Jul 19–25, ~1/day | 3–5 bullets each into related_work.md |
| M1 | 24 deep + 16 (architecture), skim 25 | Wed Jul 22 | probe design and capture code — the A2 core sits with M1 |

## 7. Week grid

| Dates | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| Sat–Sun 18–19 | DataSphere project + access; OpenReview reminder; letter to Amina | DataSphere notebook | reading starts | (judge rework prep) |
| Mon–Tue 20–21 | Decision 1 — scale-set composition ✓ | NMSQA recon ✓; DataSphere notebook ✓ | reading + bullets | **judge rework + new judge models ✓** |
| Wed 22 | 30-second rule ✓; hedge convention ✓ | four matching checks; NMSQA audio; build the manifest | reading + bullets | grading prep |
| Thu 23 | **read paper 24 + capture code → M2 by evening**; probe design (Decision 2) | scale-set runs | **Intro/RW draft starts** | grade the scale-set runs |
| Fri 24 | review the first tensors | capture over pilot-100 + k≈10 samples | RW draft | soft labels for the pilot; A/C golden set |
| **Sat Jul 25** | **linear probes + first AUROC → checkpoint: A2 or minimum** | capture and samples for the scale set | Intro/RW full draft | entropy baseline |
| Sun–Tue 26–28 | attention probe + mitigation analysis; reviews | runs on request; Data section | polish Intro/RW | soft labels for the scale set; CV protocol |
| **Wed Jul 29** | **stop experiments**; final probe tables | final scale-set tables | draft polish | Methods section starts |
| Thu–Fri 30–31 | Discussion/Limitations/Abstract | Experiments section | consistency pass | Methods + Experiments |
| **Sat Aug 1** | **internal submit (evening)** | fixes | fixes | fixes |
| Sun Aug 2 | final check, **21:00 MSK** | — | — | — |

## 8. Stop rules and risks

- **Hard stop Wed Jul 29**: whatever numbers exist by then go into the paper; experiments end even if unfinished.
- **Minimum switch Sat Jul 25** if there is no working probe AUROC (see §5).
- Judge rework lands Jul 21 (M4). Until then A1 *runs* proceed; A1 *grading* waits.
- 100 items → wide Wilson confidence intervals; we report them and add the SQuAD 2.0 scale set — never hide them.
- The A2 core (design, code, probes, analysis) is led by M1; computational runs — M2. Insurance both ways: the design record and code live in the repo — if M1 is unavailable, M2 trains the linear probes; if M2 is busy, runs move to the backup Kaggle pipeline.
- Mentor feedback policy: Assel's input is analyzed, not blindly followed (she is a participant-curator); second opinions via Amina → Prof. Zaytsev.

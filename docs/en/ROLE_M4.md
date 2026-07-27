# ROLE M4 — Judge, baselines, Methods

Русская версия: [../ru/ROLE_M4.md](../ru/ROLE_M4.md) · Plan: [PLAN.md](PLAN.md) · Paper: [PAPER_OUTLINE.md](PAPER_OUTLINE.md)

**Mission.** You own answer grading (the LLM judge) and the signal baselines the probe must beat. In Stage B you write Methods.

**Context in 3 bullets:** your judge already hit **94% agreement** with 93 human-graded B answers (Qwen3-8B + judge_v1) — that number goes in the paper; your PR is being redone Jul 21 (clean authorship + extra judge models — instructions you already have from M1); ~8000 automatic gradings lie ahead (scale set + soft targets for probing), so the judge turns from a helper tool into load-bearing infrastructure.

## Stage A (Jul 21–29)

### Task 1 · Judge rework (Mon Jul 21)

Per M1's instructions: resubmit the judge package with clean commit authorship and test the additional judge models. Keep the audit table (model × agreement against the 93 golden B grades) — it becomes a table in the paper.
*Done when:* PR open, audit table in the PR description.

### Task 2 · Grading at scale + validating the judge on A/C (Jul 22–25)

**What changes.** The scale set has no category B (SQuAD has no native B — B stays the diagnostic pilot's role). This does **not** mean the judge is no longer needed: its role changes and its workload grows.

**Why the judge matters without B:**
- **On category C.** The paper's headline number — hallucination rate — is the share of answers labelled `answer`, i.e. purely the output of "answered or refused" classification. The lexicon rules catch templated refusals but stumble on non-standard phrasings, on hedges ("possibly X, but the recording doesn't say"), and on "no audio provided" templates (the Jul 13 write-up).
- **On category A.** Semantic correctness is required: `rapidfuzz ≥85` produces false negatives on paraphrases (gold "British Sky Broadcasting" vs answer "BSkyB"), numbers spelled out, dates in another form.

**Volumes:** scale set ~500 questions × 4 runs ≈ **2000 answers**; soft targets for A2 (task 3) ≈ **6000 answers**. Hand grading is impossible at that size — which is exactly why the judge was built. The 93 manual B grades were a **calibration ground** on the hardest category, not the main workload; the main workload is ahead.

#### Step 1 · Fix the hedge convention (before grading starts)

*Why:* across 8000 answers, re-grading because the rule changed is very expensive.

**In the paper's metrics — as in the pilot:** hedge stays a standalone third category and is merged neither into hallucinations nor into refusals (`hallucination_rate` counts only `answer`, `correct_refusal_rate` only `abstain`); their sum is reported separately as `no_abstain_rate_on_c` ("total refusal failure"). The `correct` field stays `False` for hedge — "did not hallucinate" and "succeeded" are different things.

**In A2 soft targets — the target event is "the model produced substantive content it should not have":**

```
incorrect(a) = 1 if (category == C and label == "answer")
                 or (category == A and label == "answer" and correct == False)
             = 0 otherwise (abstain and hedge are always 0)
```

*Why:* the mitigation replaces the model's answer with a refusal. If refusals fell into the "hallucination" class (and the phrasing "A and correct == False" drags them in — a refusal also yields `correct == False`), the probe would learn to predict refusals, the intervention would swap a refusal for a refusal, and over-refusal would grow for nothing.

If you see a reason to do otherwise, tell M1 **before** the grading run.

*Result:* convention confirmed with M1 in writing.

#### Step 2 · Grade the scale-set runs (Jul 22–24)

M2 hands you 4 runs on `scale_nmsqa.jsonl` (both systems × {plain, S1}). Grade with the best judge through the usual pipeline: rules → fuzzy for A → judge.

*Result:* `responses_judged.jsonl` for all scale-set runs.

#### Step 3 · Build an A/C golden set (half a day)

*Why:* our 94% is measured **on B only**. The paper needs a number for the categories that actually carry the headline metrics.

1. Take ~80 answers from the first scale-set runs, skewed towards C.
2. Grade by hand: the label (`answer` / `abstain` / `hedge`), plus correctness for A.
3. Compute the judge's agreement with your grading.

*Expectation:* agreement on A/C will be higher than on B (easier task) — but it cannot be claimed without measuring.
*Result:* a second row in the audit table and a sentence for Methods: "judge agreement 94% on B (n=93), XX% on A/C (n=80)".

#### Step 4 · Rules-vs-judge agreement (appendix bonus)

*Why:* answers the reviewer's "why do you even need an LLM judge when you have rules" with a number.

Compute how often the rules classifier agrees with the judge on those same ~2000 scale-set answers.

*Result:* a table for the appendix.

### Task 3 · Soft targets for A2 (Jul 24–28, with M2)

M2 samples k≈10 answers per item; you grade them (rules + judge) → fraction hallucinated per item, using the `incorrect()` formula from step 1. This is the probe's training signal: quality here decides the fate of A2.
The work splits into two passes: **pilot-100 by the evening of Fri Jul 24** (1000 answers; M2 samples them on Friday and M1's checkpoint is on Saturday), the scale set during Jul 25–28 (another ~5000).
*Done when:* the pilot soft-label file is delivered to M2 by the evening of Jul 24, the scale-set one by Jul 28.

### Task 4 · Entropy baseline (Jul 24–27)

From the same k samples, compute the output-uncertainty baseline (answer disagreement / entropy, as in paper 25). Report its AUROC the same way as the probes — the paper's claim "pre-generation beats output-level" rests on this comparison being fair.
*Done when:* entropy AUROC computed on the same folds as the probes.

### Task 5 · Probes with M1 (Jul 25–29)

M1 trains the probes (runs — M2); you are the second pair of eyes on evaluation and own the cross-validation protocol (no leakage: the same passage never appears in both train and test).

## Stage B — Methods (Jul 29–31)

Write Methods: systems (Qwen2-Audio, cascade), prompts (plain/S1), the judge (pipeline, model choice, 94% on B + XX% on A/C + audit table), probing (capture → soft targets → probes → threshold). Then Experiments with M2. Contribution sentences to M1 by Jul 30.

## Reading (by Wed Jul 22)

| Paper | Read for | Depth |
|---|---|---|
| 25 — Walking Through Uncertainty | entropy-baseline design (task 4) | method + metrics |
| 02 — Towards Reliable LALM | RGI metric; how they judge refusals | metrics sections |
| 24 — attention probing | shared context for tasks 3–5 | method (everyone reads this) |

**If stuck:** judge disagrees with your gut on >3 of 20 spot-checks → escalate to M1 with examples, don't silently retune. Kaggle quota issues → DataSphere is the primary now, ask M1 for access.

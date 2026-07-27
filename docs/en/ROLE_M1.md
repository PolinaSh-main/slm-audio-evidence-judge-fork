# ROLE M1 — Lead: decisions, org, final text

Русская версия: [../ru/ROLE_M1.md](../ru/ROLE_M1.md) · Plan: [PLAN.md](PLAN.md) · Paper: [PAPER_OUTLINE.md](PAPER_OUTLINE.md)

**Mission.** Your zone is what cannot be delegated: decisions, unblocking the team, the final text of the paper — and **the A2 core end-to-end: design, capture code, probes, analysis** (the computational runs for A2 are executed by M2).

**Context in 3 bullets:** pilot is frozen and published ([../../results/pilot_summary.md](../../results/pilot_summary.md)); the paper deadline is Sun Aug 2, 21:00 MSK on OpenReview; the plan has one experimental core (probing transfer, A2) and one safety floor (plan-minimum, PLAN §5).

## Stage A (Jul 18–29) — org, three decisions, and the probes

1. **Org weekend (Jul 18–19), ~2 h.**
   - Create the Yandex DataSphere project, grant access to M2/M3/M4 (5,000,000 units).
   - Message the team: everyone registers on OpenReview **now** (a submission with an unregistered author = rejected).
   - Send the letter to Amina (draft in local `discussions/`; asks for probing-code details + a possible Zaytsev consult).
   - Forward the reading plan (PLAN §6) — reading has not started yet; it starts now.
2. **Decision 1 — scale-set composition (Mon–Tue Jul 20–21, ~30 min).** M2 brings the intersection counts: how many native SQuAD 2.0 C/A items land on our audio, and whether the NMSQA slice materialized (threshold ≥ ~100 C on natural audio). You approve the sizes and the slice inclusion; log in [../decisions.md](../decisions.md).
3. **Decision 2 — probe design (Wed Jul 22, ~1 h).** After reading paper 24, log in decisions.md: which layers, which pooling, k for soft targets, the cross-validation protocol (agree with M4). The target event is already fixed (Jul 22 entry): `incorrect(a) = 1` if (C and `label == answer`) or (A and `label == answer` and `correct == False`); refusals and hedges are always 0. This is the design record for the team and the hand-over insurance.
4. **The A2 core — your main part (Jul 23–29).** **Thu Jul 23:** finish reading paper 24 and write the representation-capture code — **M2 must have it by the evening** so she can run it on Friday. Capture all candidate layers {8, 16, 24, 32} and both aggregation variants (last token + mean) at once: then picking the best layer becomes analysis after the fact rather than a blind decision. **Fri Jul 24:** M2 delivers tensors and samples, M4 the soft labels; you sanity-check the first tensors (shapes, no NaNs). **Sat Jul 25:** train linear probes per layer (logreg + CV per M4's protocol) and build the AUROC table for the checkpoint. **Then:** the attention probe (runs and debugging with M2) and the mitigation analysis (operating points vs plain / S1 / entropy). Insurance: if you are unavailable — M2 trains the linear probes from the design record of step 3.
5. **Decision 3 — checkpoint (Sat Jul 25, ~1 h).** Look at the scale-set table + your own first AUROC. Rule: no working AUROC → switch to plan-minimum (PLAN §5) and announce it. Continuing A2 without numbers is not allowed.
6. **Stop (Wed Jul 29).** Announce the experiment stop; whatever numbers exist go to the paper.

## Stage B (Jul 23 → Aug 2) — the text is yours

- Review M3's Intro/RW draft (Jul 25–27) and M2's Data section — comments, not rewrites.
- Write: **Discussion, Limitations, Conclusion, Abstract** (Jul 30–31). Limitations you already know: 100 items → wide Wilson CIs; single model family; judge is an LLM (94% agreement, audited).
- **Contribution section (mandatory):** collect 2–4 sentences from each member about what *they personally* did; assemble and even out. Do not skip — the school requires it.
- **Sat Aug 1 evening: internal submit** on OpenReview (full PDF, all authors listed). Sun Aug 2: fixes only; final check by 21:00 MSK.

## Reading (by Wed Jul 22)

| Paper | Read for | Depth |
|---|---|---|
| 24 — attention probing | you transfer this method: probe design and code | **deep** |
| 16 — Qwen2-Audio | you write the capture code (encoder → LLM interface) | architecture section |
| 01 — AQUA-Bench | positioning (closest work; not our validation set) | skim |
| 25 — Walking Through Uncertainty | knowing the neighbor lab's work | abstract + tables |

## Done when

DataSphere live · all 4 on OpenReview · letter sent · 3 decisions logged in decisions.md · probe AUROC table exists (yours, or M2's from your spec) · Contribution section assembled · submission confirmed before 21:00 MSK Aug 2.

**If stuck:** cut scope, not the deadline — the minimum (PLAN §5) always fits.

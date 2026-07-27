# Glossary — Speech-LLM Basics & Every Project Term

**Project:** Can Speech LLMs Recognize When Audio Evidence Is Insufficient? (SMILES 2026)
**This is the canonical glossary** — other documents link here; role files keep only small convenience tables. Русская версия: [GLOSSARY_RU.md](../ru/GLOSSARY.md).

## Part 1 — What is a Speech LLM (read this first if you're new)

- A **Speech LLM** (also called **LALM** — large audio-language model) is a normal text LLM with an **audio encoder** attached. The encoder converts the sound wave into a sequence of embeddings that the LLM reads as if they were extra "tokens". The model doesn't literally "hear" — it reasons over a compressed representation of the audio.
- **Not all such models output audio.** The system types that appear in this project:

| System | Input → Output | Example | Role in our project |
|---|---|---|---|
| ASR (speech recognition) | audio → text transcript | Whisper | Only transcribes; first half of our cascade |
| **Speech LLM / LALM** | audio + text question → **text answer** | Qwen2-Audio, SALMONN | **What we study** |
| "Omni" speech-to-speech | audio + text → text **and optionally speech** (a separate "Talker" module voices the text) | Qwen2.5-Omni, GPT-4o voice | We use only their text output |
| TTS (speech synthesis) | text → audio | edge-tts, Kokoro | Not an LLM; we use it to manufacture dataset audio |
| Cascade | audio → (ASR) → transcript → (text LLM) → answer | Whisper + Qwen2.5 | Our diagnostic baseline |

- **Why the cascade matters:** if the cascade also hallucinates, the failure is in reasoning about evidence, not in "hearing" — it isolates *where* the problem lives.

## Part 2 — Terms

### Systems

| Term | Meaning |
|---|---|
| LLM | Large language model — a text-in, text-out neural network (ChatGPT-style) |
| Speech LLM / LALM | See Part 1: text LLM + audio encoder; input audio + text, output text |
| Audio encoder | The network that turns a sound wave into embeddings the LLM reads as extra "tokens" |
| ASR | Automatic speech recognition: audio → text transcript (e.g., Whisper) |
| Transcript | The written text of what is said in a recording |
| TTS | Text-to-speech synthesis: text → audio |
| Cascade | Two-step pipeline ASR → text LLM |
| "Omni" models | Newer Speech LLMs that can additionally voice their answer |
| QA / spoken QA | Question answering / answering questions about an audio recording |

### Model behaviors we measure

| Term | Meaning |
|---|---|
| Hallucination | A fluent, confident answer not supported by the audio |
| Abstention / refusal / IDK | Explicitly saying the audio doesn't contain the answer ("I don't know") — desired on unanswerable questions |
| Over-refusal | Refusing when the answer IS in the audio — the hidden cost of aggressive abstention |
| Hedge | A vague answer that neither commits nor clearly refuses |
| Epistemic awareness | The ability to tell apart: stated in the audio / inferable from it / impossible to determine (our categories A/B/C) |

### Methods — the Speech LLM itself always stays frozen: prompts, sampling, and (phase 2) small external probes; no LLM weights changed

| Term | Meaning |
|---|---|
| Prompt / prompting strategy | The text instruction given to the model / how we phrase it |
| CoT (chain-of-thought) | Prompting the model to reason step by step before answering; MCoT = its multimodal variant |
| Self-consistency (S4) | Ask the same question k times with sampling randomness (temperature > 0); disagreement between the answers signals guessing |
| Self-verification / CoVe (S3) | Two-stage: the model answers, then is asked to check whether its own answer is supported by the audio |
| Verbalized confidence (S5) | The model states a 0–100 confidence next to its answer; below a threshold ⇒ abstain |
| IDK-prompt (S1) | Instruction that explicitly allows/requires saying "the audio does not provide that information" |
| Explicit option (S2) | "Cannot be determined from the audio" offered as a legitimate answer choice |
| Temperature | A randomness dial; above 0 the model can answer differently each time |
| SFT | Supervised fine-tuning — training on labeled examples (we do **not** train in the first phase) |
| LoRA | Low-rank adapters — a cheap fine-tuning technique; stretch goal only |
| LLM-as-judge | A separate LLM grades free-form answers against the gold (reference) answer using a fixed rubric |
| Rubric | The fixed grading instruction, same for every answer, stored in a versioned file |
| MCQ | Multiple-choice question format; we deliberately use free-form answers instead |
| Hidden states / representations | The model's internal number vectors for each token at each layer; readable without changing the model |
| Probing / linear probe | A tiny classifier (logistic regression) trained on frozen hidden states to predict something — here: "will the answer be a hallucination?" |
| Attention probe | A probe with a small learned attention layer that reads *all* prompt-token representations instead of one vector (method of paper 24) |
| Soft targets | Probe training labels built by sampling k answers per item and grading them: the *fraction* hallucinated (0…1) instead of a hard 0/1 |
| Pre-generation detection | Predicting a coming hallucination from the model's internal state *before* it writes the answer; contrast: output-level methods look at generated answers |
| Probe→abstain (our mitigation) | If the probe's hallucination score is above a threshold, output "I don't know" instead of the model's answer |
| Entropy baseline | Output-level detector we compare against: sample k answers and measure their disagreement; high disagreement signals guessing |

### Metrics

| Term | Meaning |
|---|---|
| Hallucination rate | Share of unanswerable questions the model still answered |
| Correct-abstain rate | Share of unanswerable questions the model correctly refused |
| Over-refusal rate | Share of answerable questions the model wrongly refused |
| Precision / Recall / F1 | Standard classification scores: exactness / completeness / their harmonic mean |
| Selective prediction | The setting where a model may abstain; quality = how accurate it is on what it does answer vs how much it answers |
| Risk–coverage curve, AURC | Error rate (risk) vs share of questions answered (coverage) as the abstention threshold varies; AURC = area under that curve, lower is better |
| Calibration | Whether stated confidence matches how often the model is actually right |
| RGI | Reliability Gain Index — metric from "Towards Reliable LALM" (paper 02) for comparing refusal-improving methods |
| Wilson confidence interval (CI) | The honest range around a rate given the sample size; wide with few items — we show it anyway |
| Inter-annotator agreement | How often two human labelers assign the same label — a data-quality check |
| AUROC | Area under the ROC curve — how well a signal separates two classes (used for detectors) |
| Operating point | One chosen trade-off on a curve — e.g., the threshold giving X% hallucination at Y% over-refusal |

### Data & engineering

| Term | Meaning |
|---|---|
| SQuAD / SQuAD 2.0 | Classic text QA dataset; 2.0 adds deliberately unanswerable questions; Spoken-SQuAD = its audio version (passages read by TTS) |
| NMSQA | Another spoken version of SQuAD (for textless spoken QA); its *test* split is read aloud by 60 human speakers — the source of our natural-speech slice |
| LibriSpeech | 1000 hours of audiobook speech with transcripts — natural (non-TTS) audio |
| Hugging Face (HF) | Public hub hosting models and datasets; `transformers` / `datasets` are its Python libraries |
| JSONL | A text file where every line is one JSON record |
| Manifest | The JSONL file listing all test items |
| Diagnostic set | Pilot-100: small, hand-verified, with the full A/B/C categories and C1–C4 subtypes — used to analyse behaviour in detail |
| Scale set | The large set used to test the findings at scale: native SQuAD 2.0 questions over NMSQA natural speech (~500 items, A and C only) |
| Twin subset | The 26 paragraphs that have both a natural recording and our TTS version — enables a "TTS vs natural speech" comparison on identical questions |
| 30-second window | Qwen2-Audio processes only the first 30 s of a recording (the rest is discarded), while the cascade sees the full transcript via Whisper; sets are aligned to this boundary, otherwise the systems get different inputs |
| Schema | The agreed set of fields every record must have; never changed silently (canonical schemas: [PLAN.md](PLAN.md) §2) |
| Gold answer / span | The reference correct answer / the exact substring of the transcript containing it |
| Adversarial | Written specifically to fool the model — e.g., unanswerable questions that look answerable |
| Inference harness | Our runner: feeds audio + question to a model under a chosen strategy and logs outputs |
| Wrapper | Code hiding one model's quirks behind an interface common to all models |
| Run | One pass of one model with one prompt strategy over the whole dataset |
| Smoke test | Minimal quick run proving the pipeline works end-to-end |
| Pilot | Small first version (~100 items) validating the design before scaling |
| Data card | One-page dataset description: schema, construction, limitations |
| Ablation | Removing/replacing one component to measure its contribution (e.g., cascade vs end-to-end) |
| Colab | Google Colaboratory — free cloud GPU notebooks; our fallback until cluster GPUs arrive |
| Logprob | Token probabilities from the model — a possible uncertainty signal |
| int8 quantization | Storing weights in 8 bits to fit a model into less GPU memory |
| Dev set | A handful of examples labeled by hand first, used to tune and sanity-check automatic grading |
| Freeze | Declaring a dataset final: after this, only documented new versions |
| W&B | Weights & Biases — experiment-tracking service (optional) |

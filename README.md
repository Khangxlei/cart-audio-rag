# CART — Confidence-Aware ReTrieval for Audio RAG

A training-free retrieval layer for spoken-document RAG. It uses the speech
recognizer's **own uncertainty** to retrieve the right audio segment even when
the key term in that segment was transcribed wrongly.

**📄 [Project page with full write-up and results](https://khangxlei.github.io/cart-audio-rag/)** ·
**▶ [Interactive demo](https://khangxlei.github.io/cart-audio-rag/demo.html)** ·
**▤ [Generated report](results/REPORT.md)** ·
**⎇ [Run log and provenance](docs/RUN_LOG.md)**

---

## What problem this solves

Speech recognition errors are not spread evenly. They concentrate on **named
entities, technical terms and numbers** — exactly the words users search for.
On this collection, entities are 17% of tokens but carry 2–4× the error rate of
everything else (11.0% vs 2.8% on clean audio; 49.4% vs 34.7% at 37% WER).

A standard cascade transcribes audio, keeps only the 1-best text, and throws
away two things the recognizer knew: the **per-word confidence** (AUC 0.85 for
spotting an error) and the **competing beam hypotheses**, which often contain
the correct spelling. CART keeps both.

## The method in one screen

Given one recognition pass over a segment, CART builds three index channels and
one score.

| Channel | What goes in | Weight |
|---|---|---|
| **Text** | the 1-best transcript | BM25, weight 1 |
| **Alternative spellings** | beam alternatives for words with `p < τ` | `α = 0.5`, and they do **not** inflate document length |
| **Phonetic** | phonemes of low-confidence terms, matched by articulatory-feature edit distance | gated by segment confidence |

```
s = s_text + λ · (1 − c) · s_phonetic

  c = mean word confidence of the segment
  τ = 0.8    confidence below which a word gets expanded
  α = 0.5    posting weight for alternative spellings
  λ = 2.0    how far the phonetic channel is trusted
  θ = 0.55   minimum phonetic similarity to count as a match
```

Three design choices matter:

- **Document length comes from the 1-best only**, so a passage cannot win by
  accumulating expansion postings — which BM25's length normalisation would
  otherwise reward.
- **The phonetic channel is a fallback.** It is suppressed for any document
  whose 1-best already contains the query term, so it can only add matches
  where surface retrieval already failed.
- **Nothing is ever overwritten.** The 1-best stays primary; alternatives are
  secondary, down-weighted evidence, so a wrong recognition is never committed
  to twice.

Phonetic distance is feature-weighted rather than symbolic, because a plain
edit distance scores `S EH L` vs `S EY L` (cell/sail, a real confusion) exactly
as it scores `S EH L` vs `S P L` (not a confusion). Weighting substitutions by
place/manner/voicing and height/backness/rounding gives cell/sail **0.84** and
cell/table **0.43**.

### Getting N-best out of faster-whisper

The hardest engineering problem here. `faster-whisper` exposes only the 1-best,
and although its CTranslate2 backend can return N-best, the wrapper never asks
and the relevant attributes are read-only. [`cart/asr.py`](cart/asr.py) wraps the
inner model in a delegating proxy that injects `num_hypotheses` into each
`generate` call and captures the full result:

```python
inner = self.model.model
class _Proxy:
    def __getattr__(self, name):
        return getattr(inner, name)
    def generate(self, encoder_output, prompts, **kw):
        if kw.get("beam_size", 1) > 1:
            kw["num_hypotheses"] = cfg.num_hypotheses
        res = inner.generate(encoder_output, prompts, **kw)
        captured.append(res[0])
        return res
self.model.model = _Proxy()
```

Hypothesis 0 is identical to the 1-best, so concatenating hypothesis *k* across
the 30-second windows yields an alternative transcript alignable to the primary
one with `difflib.SequenceMatcher`, which recovers per-position competing
spellings.

## Results in brief

MRR on 4,021 held-out test queries over 1,896 spoken passages. `*` = significantly
better than BM25 / 1-best (two-sided paired bootstrap, 5,000 resamples; every
starred result reaches *p* < 2×10⁻⁴).

| System | 4.2% WER | 5.7% WER | 10.9% WER | 37.2% WER | ms/query |
|---|---|---|---|---|---|
| BM25 / reference transcript *(upper bound)* | 0.8081 | 0.8081 | 0.8081 | 0.8081 | 0.06 |
| BM25 / Whisper 1-best *(baseline)* | 0.7892 | 0.7760 | 0.7521 | 0.5736 | 0.06 |
| Uniform N-best fusion | 0.7961* | 0.7891* | 0.7646* | 0.5879* | 0.07 |
| Dense retrieval / 1-best | 0.6984 | 0.6955 | 0.6712 | 0.5121 | 3.96 |
| **CART** | **0.7960*** | **0.7883*** | **0.7694*** | **0.6046*** | 81.90 |

- CART beats the standard cascade at every noise level, and **the margin grows
  with error rate**: +0.007 → +0.012 → +0.017 → **+0.031** MRR.
- It overtakes untargeted N-best fusion from 10.9% WER upward, while using a
  **smaller index** (+11.6% postings vs +15.7% at the hardest condition).
- The gain is ~1.6× larger on queries mentioning a named entity, which is the
  mechanism the method claims.

### What the ablations say

| Variant | 4.2% WER | 37.2% WER | ms/query |
|---|---|---|---|
| CART, full | 0.7960 | 0.6046 | 81.2 |
| − phonetic channel | 0.7937 | 0.5848 | 0.06 |
| − alternative-spelling index | 0.7913 | 0.5949 | 81.0 |
| − confidence **gate** (drop `1−c`) | 0.7511 | **0.4639** | 81.1 |
| − confidence **targeting** (τ=1, expand all) | 0.8027 | 0.6195 | 92.8 |

Two findings, and the second is negative:

1. **The confidence gate is the most important component in the system.**
   Removing the `(1 − c)` factor costs 0.141 MRR at 37% WER — an order of
   magnitude more than removing either retrieval channel. The λ sweep shows why:
   trusted unconditionally, the phonetic channel *destroys* retrieval at high
   WER (λ=8 → MRR 0.291, λ=16 → 0.062).
2. **Confidence targeting does not improve accuracy.** Expanding every word
   (τ=1) scores higher in all four conditions, and the τ sweep rises
   monotonically to 1.0 with no interior optimum. What τ buys is latency:
   106.7 ms → 36.6 ms per query on clean audio for 0.0067 MRR.

So the proposal conflated two mechanisms: **scaling by confidence is essential,
thresholding on confidence is a speed knob.** That distinction is the main
scientific contribution here.

### Downstream QA

No CART/baseline difference in answer quality is significant — with 400
questions per condition the F1 interval is ±0.043 against effects of
0.003–0.010, so the stage is **underpowered**, not negative. It should be rerun
on the full test split.

The informative number is the oracle gap: at 37.2% WER a perfect transcript
doubles token-F1 (0.3311 → 0.6632). Below ~10% WER retrieval is the bottleneck
and a retrieval fix helps; at 37% WER the bottleneck has moved into the
transcript itself, which no retrieval layer can recover.

One downstream result *is* adequately powered and negative: **Alternative-Context
Generation hurts.** Handing the reader `{cell|sail} membrane` costs −0.050
token-F1 on clean audio (*p* = 0.003). Alternatives belong in the index, not the
prompt.

Full tables, confidence intervals and figures: **[results/REPORT.md](results/REPORT.md)**.

## Install

Python 3.11. A CUDA GPU is required for the ASR and QA stages; retrieval and
ablations run on CPU.

```bash
git clone https://github.com/Khangxlei/cart-audio-rag.git
cd cart-audio-rag
python -m venv env && source env/bin/activate      # or conda
pip install -r requirements.txt
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')"
```

Versions in `requirements.txt` are pinned deliberately and the comments explain
why — Qwen2.5 needs bf16 (so torch ≥ 2.1), `sentence-transformers` 3.0.1 needs
`huggingface_hub` < 0.25, and `pyarrow` must stay at 17.x to work with
`numpy < 2`. A floating `pyarrow` is what killed the first pipeline run.

### Verify the environment before running anything

```bash
python scripts/validate_env.py
```

Ten checks in ~30 seconds: parquet reading, audio degradation, SQuAD loading,
grapheme-to-phoneme (with **assertions on actual phoneme output**, not just
absence of a crash), pathological-token handling, retrieval, the dense model,
the reader tokenizer and Whisper N-best extraction.

That last point is not paranoia. A missing NLTK resource once made G2P return
empty phonemes for every word, which scores phonetic similarity 1.00 for all
pairs and silently disables half the method — while every downstream number
still looks plausible. The validator now asserts `g2p("cell") == ("S","EH","L")`
and that cell/sail scores above cell/table.

## Data

**[Spoken-SQuAD](https://huggingface.co/datasets/AudioLLMs/spoken_squad_test)**
(`AudioLLMs/spoken_squad_test`) — SQuAD v1.1 dev passages read aloud, giving
spoken documents with text questions and answers.

| | |
|---|---|
| Passages | 1,896 |
| Audio | 32.39 hours |
| Mean passage length | 122.9 words |
| Queries | 5,332 → 1,311 dev / 4,021 test |

The released dataset carries **no passage IDs**, so the retrieval collection has
to be reconstructed: [`scripts/01_prepare_data.py`](scripts/01_prepare_data.py)
de-duplicates the audio by content hash and re-attaches gold transcripts via
SQuAD v1.1 dev. Document IDs are **content-addressed** (`d_<md5[:12]>`) and
query IDs are hash-derived, so the collection can grow without invalidating
cached ASR or shuffling the dev/test split.

Four acoustic conditions are produced from the same audio by adding babble noise
at a target SNR (`cart/audio.py`), giving a WER sweep that spans the regime the
method is motivated by:

| Condition | SNR | WER | NE-WER |
|---|---|---|---|
| `clean` | — | 4.2% | 11.0% |
| `snr5` | 5 dB | 5.7% | 14.3% |
| `snr0` | 0 dB | 10.9% | 21.5% |
| `snr-5` | −5 dB | 37.2% | 49.4% |

## Running the experiments

### Locally / single machine

```bash
python scripts/01_prepare_data.py                      # build the collection
for c in clean snr5 snr0 snr-5; do
  python scripts/02_run_asr.py --condition "$c"        # GPU; resumable
done
python scripts/03_tune_cart.py                         # tune τ, λ, α, θ on dev
python scripts/04_eval_retrieval.py                    # the main experiment
python scripts/06_ablations.py --conditions clean snr5 snr0 snr-5
python scripts/05_eval_qa.py --conditions clean snr0 snr-5 --max-queries 400 --judge
python scripts/07_report.py                            # REPORT.md + figures
```

Every stage writes exactly one artifact and **skips work already done**:

| Stage | Artifact | Resumability |
|---|---|---|
| `01_prepare_data.py` | `data/corpus.jsonl`, `data/queries.jsonl` | deterministic, content-addressed |
| `02_run_asr.py` | `runs/asr/<model>/<condition>.jsonl` | skips transcribed documents |
| `03_tune_cart.py` | `results/tuning.json` | — |
| `04_eval_retrieval.py` | `results/retrieval.json`, `results/rr_*.npy` | — |
| `06_ablations.py` | `results/ablations.json` | **checkpoints per condition** |
| `05_eval_qa.py` | `results/qa.json` | **checkpoints per condition** |
| `07_report.py` | `results/REPORT.md`, `results/fig_*.png` | `exists()`-guards optional stages |

`07_report.py` tolerates missing ablations or QA, so a partial run still yields a
readable report.

### On a Slurm cluster

The run of record used the WPI Turing cluster. Job scripts are in
[`scripts/slurm/`](scripts/slurm/):

```bash
bash scripts/cluster_setup.sh                 # env + prefetch assets on the login node
sbatch scripts/slurm/01_prep.sbatch
sbatch --dependency=afterok:<prep> scripts/slurm/02_asr.sbatch    # job array, 4 conditions
sbatch scripts/slurm/03_eval.sbatch
```

`scripts/slurm/03b_eval_resume.sbatch` re-runs only the stages whose artifact is
missing, and is designed to be **chained** so long work survives short walltime
limits:

```bash
prev=<first job>
for i in 1 2 3 4; do
  prev=$(sbatch --parsable --dependency=afterany:$prev --time=03:00:00 \
                scripts/slurm/03b_eval_resume.sbatch)
done
```

Three cluster lessons are baked into those scripts, each of which cost a failed
run (see [`docs/RUN_LOG.md`](docs/RUN_LOG.md)):

- **Short walltimes get scheduled.** An 8-hour request sat at `PENDING` with an
  estimated start 13 hours out; cutting it to 3 hours started it in 40 seconds.
  A user can lower a running job's limit but never raise it, so chains of short
  self-resuming jobs beat one long job.
- **CTranslate2 needs cuDNN 9 on the loader path.** The libraries ship inside the
  pip `nvidia-*` wheels but nothing exports them; `scripts/slurm/env.sh` prepends
  them to `LD_LIBRARY_PATH`.
- **Exclude Blackwell GPUs under CUDA 12.1.** sm_120 cards fail with
  `CUBLAS_STATUS_NOT_SUPPORTED`.

Jobs run with `HF_HUB_OFFLINE=1`; assets are prefetched on the login node by
`scripts/prefetch.py`.

## Repository layout

```
cart/
  asr.py          N-best extraction from faster-whisper + per-word confidence
  phonetics.py    g2p + articulatory-feature phonetic distance
  index.py        BM25 with weighted expansion postings; phonetic index
  retrieval.py    the CART scorer and all baselines
  audio.py        babble-noise degradation at a target SNR
  data.py         Spoken-SQuAD collection reconstruction
  diagnostics.py  WER by token class, confidence AUC, index cost
  metrics.py      retrieval/QA metrics, bootstrap CIs, paired bootstrap tests
  qa.py           reader LLM, prompt construction, LLM judge
  evaluate.py     system construction and evaluation loops
  config.py       all hyperparameters in one place
scripts/          01→07 pipeline stages, env validation, Slurm jobs
docs/             project page (GitHub Pages), run log, design decisions
results/          committed findings: REPORT.md, *.json, figures
```

## Limitations

- **Latency.** 81.9 ms/query against 0.06 ms for BM25, essentially all of it the
  phonetic channel, which buys only 0.0198 MRR at 37% WER.
- **Confidence targeting does not improve accuracy** (above). τ is justified as a
  latency control only.
- **The QA evaluation is underpowered** at 400 questions per condition.
- **The dense baseline is untuned** off-the-shelf `all-MiniLM-L6-v2`, and is the
  weakest system everywhere — that reflects on this configuration, not on
  learned retrieval generally.
- **SpeechRAG is substituted, not reproduced.** Its speech encoder was never
  publicly released; a dense retriever over the 1-best stands in, keeping a
  learned retriever in the comparison. Reported as a substitution.
- **One recognizer, one language, one noise type** — `small.en` with additive
  babble on read English. Larger recognizers have better-calibrated confidence
  and fewer entity errors, which would shrink the headroom CART exploits.

## Citation

```bibtex
@misc{le2026cart,
  title  = {CART: Confidence-Aware ReTrieval for Audio RAG},
  author = {Le, Khang},
  year   = {2026},
  note   = {DS547 Information Retrieval, Worcester Polytechnic Institute},
  url    = {https://khangxlei.github.io/cart-audio-rag/}
}
```

## License

MIT for the code in this repository. Spoken-SQuAD and SQuAD v1.1 carry their own
licenses (CC BY-SA 4.0 for SQuAD); the audio is not redistributed here and is
downloaded from Hugging Face at prepare time.

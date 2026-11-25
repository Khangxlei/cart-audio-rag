# Overnight handoff — 2026-11-24, 23:55

State of the CART evaluation when the session was closed for the night, and
exactly what to check in the morning.

## What is already finished and safe on disk

| Artifact | Contents |
|---|---|
| `runs/asr/small.en/*.jsonl` | 7,584 transcripts, four noise conditions, verified |
| `results/tuning.json` | hyperparameter search, all four conditions |
| `results/retrieval.json` | **the main experiment** — 5 systems x 4 conditions, 4,021 test queries, bootstrap CIs, paired significance tests, NE/non-NE split, index cost |

These live in `/home/kle6/cart`, not on `/scratch`, so a scratch purge cannot
touch them. The headline result is secured: CART beats BM25-on-1-best at every
noise level, by +0.007 MRR at 4.2% WER rising to +0.031 at 37.2% WER, and
overtakes untargeted N-best from 10.9% WER upward with a smaller index.

## What is still running

Stages left: **ablations** (3 of 5), then **QA** (4 of 5), then the
**report** (5 of 5).

A chain of five jobs covers it, each 3 hours, each starting when the previous
one ends for any reason (`--dependency=afterany`):

```
2591868  RUNNING  started 23:16, wall 02:16   gpu-5-11
2592136  queued   after 2591868
2592137  queued   after 2592136
2592138  queued   after 2592137
2592139  queued   after 2592138
```

Three hours was chosen deliberately: an 8-hour request sat at
`PENDING (Priority)` with an estimated start 13 hours out, and cutting it to
3 hours started the job within 40 seconds. Short jobs backfill; long ones wait.

Every stage is **self-resuming**, so a job that hits its wall loses at most one
condition and the next link continues:

- `03_tune_cart.py`, `04_eval_retrieval.py` — skipped entirely, artifacts exist
- `06_ablations.py` — checkpoints `ablations.json` after each condition, skips
  conditions already present (guarded by a config-equality check)
- `05_eval_qa.py` — same, for `qa.json`, keyed on reader and top_k
- `07_report.py` — runs every time, and `exists()`-guards the optional stages,
  so it produces a `REPORT.md` from whatever is present

## Morning check — run this first

```bash
ssh kle6@turing.wpi.edu
export PATH=$PATH:/cm/shared/apps/slurm/current/bin
cd /home/kle6/cart
ls -la results/                       # REPORT.md present?
tail -40 logs/03b_eval_resume.out     # stage progress
squeue -u kle6 -n cart_eval_resume    # anything still queued?
sacct -S 2026-11-24 -u kle6 -o JobID%14,JobName%18,State%14,Elapsed,ExitCode
```

**Success looks like:** `results/REPORT.md` exists, plus `ablations.json`,
`qa.json`, and three `fig_*.png`. The log ends with `ALL STAGES COMPLETE`, and
the later jobs in the chain finish in a minute or two printing `SKIP` /
`already complete, skipping` for everything.

**Partial success:** `REPORT.md` exists but has no ablation or QA section. That
means the chain ran out before those stages finished. The retrieval results in
it are still complete and correct. Resubmit one more link:

```bash
sbatch --time=03:00:00 scripts/slurm/03b_eval_resume.sbatch
```

**Failure to investigate:** the chain has drained (nothing queued) and there is
no `REPORT.md`. Read `logs/03b_eval_resume.out` from the top. The two failure
modes seen so far are the file-count quota (symptom: a job `FAILED` with
`Elapsed 00:00:00` and an empty log) and a dead compute node.

## Known risks overnight

1. **File-count quota.** The account cap is 2,000,000 entries and it was
   *exactly* full earlier today; clearing caches left ~170k of headroom. If an
   unrelated job of yours creates a few hundred thousand files, writes fail
   again. Check with:
   `getfattr --only-values -n ceph.dir.rentries /home/kle6`
2. **`/scratch` purge.** `data/audio` (3.5G) and the full model cache live
   there. Nothing the remaining stages need depends on it any more: the reader
   and embedding models were copied into `/home/kle6/cart/.hf_cache` and
   verified to load with `HF_HUB_OFFLINE=1`. A purge would only cost the audio,
   which is re-downloadable and no longer needed because the transcripts exist.
3. **Blackwell nodes.** `gpu-6-[01-20]` are excluded on every job; CUDA 12.1
   cannot drive sm_120.

## The open scientific question

The ablation table is what the remaining compute is for. The diagnostics set up
a specific, falsifiable prediction: the share of substitution errors recoverable
from beam alternatives **collapses** with noise (35% clean -> 7% at 37% WER),
yet CART's advantage is **largest** at 37% WER. So the gain there cannot be
coming from alternative expansion; it should be the phonetic fallback.

Look for this in `results/ablations.json`, condition `snr-5`:

- if the prediction holds: `-phonetic` drops sharply, `-alt_index` barely moves
- `-conf_targeting` (tau = 1.0, expand everything) is the direct test of the
  proposal's core claim. Tuning preferred tau = 0.8 at the grid edge in all four
  conditions, which hints that targeting may not be earning its place, while
  the retrieval results — where CART pulls away from untargeted N-best exactly
  as WER rises — point the other way. This row settles it.

Either outcome is a result. A tau = 1.0 win would be a genuine negative finding
about confidence targeting, and the diagnostics above explain why it happened.

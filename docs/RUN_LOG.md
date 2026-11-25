# CART — run log and provenance

Record of how the reported results were produced: what ran where, what broke,
and what was changed in response. Kept so the experiments can be reproduced and
so the methods write-up can cite real job IDs rather than recollection.

## Environment of record

| | |
|---|---|
| Host | WPI Turing, `kle6@turing.wpi.edu`, Slurm account `kmlee` |
| Project root | `/home/kle6/cart` |
| Python env | `./env` (conda, Python 3.11) built from `requirements.txt` |
| Partitions | `quick,short` (the `academic` partition rejects this account) |
| Excluded nodes | `gpu-6-[01-20]` — RTX PRO 6000 Blackwell, unusable under CUDA 12.1 |
| Network | assets prefetched on the login node; jobs run `HF_HUB_OFFLINE=1` |

All development and smoke-testing happened earlier on a local RTX 3060; **no
reported result comes from the laptop.** Local partial output was discarded so
every number derives from one consistent cluster environment.

## Dataset of record

Produced by `scripts/01_prepare_data.py` (job 2591149, `PREP COMPLETE`):

```
passages            1896
queries             5332   (dev 1311 / test 4021)
audio               32.39 hours
mean passage len    122.9 words
```

This is the full Spoken-SQuAD test split. The corpus was reconstructed by
de-duplicating the released audio by content hash (the HF release carries no
passage ids) and re-attaching gold transcripts via SQuAD v1.1 dev. Document ids
are content-addressed, so the corpus can grow without invalidating cached ASR.

## Job ledger

| Stage | Job | Outcome |
|---|---|---|
| probe | 2452239 / 2452358 | completed / cancelled — partition reconnaissance |
| prep | 2591073 | **FAILED** — pyarrow/numpy conflict |
| asr | 2591074 | never ran (`DependencyNeverSatisfied`) |
| prep | 2591118 | **COMPLETED** (01:07) — finished before the partition switch took effect |
| prep | 2591149 | **COMPLETED** (00:51) — dataset of record |
| asr | 2591150_[0-3] | **FAILED** (all 4) — cuDNN 9 not on loader path |
| gputest | 2591179 | **COMPLETED** — validation passed on L40S |
| asr | 2591181_0 | **FAILED** — Blackwell cuBLAS |
| asr | 2591181_1 | **COMPLETED** (41:36) — snr5, H100 `gpu-5-12` |
| asr | 2591188_0 | **COMPLETED** (32:54) — clean, L40S `gpu-5-28` |
| asr | 2591188_3 | **COMPLETED** (33:57) — snr-5, L40S `gpu-5-28` |
| asr | 2591188_2 | snr0 — requeued after `gpu-5-12` died; resumes from doc 625 |
| eval | 2591189 | **FAILED** (00:00:10) — `inflect.NumOutOfRangeError` on a 334-digit token |
| eval | 2591698 | **FAILED** (00:00:00, empty log) — could not create its log file; home file-count quota exhausted |
| eval | 2591765 | cancelled — stale duplicate |
| eval | 2591774 | **CANCELLED** at 02:09 by choice — tuning + retrieval done; ablations would not fit in its 3 h |
| eval | 2591868 | **TIMEOUT** at 3 h — ran part of the ablations, checkpointing each condition |
| eval | 2592136 | **COMPLETED** (01:13) — finished the ablations, ran QA, wrote `REPORT.md` |
| eval | 2592137-9 | **COMPLETED** (~15 s each) — chain tail; every stage skipped as already complete |

Both prep jobs completed; the corpus is deterministic (content-addressed
document ids), so the second run reproduced the first exactly. 2591149 is cited
as the job of record.

`snr0`'s requeue was **node failure, not preemption**: `gpu-5-12` went
`DOWN+NOT_RESPONDING`. The QOS matrix shows `quick` and `short` have identical
preemption standing (both preempt only `scavenger`, neither preempts the
other), so no partition choice would have prevented it. The resumable design
absorbed it — all 625 completed records were verified parseable, with no
truncated final line and no duplicate document ids.

## Outcome

The chain finished unattended. `results/REPORT.md` was written 2026-11-25
07:43, with `ablations.json`, `qa.json` and three figures beside it. The
per-condition checkpointing is what saved it: 2591868 hit its 3 h wall partway
through the ablations, and 2592136 resumed from the completed conditions
instead of restarting the stage.

Headline numbers (full tables in `results/REPORT.md`):

- CART beats BM25-on-1-best at every noise level, +0.007 MRR at 4.2% WER rising
  to +0.031 at 37.2% WER, and beats untargeted N-best fusion from 10.9% WER
  upward while holding a smaller index (+11.6% postings against +15.7%).
- The motivation reproduces: named entities are 17% of tokens but carry 2-4x the
  word error rate, and CART's gain is ~1.6x larger on queries that mention one.
- **Confidence targeting does not improve accuracy.** The tau sweep rises
  monotonically to tau = 1.0 in all four conditions, with no interior optimum
  anywhere, and the -conf_targeting variant (expand every word) beats full CART
  in every condition at p < 2e-4. What tau actually buys is latency: 106.7 ms to
  36.6 ms per query on clean audio, for -0.0067 MRR. That saving shrinks as
  noise rises (92.8 to 81.2 ms at 37% WER, for -0.0149 MRR) because half the
  words are already below the threshold. So targeting is an accuracy/latency
  knob that pays off on clean audio and stops paying exactly where the method
  was motivated.
- **Confidence weighting, by contrast, is the most important component in the
  system.** Removing the (1 - c) gate costs 0.045 MRR on clean audio and 0.141
  at 37% WER, an order of magnitude more than any other ablation. The proposal
  conflated two mechanisms: scaling the phonetic channel by confidence is
  essential; thresholding on confidence is not.
- The phonetic channel is where all the cost sits. Dropping it costs only 0.0198
  MRR at 37% WER but takes latency from 81.2 ms to 0.06 ms per query.
- The lambda sweep has a real interior optimum that moves with noise (4.0 clean,
  2.0 at 37% WER), and over-weighting is catastrophic at high WER: lambda = 8
  collapses MRR to 0.291 and lambda = 16 to 0.062.
- **The retrieval gains did not reach answer accuracy.** No CART/baseline QA
  difference is significant, and at 37% WER CART is nominally behind (F1 0.3212
  against 0.3311, p = 0.37). With 400 queries the F1 CI half-width is +/-0.043
  against effects of 0.003-0.007, so this stage cannot resolve a retrieval gain
  of that size: underpowered, not negative. The one significant QA result is
  that Alternative-Context Generation *hurts* the reader (-0.050 F1 on clean,
  p = 0.003; -0.033 at 37% WER, p = 0.043).

## Failures and fixes

Each of these cost a queue wait, which is why `scripts/validate_env.py` now
exists: it exercises every import and code path offline in ~30 s.

1. **`ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4`**
   `requirements.txt` pinned `numpy<2` but left pyarrow floating, so pip took a
   build that needs numpy 2. Prep died in 4 s and `--dependency=afterok` left
   the whole chain permanently unschedulable.
   → Pinned `pyarrow==17.0.0`, the last line supporting numpy 1.x.

2. **`Unable to load any of {libcudnn_ops.so.9.1.0, ...}`**
   CTranslate2 links cuDNN 9 at runtime. The libraries were present — they ship
   in the pip `nvidia-*` wheels torch pulls in — but nothing put them on the
   loader path.
   → `scripts/slurm/env.sh` prepends `$SITE/nvidia/cudnn/lib` and
   `$SITE/nvidia/cublas/lib` to `LD_LIBRARY_PATH`.

3. **`RuntimeError: cuBLAS failed with status CUBLAS_STATUS_NOT_SUPPORTED`**
   The task had landed on an RTX PRO 6000 Blackwell (sm_120), which CUDA 12.1
   predates. Affects CTranslate2 and torch alike.
   → `#SBATCH --exclude=gpu-6-[01-20]` on every GPU job.

4. **`inflect.NumOutOfRangeError`** (eval 2591189, died in 10 s)
   Whisper degenerates into long runs of digits on noisy audio — 238 distinct
   such tokens, the longest 334 characters. `g2p_en` spells numbers out through
   `inflect`, which raises past ~10^303.
   → Clamp long digit runs before conversion, cap index terms at
   `MAX_TERM_LEN = 25`, and tolerate a G2P failure *only* on such pathological
   tokens. Regression-tested in `scripts/validate_env.py`.

5. **`Disk quota exceeded` — a file-count quota, not a byte quota.**
   Eval 2591698 reported `FAILED` with `Elapsed 00:00:00`, exit `0:53` and an
   empty log: Slurm could not create the log file at all. The home filesystem
   showed 626 TB free, so the first reading of this was a byte quota, and
   relocating 14.5 GB of caches and audio to `/scratch/kle6/cart` (with
   symlinks back) appeared to fix it. That diagnosis was wrong in kind. The
   CephFS attributes give:

   ```
   ceph.quota.max_bytes = 3365574426500   (used 2.21 TB, 66%)
   ceph.quota.max_files = 2000000         (used 2000000, 100%)
   ```

   The limit is **2,000,000 directory entries**, and the account sits on it.
   The relocation helped only incidentally, by removing ~13 k entries along
   with the bytes. Under this limit `/home/kle6/ondemand/data` (977 k entries,
   1.9 TB) is the standing pressure; `cart` holds ~60 k, of which the conda
   environment is 47 k.
   → Freed 7,670 entries by deleting the pip wheel cache (pure download cache)
   and the conda **package** cache. The latter is safe here because `env/` is
   linked to it by *hardlink*, not symlink — verified before deleting
   (`find env -type l -lname '*conda_pkgs*'` returned 0, and sampled files in
   `env/lib` showed link count 2), so removing the cache drops one reference
   and the environment keeps its own. Imports verified afterwards.
   **Headroom is thin (~7.6 k entries) and the cap is account-wide**, so an
   unrelated job of the user's can exhaust it again mid-run.

   With the account holder's approval, two caches outside this project were
   then cleared as well: `~/.npm/_cacache` and `~/.conda/pkgs`. Both are pure
   download caches, and the same hardlink check was run first across
   `~/.conda/envs`, `~/jupyter_ondemand_env`, `~/venvs` and `cart/env` (zero
   softlinks into `.conda/pkgs`). All four of the account's conda environments
   were confirmed to still launch afterwards. Entries fell 1,994,060 ->
   1,829,690, leaving ~170 k of headroom. `tuning.json` was written four
   minutes later; without this it would have failed and taken 80 minutes of
   tuning with it.

6. **A running job's time limit cannot be raised.**
   `scontrol update jobid=... TimeLimit=08:00:00` returns
   `Access/permission denied`: users may only *reduce* a limit. Since the
   3 h request was itself chosen to get scheduled promptly, the mitigation is
   structural rather than a resubmission — `scripts/slurm/03b_eval_resume.sbatch`
   re-runs only the stages whose artifact (`tuning.json`, `retrieval.json`,
   `ablations.json`, `qa.json`) is missing, and `scripts/07_report.py` already
   guards the optional ones with `exists()`, so a partial run still yields a
   `REPORT.md`.

7. **Two reporting defects in `scripts/07_report.py`, found while reading the
   first significance tests.** Every comparison came back `p = 0.0000`. The
   paired bootstrap itself is correct (it centres the resampled differences at
   zero and counts how often they reach the observed difference); `p = 0` is
   genuine here because CART changes few queries but almost always in the same
   direction, which makes the variance of the paired difference very small.
   Two things still needed fixing:
   - `p = 0` is not a reportable value. With 5000 resamples the test cannot
     resolve below `1/5000 = 2e-4`, which is what the tables now say.
   - The test is two-sided, so a small p means "differs from the reference",
     not "beats it" \u2014 and the report starred anything with `p < 0.05`
     regardless of direction. `dense_1best`, which is significantly *worse*
     than the baseline in all four conditions, would have been starred in
     every row and read as a win.
   \u2192 `fmt()` now takes the reference value and marks a trailing star for
   significantly better, a trailing `v` for significantly worse, with the
   resolution floor stated in each caption.

8. **The ablation stage wrote its artifact only after the last condition.**
   Each condition costs ~30 min (13 index builds plus 21 sweeps over 4021 test
   queries), so a walltime cut discarded every completed condition. This is
   what the 3 h limit on 2591774 was about to do.
   \u2192 `scripts/06_ablations.py` now checkpoints `ablations.json` after each
   condition and skips conditions already present (guarded by a config-equality
   check, so a changed configuration recomputes everything). The resume job
   therefore always invokes the stage rather than testing for the artifact,
   since a partial file would otherwise look complete.

9. **Unattended operation (2026-11-24, overnight).** Two gaps had to close
   before the pipeline could run without supervision.
   - The QA stage had the same all-or-nothing write as the ablation stage.
     `scripts/05_eval_qa.py` now checkpoints `qa.json` after each condition and
     skips conditions already present, keyed on reader and `top_k`.
   - A single safety-net job was not enough, and the one that was queued held a
     *spooled* copy of the old sbatch (Slurm snapshots the batch script at
     submit time, so editing the file does not change an already-queued job).
     It was cancelled and replaced by a five-deep chain, each link 3 h and
     `--dependency=afterany` on the previous one, submitted from the updated
     script: 2591868 -> 2592136 -> 2592137 -> 2592138 -> 2592139.

   The resume script now *always* invokes stages 06 and 05 rather than testing
   for their artifacts, because a checkpointed partial file would otherwise
   look complete and be skipped.

   The reader (`Qwen2.5-1.5B-Instruct`) and embedding (`all-MiniLM-L6-v2`)
   caches were copied out of `/scratch` into `/home/kle6/cart/.hf_cache`, which
   stopped being a symlink, and both were verified to load under
   `HF_HUB_OFFLINE=1`. 40 files, 3.8 GB. Nothing the remaining stages need now
   depends on scratch; the scratch copies were left in place.

10. **Walltime is the scheduling lever, confirmed twice.** After 2591774 was
    cancelled, the 8 h resume job went to `PENDING (Priority)` with an
    estimated start **13 hours out**. Reducing its limit to 3 h started it
    within 40 seconds. Users can lower a limit but never raise one, so the
    right pattern on this cluster is a chain of short, self-resuming jobs
    rather than one long job.

11. **Chained jobs overwrote one another's log.** Every link used the same
    `#SBATCH -o` path, so each job truncated the previous one's log at start.
    The per-condition stdout of the ablation and QA stages run by 2591868 and
    2592136 is gone; only the final no-op job's output survives. No result was
    lost, since every stage's numbers live in its JSON artifact, but the log is
    no longer a full narrative of the run.
    -> Put `%j` in the log path for any future chain.

## Scheduling notes

- `academic` rejects this account (`Invalid account or account/partition
  combination`); its allow-list is course accounts.
- `quick` was effectively empty (0 queued) against ~250 queued on `short`.
- **Cutting the requested walltime was the decisive lever**: a job pinned at
  `PENDING (Priority)` started immediately once its limit went from 4 h to
  1.5 h, because short jobs backfill into gaps the scheduler cannot otherwise
  fill.
- The ASR stage is resumable — it skips documents already transcribed — so a
  preempted `quick` job loses time, not work.

## Deviation from the proposal

SpeechRAG (Min et al., 2025) is named as a baseline, but its trained speech
encoder was never publicly released and cannot be reproduced. A neural dense
retriever over the 1-best transcript stands in, preserving the property that
made it informative: a *learned* retriever in the comparison. Reported as a
substitution, not a reproduction of SpeechRAG.

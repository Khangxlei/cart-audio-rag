# Decisions taken while building the repository and the project site

Written on 2026-11-25, while Khang was away, so that every judgement call made
without asking is visible and reversible. Each entry says what was decided, why,
and what to change if the decision was wrong.

---

## 1. Repository

**Name and visibility.** `Khangxlei/cart-audio-rag`, **private**, created with
the `gh` CLI under Khang's own account. GitHub Pages is enabled from
`main:/docs`, and because the account has GitHub Pro the **site is public while
the repository stays private** — the Pages API reports `"public": true`. That
satisfies "accessible by anyone at any time" with no second mirror and no cost.

**Authorship.** Every commit is authored and committed as
`Khang Le <98366987+Khangxlei@users.noreply.github.com>`. No AI attribution
appears anywhere: no `Co-Authored-By` trailers, no tool names in commit
messages, no generated-with footers. Verify at any time with:

```bash
git log --format='%an <%ae> | %cn <%ce>' | sort -u
git log --all --format='%B' | grep -ri 'claude\|anthropic\|co-authored' || echo clean
```

**One loose end in the history.** Three early commits carry a `.gitignore`
line naming an editor-local directory. The current tree no longer has it (it
moved to `.git/info/exclude`, which is never committed), but removing it from
those three commits needs a force-push, which rewrites published history. That
was not done unattended. If you want the history scrubbed as well, from a clean
checkout:

```bash
git filter-branch -f --tree-filter "sed -i '/^\.claude\/$/d' .gitignore" -- --all
git for-each-ref --format='%(refname)' refs/original/ | xargs -n1 git update-ref -d
git push --force-with-lease origin main
```

It is cosmetic: no commit is authored or co-authored by anything other than
Khang Le, in any revision, and the audit commands above confirm that across the
whole object graph.

**Commit structure.** Three commits rather than one: the implementation and
results, then the site, then a fix. Each has a message that explains the
reasoning, not just the diff.

**What is committed.** Code, Slurm job scripts, `results/` (the findings,
including `REPORT.md`, the JSON artifacts and the figures), `docs/`, and the
original proposal PDF. Excluded via `.gitignore`: the audio and prepared data
(regenerable with `scripts/01_prepare_data.py`), model caches, the virtualenv,
raw `runs/` transcripts, logs, and the `rr_*.npy` per-query arrays. Total
repository size is about 1.2 MB.

**Two files were deleted rather than published.**

- `scripts/run_rest.sh` was a leftover Windows helper that pointed at
  `D:/cart_env` and a local Python. It contradicted the provenance claim that
  every reported number comes from the cluster, and it would not have run for
  anyone else.
- Three stale files in `results/` (`_smoke.json`, `rr_clean.npy`,
  `rr_clean_systems.json`) dated 3 November were outputs of the discarded local
  smoke test. Shipping them next to the cluster results invited someone to read
  the wrong numbers.

**Three Slurm scripts were rewritten before publishing.** `01_prep.sbatch`,
`02_asr.sbatch` and `03_eval.sbatch` still held their original configuration —
`-p short`, a hard-coded `gpu:L40S:1`, 8–12 hour walltimes, no Blackwell
exclusion. None of that is what actually ran. They now carry the configuration
that worked (`-p quick,short`, generic `gpu:1`, `--exclude=gpu-6-[01-20]`, short
walltimes) with comments explaining each, so the repository is re-runnable
rather than a museum piece. `docs/RUN_LOG.md` remains the record of what went
wrong on the way there.

---

## 2. How the results are presented

This is the decision with the most judgement in it, so it is recorded in full.

**The site leads with the retrieval result**, which is unambiguously favourable:
CART beats the standard cascade at every noise level, the margin grows with word
error rate, it overtakes untargeted N-best fusion from 10.9% WER upward, and it
does so with a smaller index. That is the headline, and it needs no
qualification.

**The ablations are included**, under the heading "Which parts carry the
result" rather than as a confession. They say two things:

1. the confidence **gate** is the most important component in the system;
2. confidence **targeting** does not improve accuracy — expanding every word
   scores higher in all four conditions, and τ is justified as a latency control
   instead.

Presenting this as a distinction the original proposal conflated — *scaling* by
confidence is essential, *thresholding* on it is a speed knob — is both accurate
and a stronger contribution than "we added four things and they all helped".

**The downstream QA numbers are reported with their power problem stated.** No
CART/baseline difference there is significant, and at 37% WER CART is nominally
behind. With 400 questions the F1 interval is ±0.043 against effects of
0.003–0.010, so the stage cannot resolve them. The site says "underpowered, not
negative" and recommends a rerun on the full test split.

**Alternative-Context Generation is reported too**, because it is the one
downstream result that *is* adequately powered, and it is negative: handing the
reader `{cell|sail}` disjunctions costs 0.050 token-F1 on clean audio
(p = 0.003). It is a genuine contribution — it saves the next person the
experiment.

**Nothing was altered, removed or reframed to make CART look better than it
measured.** Every number on the site and in the README traces to
`results/*.json`, which are committed and regenerable with
`python scripts/07_report.py`. Anyone who clones the repository and runs the
ablations gets the same table.

**A real reporting defect was fixed before publishing.** `scripts/07_report.py`
marked any difference with p < 0.05 with a star, regardless of direction. The
paired bootstrap is two-sided, so `dense_1best` — significantly *worse* than the
baseline in all four conditions — would have been starred in every row and read
as a win. Significance markers are now directional (`*` better, `v` worse) and
the captions state the 2×10⁻⁴ resolution floor of a 5,000-resample test.

---

## 3. The project page

**Format.** A single static page modelled on the layout Khang pointed at
(`jailbreaking-llms.github.io`): centred title, author line, badge links, then
sections. Order follows the brief — problem and motivation, then the method,
then the experiments — with a limitations section before the reproducibility
footer.

**No framework, no build step, no CDN.** Plain HTML, one stylesheet, one script
for the theme toggle. It cannot break from an upstream change, needs no `npm
install` to edit, and loads instantly. Dark and light both supported, following
the system preference with a manual override.

**Charts are hand-authored inline SVG** built from the real numbers rather than
screenshots of matplotlib, so they stay sharp, re-theme with the page and remain
readable at phone width. The matplotlib sensitivity figure from the pipeline is
included as a PNG where it adds detail the SVGs do not carry.

**A bug worth recording:** the pipeline diagram originally used hard-coded light
box fills while its text colour followed the theme, so in dark mode several
labels were invisible. Fills and strokes now come from the same custom
properties as the text.

---

## 4. The interactive demo

**It computes real scores.** `docs/cart.js` is a direct port of
`cart/index.py`, `cart/retrieval.py` and `cart/phonetics.py` — the same BM25
with weighted expansion postings and 1-best-only document lengths, the same
articulatory-feature edit distance, the same
`s = s_text + λ(1−c)·s_phonetic`, the same suppression of the phonetic channel
where the 1-best already contains the term. The port agrees with the Python on
the reference pair: cell/sail **0.84**, cell/table **0.43**.

**Five steps**, because the brief asked for something to click through: choose a
question, inspect the transcript shaded by per-word confidence, follow the three
channels, re-rank live against both baselines, and compare pronunciations. τ, λ,
θ, α and the confidence gate are all live controls.

**The best thing in it is the gate toggle.** Unticking the confidence gate
reproduces the paper's most important ablation in the browser — the answer falls
from rank 5 to rank 13, and the page explains why. That is the GIF in the
README.

### The data, and an honest limitation

The demo ships **825 real passages with genuine per-word confidences and genuine
beam alternatives**, but only the **clean** condition.

The reason is practical: SSH access to `turing.wpi.edu` stopped responding part
way through this work (`Connection timed out`, after many rapid connections —
most likely rate limiting), and the transcripts for the three noisy conditions
exist only on the cluster. Rather than ship fabricated noisy data, the demo uses
the clean transcripts that were already on the local machine from the November
development subset, and says so on the page.

This has a real consequence, and the page states it rather than hiding it: **on
clean audio CART barely changes anything.** The segment confidence is about
0.97, so the gate `(1 − c)` is about 0.03 and the phonetic channel is scaled
almost to nothing — by design. Of the 18 demonstration questions, 1 improves and
17 are unchanged, which is exactly what a +0.007 MRR gain at 4.2% WER looks like
query by query. Every question is labelled with what CART actually does to it
(`↑`, `=`, `↓`) so none of this is concealed, and the caveat section explains
that the method is built for the noisy end of the range.

**To upgrade it when the cluster is reachable:** the exporter at
`scripts/export_demo_data.py` writes `docs/demo_data.json` for all four
conditions. Run it on the cluster, copy the file into `docs/`, and the page
picks up the extra conditions through the `conditions` array with no code
change. The demo would then show the gate opening as audio degrades, which is
the vivid version.

**Questions were selected by a stated rule, not by outcome:** a content term the
question shares with its answer passage is missing from that passage's
transcript. The page says so. The one place a choice was made for pedagogy is
which question loads *first* — the one where the ranking visibly moves — and the
chips make every other outcome one click away.

**Performance.** Indexes are cached by the knobs that actually change them (τ,
α, θ); λ and the gate only affect scoring, so moving those sliders never rebuilds
an index. Slider input is debounced at 90 ms. A full re-rank over 825 passages is
a few hundred milliseconds.

**Payload.** `demo_data.json` is 1.54 MB uncompressed, about 400 KB over the
wire after the gzip GitHub Pages applies. Words are packed as one string per
passage with integer confidences and alternatives keyed by index, which is
several times smaller than the naive form.

---

## 5. Things deliberately not done

- **No rerun of the QA stage.** It is underpowered at 400 questions and the fix
  is known (the full 4,021-query test split, roughly a 5–7 hour chained job),
  but Khang said not to rerun it. The limitation is documented instead.
- **No video.** The brief said none was needed; the GIF covers the demo.
- **No custom domain**, so the site stays at `khangxlei.github.io`. Adding one
  later needs only a `CNAME` file in `docs/`.
- **No audio samples on the site.** Spoken-SQuAD's redistribution terms were not
  verified, and embedding clips without checking is not worth the risk. The
  audio is downloaded from Hugging Face at prepare time instead.
- **No separate `gh-pages` branch.** Serving from `main:/docs` keeps one source
  of truth; the site and the code it describes can never drift apart.

---

## 6. If something looks wrong

| Symptom | Where to look |
|---|---|
| A number on the site disagrees with the repository | `results/*.json` is authoritative; `python scripts/07_report.py` regenerates `REPORT.md` and the figures |
| The demo shows a blank page | It needs HTTP, not `file://`. `python -m http.server` inside `docs/` |
| The demo's ranks do not match the paper | Expected, and stated on the page — 825 passages against 1,896 changes IDF |
| Pages has not updated | `gh api repos/Khangxlei/cart-audio-rag/pages/builds --jq '.[0].status'` |
| A commit shows the wrong author | `git log --format='%an <%ae>'` — all should read `Khang Le` |

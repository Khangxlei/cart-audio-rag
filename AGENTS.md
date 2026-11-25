# Notes for agents working on this repository

This is the code and project website for **CART** (Confidence-Aware ReTrieval for
Audio RAG), a course research project. The numbers on the site are real
experimental results from a GPU cluster run, so the main risk when changing
things here is **silently invalidating a published result** rather than breaking
a build.

Read `docs/DECISIONS.md` first. It records why the current design is the way it
is, which prevents "improvements" that undo deliberate choices.

---

## 1. Commit identity

Every commit must be authored **and** committed as:

```
Khang Le <98366987+Khangxlei@users.noreply.github.com>
```

No `Co-Authored-By:` trailers, no "generated with" lines, and no AI or tool
names anywhere in commit messages, PR descriptions, or code comments. The
history is clean of these today and must stay that way.

This matters more than it looks: removing a trailer after the fact requires
rewriting history and a force-push, which has already been blocked once on this
repo. Prevention is cheap; the cure is not.

Verify before you finish:

```bash
git log --format='%an <%ae> | %cn <%ce>' | sort -u   # one line, the address above
git log --format='%B' | grep -iE 'co-authored|generated with|claude|codex|gpt|copilot'
```

The second command should print nothing.

## 2. Repository access

The repository is **private**. The published website is **public** anyway, via
the owner's GitHub Pro account (`"public": true` on the Pages config). Do not
change repository visibility, and do not "fix" the private/public mismatch — it
is intentional.

---

## 3. How the site is built and deployed

- Source: **`main` branch, `/docs` folder**, served by GitHub's legacy Pages
  builder. `docs/.nojekyll` is present and must stay.
- **No build step, no bundler, no package manager, no CDN.** Hand-authored
  static HTML, CSS and JS. Keep it that way: a requirement of the project is
  that the site runs for anyone, with no server and no toolchain.
- A push to `main` triggers a rebuild automatically. If the build stalls (it did
  once during a GitHub incident), retrigger it:
  ```bash
  gh api -X POST repos/Khangxlei/cart-audio-rag/pages/builds
  ```
- `.github/workflows/pages.yml` exists but is **deliberately manual-only**. Do
  not enable it on push; a workflow deploy cannot succeed while the Pages source
  is still the legacy builder.

Live URLs:

- <https://khangxlei.github.io/cart-audio-rag/>
- <https://khangxlei.github.io/cart-audio-rag/demo.html>

## 4. Where design changes belong

| File | Role |
| --- | --- |
| `docs/style.css` | All styling for the main page |
| `docs/demo.css` | All styling for the interactive demo |
| `docs/index.html` | Main page structure and content |
| `docs/demo.html` | Demo page structure (shell only; UI is built by `demo.js`) |

There are **no inline `<style>` blocks**. Most visual work is the two CSS files
only. Please keep styling out of the HTML and JS.

### Theming

Colors are CSS custom properties on `:root` in `style.css`, redefined for dark
mode under `@media (prefers-color-scheme: dark)` (guarded by
`:root:not([data-theme="light"])`) and again under `:root[data-theme="dark"]`.
Follow that same three-block pattern for anything new.

**Dark mode is the easiest thing to break here**, and it has broken before: the
pipeline diagram once had hard-coded light box fills with theme-following text,
so it rendered invisible on dark backgrounds. The fix was CSS classes. So:

- Structural SVG fills (boxes, panels, backgrounds, rules) → **CSS classes**,
  never a literal `fill="#fff"`.
- The five hard-coded hex fills remaining in `index.html` are the **categorical
  data-series palette** (`#1a7f4b`, `#e4572e`, `#8b95a1`, `#b07aa1`, `#f0a202`).
  Those are intentionally theme-independent and legible on both backgrounds —
  leave them alone unless you are deliberately reworking the palette, in which
  case check contrast in both schemes.

Always check both light and dark before calling a change done.

---

## 5. Do not touch these

**`docs/cart.js` — treat as frozen.** It is a faithful line-by-line port of the
Python scorer (`cart/phonetics.py`, `cart/index.py`) so the demo can run in the
browser with no backend. The demo's honesty depends on it producing the same
numbers as the experiment. Refactoring it "for readability" can silently change
results. If you must verify it still works, these must hold:

```
phonetic distance  cell/sail  = 0.84
phonetic distance  cell/table = 0.43
config: tau=0.8  lambda=2.0  alpha=0.5  theta=0.55
```

**`docs/demo_data.json` — generated artifact, 1.54 MB.** Produced by
`scripts/export_demo_data.py` from real recognizer output on the cluster. It is
packed and minified on purpose (words as one string, confidences as integer
percents, alternatives keyed by index). Do not pretty-print, reformat, or
regenerate it — the page will still work, but every visitor downloads several
times more data. Do not hand-edit values; they are real `faster-whisper` output.

**The four hand-authored SVG charts in `index.html` encode real measurements in
their coordinates.** Restyling colors, type and spacing is fine. Moving a bar,
point or axis because it "looks better" edits an experimental result. If a chart
needs different data, it comes from `results/`, not from adjusting geometry.

**`results/` and `docs/RUN_LOG.md`** are the experimental record. Read-only.

---

## 6. Content guidance

The site states its own limitations on purpose — the clean-condition demo sample
shows only 1 of 18 queries improving, the QA stage is underpowered, and the
demo currently ships a single condition. These are deliberate and honest. Do not
delete, soften, or rephrase them into claims the results do not support, and do
not add performance claims that are not in `results/`.

The demo also ships only the **clean** condition because the noisy transcripts
live on the cluster. `docs/demo.js` already reads the `conditions` array and
shows a switcher when more than one is present, so adding conditions is a data
file swap, not a code change.

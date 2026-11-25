# CART website redesign

## Repository layout

GitHub Pages serves the static files in `docs/` from `main`. The overview is
`docs/index.html`, and the demo uses `docs/demo.html`, `docs/demo.js`, and its
own stylesheet. Both pages share `docs/style.css`. The demo runs its ranking
in the browser with `docs/cart.js` and the recognizer output in
`docs/demo_data.json`.

The Python research implementation lives in `cart/`: speech recognition,
phonetics, indexing, retrieval, evaluation, and question answering. The pipeline
and export scripts live in `scripts/`, and the experiment outputs and full
report live in `results/`.

## Changes

| File | Change |
| --- | --- |
| `docs/index.html` | Shortened the overview to the problem, three retrieval channels, and the headline result. Added a transcript example and a compact table comparing CART with transcript-only search at all four measured word error rates. Removed the limitations section and detailed experiment walkthrough. Linked the full report. |
| `docs/style.css` | Introduced a shared light palette, serif headlines, thin rules, and responsive layouts. Removed automatic and manual dark mode. |
| `docs/demo.html` | Reworked the introduction and walkthrough labels. Removed the demo caveats section. Replaced the question chips with a single question picker and kept the transcript, match breakdown, ranking controls, and pronunciation comparison. |
| `docs/demo.css` | Matched the overview’s light design, styled the question picker and status messages, and adjusted grid sizing for narrow screens. |
| `docs/demo.js` | Updated the question picker, made transcript words keyboard-operable, and simplified ranking feedback. Kept per-question outcome markers and the uncertainty toggle. |

The research code, scorer, recognizer dataset, experiment outputs, and Pages
configuration were outside the redesign scope. The displayed MRR values come
from `results/REPORT.md`; gains are the differences between the displayed
CART and transcript-only values.

## Verification

- Checked the four results rows against the full report and recalculated each gain.
- Ran the original and updated demo logic with the same 825 passages and 18
  questions. The opening example stayed at rank 4 with uncertainty scaling on
  and rank 13 with it off.
- Exercised question selection, transcript inspection, the ranking reset and
  slider controls, and the cell/sail and cell/table pronunciation comparisons.

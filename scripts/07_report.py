"""Render results/*.json into a markdown report with tables and figures."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cart.config import RESULTS

SYS_LABEL = {
    "oracle_bm25": "BM25 / reference transcript (upper bound)",
    "bm25_1best": "BM25 / Whisper 1-best (baseline 1)",
    "nbest_uniform": "Uniform N-best fusion (baseline 2)",
    "dense_1best": "Dense retrieval / 1-best (baseline 3)",
    "cart": "CART (ours)",
}
ORDER = ["oracle_bm25", "bm25_1best", "nbest_uniform", "dense_1best", "cart"]
COND_LABEL = {"clean": "clean", "snr5": "babble 5 dB", "snr0": "babble 0 dB", "snr-5": "babble -5 dB"}


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def fmt(v, p=None, ref=None):
    """Format a metric, marking significance *with its direction*.

    The paired bootstrap is two-sided: a small p says "differs from the
    reference", not "beats it". Marking both directions with the same star made
    a system that is significantly *worse* than the baseline (dense_1best, in
    every condition) read like a win. So a trailing star = significantly
    better than the reference, a trailing v = significantly worse.
    """
    if v is None:
        return "-"
    s = f"{v:.4f}"
    if p is not None and p < 0.05:
        if ref is None or v > ref:
            s += "*"
        elif v < ref:
            s += "v"
    return s


def main():
    R = json.loads((RESULTS / "retrieval.json").read_text())
    conds = list(R["conditions"])
    L = ["# CART: Confidence-Aware Retrieval for Audio RAG - experimental results",
         "",
         f"Recognizer: `faster-whisper {R['model']}`, beam search with 8 beams. "
         f"CART configuration (tuned on the dev split): "
         f"tau={R['config']['tau']}, lambda={R['config']['lam']}, "
         f"alpha={R['config']['alpha']}, theta={R['config']['theta']}.",
         ""]

    # ---- transcription quality ------------------------------------------
    L += ["## 1. Transcription quality per condition", ""]
    rows = []
    for c in conds:
        a = R["conditions"][c]["asr"]
        rows.append([COND_LABEL.get(c, c), f"{a['wer']*100:.1f}", f"{a['ne_wer']*100:.1f}",
                     f"{a['non_ne_wer']*100:.1f}", f"{a['pct_words_lowconf']*100:.1f}",
                     f"{a['conf_auc']:.3f}", f"{a['alt_coverage_lowconf']*100:.1f}"])
    L += [md_table(["condition", "WER %", "NE-WER %", "non-NE WER %",
                    "words below tau %", "conf AUC", "low-conf words with an alternative %"], rows), ""]

    # ---- main retrieval table -------------------------------------------
    L += ["## 2. Retrieval effectiveness (held-out test queries)", "",
          "`*` = significantly better than BM25 / 1-best, `v` = significantly worse "
          "(two-sided paired bootstrap on reciprocal rank, 5000 resamples, p < 0.05; "
          "p values below 1/5000 = 2e-4 cannot be resolved by this test).", ""]
    for c in conds:
        d = R["conditions"][c]
        L += [f"### {COND_LABEL.get(c, c)}  (WER {d['asr']['wer']*100:.1f}%, "
              f"n={d['n_queries']} queries over {d['n_docs']} passages)", ""]
        rows = []
        base_mrr = (d["systems"].get("bm25_1best") or {}).get("mrr")
        for s in ORDER:
            m = d["systems"].get(s)
            if not m:
                continue
            p = m.get("p_vs_bm25_1best")
            rows.append([SYS_LABEL[s], fmt(m["recall@1"]), fmt(m["recall@5"]),
                         fmt(m["recall@10"]), fmt(m["ndcg@10"]), fmt(m["mrr"], p, base_mrr),
                         f"[{m['mrr_ci'][0]:.3f}, {m['mrr_ci'][1]:.3f}]",
                         f"{m['ms_per_query']:.2f}"])
        L += [md_table(["system", "R@1", "R@5", "R@10", "nDCG@10", "MRR",
                        "MRR 95% CI", "ms/query"], rows), ""]

    # ---- named-entity queries -------------------------------------------
    L += ["## 3. Queries that mention a named entity", "",
          "ASR errors concentrate on names, so this is where CART is supposed to pay off.", ""]
    rows = []
    for c in conds:
        d = R["conditions"][c]
        ne, non = d["ne_queries"], d["non_ne_queries"]
        rows.append([COND_LABEL.get(c, c), ne["n"],
                     fmt(ne["mrr"]["bm25_1best"]["mrr"]), fmt(ne["mrr"]["cart"]["mrr"]),
                     non["n"], fmt(non["mrr"]["bm25_1best"]["mrr"]),
                     fmt(non["mrr"]["cart"]["mrr"])])
    L += [md_table(["condition", "n (NE queries)", "BM25 MRR", "CART MRR",
                    "n (other)", "BM25 MRR", "CART MRR"], rows), ""]

    # ---- ablations -------------------------------------------------------
    abl_path = RESULTS / "ablations.json"
    if abl_path.exists():
        A = json.loads(abl_path.read_text())
        L += ["## 4. Ablations", "",
              "`*` = significantly better than full CART, `v` = significantly worse "
              "(two-sided paired bootstrap, p < 0.05). A variant marked `v` is one "
              "whose removed component was doing real work.", ""]
        for c, d in A["conditions"].items():
            L += [f"### {COND_LABEL.get(c, c)}", ""]
            base_abl = (d["variants"].get("cart") or {}).get("mrr")
            rows = [[k, fmt(v["recall@1"]), fmt(v["mrr"], v.get("p_vs_cart"), base_abl),
                     f"{v['ms_per_query']:.2f}"] for k, v in d["variants"].items()]
            L += [md_table(["variant", "R@1", "MRR", "ms/query"], rows), ""]

    # ---- QA --------------------------------------------------------------
    qa_path = RESULTS / "qa.json"
    if qa_path.exists():
        Q = json.loads(qa_path.read_text())
        L += ["## 5. Downstream question answering", "",
              f"Reader: `{Q['reader']}`, top-{Q['top_k']} retrieved segments as context. "
              "`*` = significantly better than BM25 / 1-best on token-F1, "
              "`v` = significantly worse (two-sided paired bootstrap, p < 0.05).", ""]
        for c, d in Q["conditions"].items():
            L += [f"### {COND_LABEL.get(c, c)}  (n={d['n_queries']})", ""]
            rows = []
            base_f1 = (d["systems"].get("bm25_1best") or {}).get("f1")
            for s, m in d["systems"].items():
                rows.append([SYS_LABEL.get(s, s), fmt(m["em"]),
                             fmt(m["f1"], m.get("p_f1_vs_bm25_1best"), base_f1),
                             f"[{m['f1_ci'][0]:.3f}, {m['f1_ci'][1]:.3f}]",
                             fmt(m.get("llm_judge"))])
            L += [md_table(["system", "EM", "token-F1", "F1 95% CI", "LLM judge"], rows), ""]

    # ---- index cost ------------------------------------------------------
    L += ["## 6. Index cost", ""]
    rows = []
    for c in conds:
        ic = R["conditions"][c]["index_cost"]
        for s in ["bm25_1best", "nbest_uniform", "cart"]:
            if s in ic:
                rows.append([COND_LABEL.get(c, c), SYS_LABEL[s], ic[s]["vocab"],
                             ic[s]["postings"], ic[s]["phonetic_terms"]])
    L += [md_table(["condition", "system", "vocabulary", "postings", "phonetic terms"], rows), ""]

    figs = make_figures(R, abl_path)
    L += ["## 7. Figures", ""] + [f"![{f}]({f})" for f in figs] + [""]
    (RESULTS / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote results/REPORT.md")


def make_figures(R, abl_path):
    conds = list(R["conditions"])
    wers = [R["conditions"][c]["asr"]["wer"] * 100 for c in conds]
    figs = []

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for s in ORDER:
        if s not in R["conditions"][conds[0]]["systems"]:
            continue
        y = [R["conditions"][c]["systems"][s]["mrr"] for c in conds]
        ax[0].plot(wers, y, marker="o", label=SYS_LABEL[s])
        y1 = [R["conditions"][c]["systems"][s]["recall@1"] for c in conds]
        ax[1].plot(wers, y1, marker="o", label=SYS_LABEL[s])
    for i, t in enumerate(["MRR", "Recall@1"]):
        ax[i].set_xlabel("word error rate (%)")
        ax[i].set_ylabel(t)
        ax[i].grid(alpha=.3)
    ax[0].legend(fontsize=7)
    fig.suptitle("Retrieval quality vs. transcription quality")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_retrieval_vs_wer.png", dpi=150)
    figs.append("fig_retrieval_vs_wer.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    d = [R["conditions"][c]["asr"] for c in conds]
    x = np.arange(len(conds))
    ax.bar(x - .2, [a["ne_wer"] * 100 for a in d], .4, label="named entities")
    ax.bar(x + .2, [a["non_ne_wer"] * 100 for a in d], .4, label="other words")
    ax.set_xticks(x); ax.set_xticklabels([COND_LABEL.get(c, c) for c in conds])
    ax.set_ylabel("WER (%)"); ax.legend(); ax.grid(alpha=.3, axis="y")
    ax.set_title("ASR errors concentrate on named entities")
    fig.tight_layout(); fig.savefig(RESULTS / "fig_ne_wer.png", dpi=150)
    figs.append("fig_ne_wer.png")

    if abl_path.exists():
        A = json.loads(abl_path.read_text())
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for c, dd in A["conditions"].items():
            lam = dd["lambda_sweep"]
            ax[0].plot([float(k) for k in lam], list(lam.values()), marker="o",
                       label=COND_LABEL.get(c, c))
            tau = dd["tau_sweep"]
            ax[1].plot([float(k) for k in tau], list(tau.values()), marker="o",
                       label=COND_LABEL.get(c, c))
        ax[0].set_xlabel("lambda (phonetic weight)"); ax[0].set_xscale("symlog", linthresh=.25)
        ax[1].set_xlabel("tau (confidence threshold)")
        for a in ax:
            a.set_ylabel("MRR"); a.grid(alpha=.3); a.legend(fontsize=8)
        fig.suptitle("Sensitivity to CART's hyperparameters")
        fig.tight_layout(); fig.savefig(RESULTS / "fig_sensitivity.png", dpi=150)
        figs.append("fig_sensitivity.png")
    return figs


if __name__ == "__main__":
    main()

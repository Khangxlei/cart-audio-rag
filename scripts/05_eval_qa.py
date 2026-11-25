"""Downstream QA: does better retrieval (and alternative-annotated context)
translate into better answers?

For each retrieval system we take its top-k segments, hand them to a local
reader LLM, and score EM / token-F1 / LLM-judge against the SQuAD answers.
`cart+altctx` additionally annotates low-confidence words with their beam
alternatives, which is the proposal's Alternative-Context Generation step.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cart.config import CARTConfig, DATA, DEFAULT, RESULTS
from cart.data import read_jsonl
from cart.evaluate import build_systems, filter_queries, load_condition
from cart.metrics import bootstrap_ci, exact_match, paired_bootstrap_p, qa_metrics, token_f1
from cart.phonetics import G2P
from cart.qa import Reader, build_prompt, judge_prompts, parse_verdicts, render_context

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--conditions", nargs="+", default=["clean", "snr0", "snr-5"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-queries", type=int, default=600)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--config", default=str(RESULTS / "tuning.json"))
    a = ap.parse_args()

    cfg = CARTConfig()
    if Path(a.config).exists():
        b = json.loads(Path(a.config).read_text())["pooled_best"]
        cfg = CARTConfig(tau=b["tau"], alpha=b["alpha"], theta=b["theta"], lam=b["lam"])

    g2p = G2P()
    reader = Reader(DEFAULT.reader_model)
    all_q = read_jsonl(DATA / "queries.jsonl")
    # Resumable: a condition runs 5 systems x max_queries generations plus the
    # same number of judge calls, and the stage used to write its artifact only
    # after the last condition, so a walltime cut discarded all of it. Reuse
    # whatever is already on disk, provided it came from the same reader and
    # top_k, and skip those conditions.
    qa_path = RESULTS / "qa.json"
    out = {"reader": DEFAULT.reader_model, "top_k": a.top_k, "conditions": {}}
    if qa_path.exists():
        try:
            prev = json.loads(qa_path.read_text())
            if prev.get("reader") == out["reader"] and prev.get("top_k") == out["top_k"]:
                out = prev
                print("resuming, already done:", sorted(out["conditions"]), flush=True)
            else:
                print("reader or top_k changed, recomputing all conditions", flush=True)
        except Exception as e:
            print("could not read existing qa.json, starting over:", e, flush=True)

    for cond in a.conditions:
        if cond in out["conditions"]:
            print(f"[{cond}] already complete, skipping", flush=True)
            continue
        corpus, asr = load_condition(a.model, cond)
        qs = filter_queries(all_q, corpus, split=a.split)
        if not qs:
            continue
        rng = np.random.default_rng(7)
        if a.max_queries and len(qs) > a.max_queries:
            qs = [qs[i] for i in sorted(rng.choice(len(qs), a.max_queries, replace=False))]
        golds = [q["answers"] for q in qs]
        systems = build_systems(corpus, asr, cfg, g2p, include_dense=False)

        runs = {}
        for name, ret in systems.items():
            style = "plain"
            ctxs = []
            for q in qs:
                top = [d for d, _ in ret.search(q["question"], a.top_k)]
                if name == "oracle_bm25":
                    txt = {c["doc_id"]: c["gold_text"] for c in corpus}
                    ctxs.append([txt[d] for d in top])
                else:
                    ctxs.append([render_context(asr[d], style, cfg.tau, cfg.max_alts)
                                 for d in top])
            runs[name] = ctxs
        # Alternative-Context Generation on top of CART's own ranking
        ret = systems["cart"]
        runs["cart+altctx"] = [
            [render_context(asr[d], "alt", cfg.tau, cfg.max_alts)
             for d, _ in ret.search(q["question"], a.top_k)] for q in qs]

        res, per_q = {}, {}
        for name, ctxs in runs.items():
            preds = reader.generate([build_prompt(q["question"], c) for q, c in zip(qs, ctxs)])
            m = qa_metrics(preds, golds)
            f1s = np.array([token_f1(p, g) for p, g in zip(preds, golds)])
            ems = np.array([exact_match(p, g) for p, g in zip(preds, golds)])
            per_q[name] = (ems, f1s)
            mean, lo, hi = bootstrap_ci(f1s)
            m["f1_ci"] = [lo, hi]
            if a.judge:
                raw = reader.generate(judge_prompts([q["question"] for q in qs], preds, golds),
                                      batch_size=16)
                m["llm_judge"] = round(float(np.mean(parse_verdicts(raw))), 4)
            res[name] = m
            print(f"[{cond}] {name}: {m}", flush=True)
        for name in res:
            if name != "bm25_1best":
                res[name]["p_f1_vs_bm25_1best"] = paired_bootstrap_p(
                    per_q[name][1], per_q["bm25_1best"][1])
        out["conditions"][cond] = {"n_queries": len(qs), "systems": res}
        # checkpoint after every condition, not once at the very end
        qa_path.write_text(json.dumps(out, indent=2))
        g2p.save()
        print(f"[{cond}] checkpointed", flush=True)

    g2p.save()
    qa_path.write_text(json.dumps(out, indent=2))
    missing = [c for c in a.conditions if c not in out["conditions"]]
    print("QA COMPLETE" if not missing else f"QA INCOMPLETE, missing {missing}")

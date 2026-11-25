"""Component ablations and sensitivity curves for CART.

  cart                full method
  -phonetic           alternative-spelling expansion only
  -alt_index          phonetic fallback only
  -conf_gate          phonetic channel applied without the (1 - c) factor
  -conf_targeting     tau = 1.0, i.e. expand *every* word rather than only the
                      spans the recognizer was unsure about

Plus lambda and tau sweeps on the test split, to show how sharp the optimum is.
"""
import argparse, json, sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from cart.config import CARTConfig, DATA, RESULTS
from cart.data import read_jsonl
from cart.evaluate import (evaluate_systems, filter_queries, load_condition,
                           per_query_rr, run_queries)
from cart.metrics import paired_bootstrap_p, retrieval_metrics
from cart.phonetics import G2P
from cart.retrieval import make_retriever

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--conditions", nargs="+", default=["clean", "snr0", "snr-5"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--config", default=str(RESULTS / "tuning.json"))
    a = ap.parse_args()

    cfg = CARTConfig()
    if Path(a.config).exists():
        b = json.loads(Path(a.config).read_text())["pooled_best"]
        cfg = CARTConfig(tau=b["tau"], alpha=b["alpha"], theta=b["theta"], lam=b["lam"])

    variants = {
        "cart": cfg,
        "-phonetic": replace(cfg, use_phonetic=False),
        "-alt_index": replace(cfg, use_alt_index=False),
        "-conf_gate": replace(cfg, confidence_gate=False),
        "-conf_targeting": replace(cfg, tau=1.0),
    }

    g2p = G2P()
    all_q = read_jsonl(DATA / "queries.jsonl")
    # Resumable: each condition costs ~30 min (13 index builds + 21 sweeps over
    # 4021 test queries), and the stage used to write its artifact only after
    # the final condition, so a walltime cut threw away every completed
    # condition. Load what is already there and skip those.
    abl_path = RESULTS / "ablations.json"
    out = {"config": cfg.__dict__, "conditions": {}}
    if abl_path.exists():
        try:
            prev = json.loads(abl_path.read_text())
            if prev.get("config") == cfg.__dict__:
                out = prev
                print("resuming, already done:", sorted(out["conditions"]), flush=True)
            else:
                print("config changed since last run, recomputing all conditions", flush=True)
        except Exception as e:
            print("could not read existing ablations.json, starting over:", e, flush=True)

    for cond in a.conditions:
        if cond in out["conditions"]:
            print(f"[{cond}] already complete, skipping", flush=True)
            continue
        corpus, asr = load_condition(a.model, cond)
        qs = filter_queries(all_q, corpus, split=a.split)
        if not qs:
            continue
        golds = [q["gold_doc_id"] for q in qs]
        res, rr = {}, {}
        for name, c in variants.items():
            r = make_retriever(name, corpus, asr, "cart", c, g2p)
            ranked, ms = run_queries(r, qs)
            res[name] = retrieval_metrics(ranked, golds)
            res[name]["ms_per_query"] = round(ms, 2)
            rr[name] = per_query_rr(ranked, qs)
        for name in res:
            if name != "cart":
                res[name]["p_vs_cart"] = paired_bootstrap_p(rr[name], rr["cart"])

        # --- sensitivity sweeps -------------------------------------------
        lam_curve = {}
        base = make_retriever("s", corpus, asr, "cart", cfg, g2p)
        for lam in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
            base.cfg = replace(cfg, lam=lam)
            lam_curve[lam] = retrieval_metrics(run_queries(base, qs)[0], golds)["mrr"]
        tau_curve = {}
        for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            r = make_retriever("s", corpus, asr, "cart", replace(cfg, tau=tau), g2p)
            tau_curve[tau] = retrieval_metrics(run_queries(r, qs)[0], golds)["mrr"]

        out["conditions"][cond] = {"variants": res, "lambda_sweep": lam_curve,
                                   "tau_sweep": tau_curve}
        print(f"[{cond}] " + "  ".join(f"{k}={v['mrr']:.4f}" for k, v in res.items()), flush=True)
        # checkpoint after every condition, not once at the very end
        abl_path.write_text(json.dumps(out, indent=2))
        g2p.save()

    g2p.save()
    abl_path.write_text(json.dumps(out, indent=2))
    missing = [c for c in a.conditions if c not in out["conditions"]]
    print("ABLATIONS COMPLETE" if not missing else f"ABLATIONS INCOMPLETE, missing {missing}")
    print("wrote results/ablations.json")

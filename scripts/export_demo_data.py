"""Export a slice of the experiment for the static demo page.

The project page's demo (docs/demo.html) re-implements CART's scoring in
JavaScript so it can run on GitHub Pages with no backend. For that to be honest
it has to run on genuine recognizer output, so this writes out real transcripts,
real per-word confidences and the real beam alternatives, plus the phoneme
strings the phonetic channel needs -- grapheme-to-phoneme conversion needs a
model that cannot run in a static page.

Run it where the transcripts are (the cluster), then copy the result:

    python scripts/export_demo_data.py --conditions clean snr5 snr0 snr-5
    scp <host>:/path/cart/docs/demo_data.json docs/demo_data.json

The page reads the `conditions` array and offers a switcher when more than one
is present, so adding the noisy conditions needs no change to the page.

Output is packed: each passage stores its words as one string, confidences as
integer percents, and alternatives keyed by word index. That is several times
smaller than the naive form, which matters for a file served to every visitor.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cart.asr import tokenize
from cart.config import DATA, RESULTS
from cart.data import read_jsonl
from cart.evaluate import filter_queries, load_condition
from cart.index import STOPWORDS, content_terms, indexable
from cart.phonetics import G2P

_STRESS = re.compile(r"\d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small.en")
    ap.add_argument("--conditions", nargs="+",
                    default=["clean", "snr5", "snr0", "snr-5"])
    ap.add_argument("--n-queries", type=int, default=18)
    ap.add_argument("--max-passages", type=int, default=825,
                    help="cap the collection so the payload stays servable")
    ap.add_argument("--max-alts", type=int, default=2)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", default="docs/demo_data.json")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    corpus_all = read_jsonl(DATA / "corpus.jsonl")
    queries_all = read_jsonl(DATA / "queries.jsonl")

    per_cond = {}
    for c in a.conditions:
        try:
            per_cond[c] = load_condition(a.model, c)[1]
        except FileNotFoundError:
            print("no transcripts for condition %r, skipping" % c)
    conds = [c for c in a.conditions if c in per_cond]
    if not conds:
        sys.exit("no transcripts found for any requested condition")

    # only passages transcribed under *every* kept condition, so switching
    # condition in the page never changes the collection underneath
    docs_all = [c for c in corpus_all
                if all(c["doc_id"] in per_cond[k] for k in conds)]
    by_id = {c["doc_id"]: c for c in docs_all}
    print("conditions: %s | passages with all of them: %d"
          % (", ".join(conds), len(docs_all)))

    # The hardest condition is the interesting one for picking examples: a
    # question is a good demonstration when a content term it shares with its
    # answer passage is MISSING from that passage's 1-best. Surface matching has
    # failed, which is the case CART exists for. Selected by that rule, never by
    # whether CART happens to win.
    probe = conds[-1]

    def missing_terms(q):
        g = q["gold_doc_id"]
        one_best = set(content_terms(tokenize(per_cond[probe][g].text)))
        gold = set(content_terms(tokenize(by_id[g]["gold_text"])))
        qt = set(content_terms(tokenize(q["question"])))
        return sorted((qt & gold) - one_best)

    cands = []
    for q in filter_queries(queries_all, docs_all, split=a.split):
        miss = missing_terms(q)
        if miss:
            cands.append((len(miss), q, miss))
    cands.sort(key=lambda t: (-t[0], t[1]["qid"]))
    pool = cands[: a.n_queries * 5]
    rng.shuffle(pool)
    picked = pool[: a.n_queries]
    gold_ids = {q["gold_doc_id"] for _, q, _ in picked}
    print("demonstration questions: %d (probe condition %r)" % (len(picked), probe))

    ids = [c["doc_id"] for c in docs_all if c["doc_id"] not in gold_ids]
    rng.shuffle(ids)
    keep = list(gold_ids) + ids[: max(0, a.max_passages - len(gold_ids))]
    keep_set = set(keep)

    vocab, docs = set(), []
    for c in docs_all:
        did = c["doc_id"]
        if did not in keep_set:
            continue
        entry = {"id": did, "ti": c.get("title", "")}
        if did in gold_ids:
            entry["g"] = c["gold_text"]
            vocab |= set(content_terms(tokenize(c["gold_text"])))
        packed = {}
        for k in conds:
            r = per_cond[k][did]
            words = [w.word for w in r.words]
            probs = [int(round(float(w.prob) * 100)) for w in r.words]
            alts = {}
            for i, w in enumerate(r.words):
                aa = [x.lower() for x in w.alts
                      if x and indexable(x.lower())][: a.max_alts]
                if aa:
                    alts[str(i)] = aa
                    vocab |= set(aa)
            vocab |= {t for t in words if indexable(t)}
            block = {"t": " ".join(words), "p": probs,
                     "c": round(float(r.mean_conf), 3)}
            if alts:
                block["a"] = alts
            packed[k] = block
        entry["cond"] = packed
        docs.append(entry)

    qs = []
    for _, q, miss in picked:
        qs.append({"qid": q["qid"], "q": q["question"], "gold": q["gold_doc_id"],
                   "answers": q["answers"][:2], "missing": miss})
        vocab |= set(content_terms(tokenize(q["question"])))

    g2p = G2P()
    phones = {}
    for t in sorted(vocab):
        if indexable(t):
            p = g2p(t)
            if p:
                phones[t] = [_STRESS.sub("", x) for x in p]
    g2p.save()

    cfg = {"tau": 0.8, "lam": 2.0, "alpha": 0.5, "theta": 0.55, "max_alts": 3}
    tuning = RESULTS / "tuning.json"
    if tuning.exists():
        best = json.loads(tuning.read_text())["pooled_best"]
        cfg.update({k: best[k] for k in ("tau", "lam", "alpha", "theta")})

    wer = {}
    retr = RESULTS / "retrieval.json"
    if retr.exists():
        R = json.loads(retr.read_text())
        for k in conds:
            if k in R["conditions"]:
                wer[k] = round(R["conditions"][k]["asr"]["wer"], 4)

    payload = {
        "provenance": (
            "Real faster-whisper %s output (beam search, 8 beams) for %d spoken "
            "passages of Spoken-SQuAD, condition(s): %s. Per-word confidences and "
            "beam alternatives are exactly what the recognizer produced. The "
            "reported experiment uses 1896 passages, so IDF differs here and "
            "rankings on this page will not match the reported tables."
            % (a.model, len(docs), ", ".join(conds))),
        "config": cfg,
        "bm25": {"k1": 0.9, "b": 0.4},
        "conditions": conds,
        "wer": wer,
        "stopwords": sorted(STOPWORDS),
        "packed": True,
        "multi": True,
        "docs": docs,
        "queries": qs,
        "phones": phones,
    }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print("wrote %s  %.2f MB  | %d passages, %d questions, %d pronunciations"
          % (out, out.stat().st_size / 1e6, len(docs), len(qs), len(phones)))


if __name__ == "__main__":
    main()

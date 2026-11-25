"""Fast pre-flight: exercise every import and code path the Slurm jobs use,
offline, so a trivial error surfaces here instead of after hours of queueing."""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ok = True
def check(name, fn):
    global ok
    try:
        r = fn()
        print(f"  OK   {name}" + (f"  [{r}]" if r else ""))
    except Exception as e:
        ok = False
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)

from cart.config import DEFAULT, DATA, RUNS, RESULTS   # sets HF_HOME

def t_parquet():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from cart.data import HF_REPO, N_SHARDS
    p = hf_hub_download(HF_REPO, f"data/test-00000-of-{N_SHARDS:05d}.parquet", repo_type="dataset")
    f = pq.ParquetFile(p)
    row = f.read_row_group(0).slice(0, 1).to_pylist()[0]
    return f"{f.metadata.num_rows} rows, audio {len(row['context']['bytes'])}B"

def t_audio():
    import io, numpy as np, soundfile as sf
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from cart.data import HF_REPO, N_SHARDS
    from cart import audio as A
    p = hf_hub_download(HF_REPO, f"data/test-00000-of-{N_SHARDS:05d}.parquet", repo_type="dataset")
    row = pq.ParquetFile(p).read_row_group(0).slice(0, 1).to_pylist()[0]
    x, sr = sf.read(io.BytesIO(row["context"]["bytes"]), dtype="float32")
    y = A.degrade(x, "snr-5", seed=1)
    return f"{sr}Hz {len(x)/sr:.1f}s -> noisy rms {float(np.sqrt((y**2).mean())):.3f}"

def t_squad():
    from cart.data import load_squad_dev
    return f"{len(load_squad_dev())} questions"

def t_g2p():
    """Assert real phoneme output, not just absence of an exception. A broken
    NLTK install once made every word return no phonemes, which scores
    similarity 1.00 for every pair and silently disables the phonetic channel."""
    from cart.phonetics import G2P, phonetic_similarity
    g = G2P()
    cell, sail, table = g("cell"), g("sail"), g("table")
    assert cell == ("S", "EH", "L"), f"g2p('cell') = {cell}, expected S EH L"
    assert sail and table, "g2p returned no phonemes - NLTK data missing?"
    near, far = phonetic_similarity(cell, sail), phonetic_similarity(cell, table)
    assert 0.6 < near < 0.99, f"cell/sail similarity {near:.2f} out of range"
    assert far < near, f"cell/table ({far:.2f}) should be below cell/sail ({near:.2f})"
    return f"cell/sail {near:.2f}, cell/table {far:.2f}"


def t_g2p_pathological():
    """Whisper emits runs of hundreds of digits on noisy audio; g2p_en hands
    those to inflect, which raises NumOutOfRangeError past ~10^303 and killed a
    whole eval run. Must degrade to 'no pronunciation', never raise."""
    from cart.phonetics import G2P, phonetic_similarity
    from cart.index import content_terms, indexable
    g = G2P()
    bad = ["0" * 334, "1" + "0" * 90, "9" * 400, "", "x" * 120]
    for b in bad:
        phones = g(b)
        assert isinstance(phones, tuple)
        phonetic_similarity(phones, g("cell"))
    # ordinary words must still pronounce after the pathological ones
    assert g("lorenz"), "ordinary word lost its pronunciation"
    assert not indexable("0" * 334), "long junk token must not be indexable"
    assert content_terms(["cell", "0" * 334]) == ["cell"]
    return f"{len(bad)} pathological tokens handled"

def t_whisper():
    import io, numpy as np, soundfile as sf
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from cart.asr import NBestWhisper
    from cart.config import ASRConfig
    from cart.data import HF_REPO, N_SHARDS
    p = hf_hub_download(HF_REPO, f"data/test-00000-of-{N_SHARDS:05d}.parquet", repo_type="dataset")
    row = pq.ParquetFile(p).read_row_group(0).slice(0, 1).to_pylist()[0]
    x, _ = sf.read(io.BytesIO(row["context"]["bytes"]), dtype="float32")
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ct = "int8_float16" if dev == "cuda" else "int8"
    m = NBestWhisper(ASRConfig(model="small.en", device=dev, compute_type=ct,
                               beam_size=8, num_hypotheses=8))
    r = m.transcribe(x[:16000 * 20], "probe", "clean")
    n_alt = sum(1 for w in r.words if w.alts)
    return f"{dev}: {len(r.words)} words, {len(r.nbest)} alt-hyps, {n_alt} words with alts"

def t_retrieval():
    from cart.config import CARTConfig
    from cart.retrieval import make_retriever
    from cart.asr import ASRResult, ASRWord
    corpus = [{"doc_id": f"d{i}", "gold_text": t} for i, t in enumerate(
        ["the cell membrane of the organism", "a sail on the old wooden ship",
         "lorenz attractor chaotic dynamics"])]
    asr = {c["doc_id"]: ASRResult(c["doc_id"], "clean", c["gold_text"],
            [ASRWord(w, 0, 1, 0.5 if i == 1 else 0.99, ["sail"] if i == 1 else [])
             for i, w in enumerate(c["gold_text"].split())], ["alt hyp"], 0.9)
           for c in corpus}
    r = make_retriever("cart", corpus, asr, "cart", CARTConfig(), None)
    return f"top={r.search('what is a cell', 2)[0][0]}"

def t_reader():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(DEFAULT.reader_model)
    return f"reader tokenizer vocab {len(tok)}"

def t_dense():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(DEFAULT.dense_model, device="cpu")
    return f"dim {m.get_sentence_embedding_dimension()}"

def t_misc():
    import jiwer, matplotlib, rank_bm25
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return f"jiwer wer {jiwer.wer('a b c', 'a b d'):.2f}"

for n, f in [("parquet", t_parquet), ("audio+noise", t_audio), ("squad", t_squad),
             ("g2p", t_g2p), ("g2p pathological", t_g2p_pathological),
             ("retrieval", t_retrieval), ("misc", t_misc),
             ("dense model", t_dense), ("reader tokenizer", t_reader),
             ("whisper n-best", t_whisper)]:
    check(n, f)

print("VALIDATION", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)

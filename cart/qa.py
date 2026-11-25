"""Downstream reader: answer generation over retrieved audio segments.

Two context formats are compared, which is the proposal's
"Alternative-Context Generation" component:

* ``plain``  -- the 1-best transcript of each retrieved segment.
* ``alt``    -- the same transcript, but every low-confidence word is annotated
                inline with the spellings the other beams proposed, e.g.
                ``cell [or sail, shell]``, so the reader can recover from a
                mistranscription instead of being misled by it.
"""
from __future__ import annotations

import re

import torch

ANSWER_RE = re.compile(r"(?i)\banswer\s*:\s*")

SYSTEM = (
    "You answer questions about automatic transcripts of spoken recordings. "
    "The transcripts contain recognition errors. Answer with the shortest exact "
    "span from the transcript that answers the question - a few words at most. "
    "Do not explain.\n"
    "A token written as {a|b|c} means the recognizer was unsure and these are "
    "the competing spellings. Choose the single spelling that is most likely "
    "correct and write only that word. Never copy the braces, the bars, or more "
    "than one alternative. Example: transcript `the {filo|philo} of byzantium` "
    "-> answer `Philo of Byzantium`."
)

# `a [or b, c]` / `{a|b|c}` left in an answer is a formatting failure; fall back
# to the 1-best spelling so the alternative-context condition is never penalised
# relative to plain context by formatting alone.
_ALT_BRACE = re.compile(r"\{([^{}|]+)(?:\|[^{}]*)?\}")
_ALT_BRACK = re.compile(r"\s*\[\s*or[^\]]*\]")


def render_context(asr_result, style: str = "plain", tau: float = 0.6, max_alts: int = 3) -> str:
    if style == "plain":
        return asr_result.text
    parts = []
    for w in asr_result.words:
        if w.prob < tau and w.alts:
            parts.append("{" + "|".join([w.word, *w.alts[:max_alts]]) + "}")
        else:
            parts.append(w.word)
    return " ".join(parts)


def build_prompt(question: str, contexts: list[str]) -> list[dict]:
    blocks = "\n\n".join(f"[Segment {i+1}]\n{c}" for i, c in enumerate(contexts))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{blocks}\n\nQuestion: {question}\nAnswer:"},
    ]


class Reader:
    def __init__(self, model_name: str, device: str = "cuda", max_new_tokens: int = 24):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_name)
        # Qwen2 overflows in fp16 (NaN logits -> "!!!!" output); bf16 is stable
        # on Ampere and newer, with fp32 as the fallback.
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

    @torch.inference_mode()
    def generate(self, chats: list[list[dict]], batch_size: int = 8) -> list[str]:
        outs = []
        for i in range(0, len(chats), batch_size):
            batch = chats[i: i + batch_size]
            texts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                     for c in batch]
            enc = self.tok(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=3072).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens,
                                      do_sample=False, temperature=None, top_p=None,
                                      pad_token_id=self.tok.pad_token_id)
            for j in range(len(batch)):
                new = gen[j][enc["input_ids"].shape[1]:]
                outs.append(clean_answer(self.tok.decode(new, skip_special_tokens=True)))
        return outs


def clean_answer(s: str) -> str:
    s = s.strip()
    s = ANSWER_RE.sub("", s)
    s = s.split("\n")[0].strip()
    s = _ALT_BRACE.sub(r"\1", s)
    s = _ALT_BRACK.sub("", s)
    s = s.strip(' "\'`.')
    return " ".join(s.split())


JUDGE_SYSTEM = (
    "You grade short answers. Reply with exactly one word: CORRECT if the "
    "predicted answer means the same as any reference answer, otherwise WRONG."
)


def judge_prompts(questions, preds, golds):
    out = []
    for q, p, g in zip(questions, preds, golds):
        refs = "; ".join(g)
        out.append([
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content":
                f"Question: {q}\nReference answer(s): {refs}\nPredicted answer: {p}\nVerdict:"},
        ])
    return out


def parse_verdicts(raw):
    return [1.0 if "correct" in r.lower() and "incorrect" not in r.lower() else 0.0 for r in raw]

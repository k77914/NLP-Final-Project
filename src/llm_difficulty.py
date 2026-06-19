from __future__ import annotations
import re
import json
import numpy as np
import pandas as pd

# Gold-free difficulty features from a general reasoning LLM: self-consistency over k samples,
# generation stats, and a 1-10 difficulty judgment. Parsing/aggregation helpers are pure and
# unit-tested; compute_llm_features drives vLLM (GPU) and caches the result.

FEATURE_COLS = [
    "sc_agreement", "sc_entropy", "sc_n_distinct",
    "gen_len_mean", "gen_len_std", "refuse_rate",
    "judge_difficulty", "judge_p_solvable",
]

_ANS_RE = re.compile(r"final answer\s*[:\-]?\s*(.+)", re.I)
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_MCQ_RE = re.compile(r"\b([A-E])\b")
_REFUSE_RE = re.compile(r"\b(cannot|can't|unable|not sure|don't know|impossible)\b", re.I)

SOLVE_PROMPT = ("Solve the problem. Show brief reasoning, then end with "
                "'Final answer: <answer>'.\n\nProblem:\n{q}")
JUDGE_PROMPT = ('Rate how hard this question is for a mid-tier language model. Respond with ONLY '
                'a JSON object: {{"difficulty": <integer 1-10>, "p_solvable": <float 0-1>}}.'
                '\n\nQuestion:\n{q}')


def extract_answer(text: str) -> str:
    """Best-effort final-answer extraction: \\boxed{}, 'Final answer:', MCQ letter, or last number."""
    if not text:
        return ""
    m = _BOXED_RE.search(text)
    if m:
        tail = m.group(1)
    else:
        m = _ANS_RE.search(text)
        if m:
            tail = m.group(1)
        else:
            lines = [ln for ln in text.strip().splitlines() if ln.strip()]
            tail = lines[-1] if lines else ""
    tail = tail.strip()
    mcq = _MCQ_RE.findall(tail[:8])
    if mcq:
        return mcq[0].upper()
    nums = _NUM_RE.findall(tail)
    if nums:
        return nums[-1]
    return tail.lower()[:40]


def self_consistency_features(answers, lengths, refusals) -> dict:
    from collections import Counter
    n = max(len(answers), 1)
    c = Counter(a for a in answers if a != "")
    total = sum(c.values())
    top = c.most_common(1)[0][1] if c else 0
    probs = np.array([v / total for v in c.values()]) if total > 0 else np.array([1.0])
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return {
        "sc_agreement": top / n,
        "sc_entropy": entropy,
        "sc_n_distinct": float(len(c)),
        "gen_len_mean": float(np.mean(lengths)) if len(lengths) else 0.0,
        "gen_len_std": float(np.std(lengths)) if len(lengths) else 0.0,
        "refuse_rate": float(np.mean(refusals)) if len(refusals) else 0.0,
    }


def parse_judge(text: str):
    """Parse {'difficulty':1-10,'p_solvable':0-1} from judge output; clamp; default to (5, 0.5)."""
    diff, psolv = 5.0, 0.5
    try:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            j = json.loads(m.group(0))
            diff = float(j.get("difficulty", 5))
            psolv = float(j.get("p_solvable", 0.5))
    except Exception:
        return 5.0, 0.5
    return float(np.clip(diff, 1, 10)), float(np.clip(psolv, 0, 1))


def _trim(q: str, max_chars: int = 6000) -> str:
    q = q or ""
    return q if len(q) <= max_chars else q[:4000] + "\n[...]\n" + q[-1500:]


def _features_frame(sc_lists, judge_texts) -> pd.DataFrame:
    """Build the feature DataFrame from per-query sample lists + judge text (backend-agnostic)."""
    rows = []
    for samples, jt in zip(sc_lists, judge_texts):
        answers = [extract_answer(t) for t in samples]
        lengths = [len(t.split()) for t in samples]
        refusals = [1 if _REFUSE_RE.search(t) else 0 for t in samples]
        feat = self_consistency_features(answers, lengths, refusals)
        d, p = parse_judge(jt)
        feat["judge_difficulty"] = d
        feat["judge_p_solvable"] = p
        rows.append(feat)
    return pd.DataFrame(rows)[FEATURE_COLS].astype(np.float32)


def _compute_vllm(cfg, df):
    from vllm import LLM, SamplingParams
    llm = LLM(model=cfg.llm_id, dtype="bfloat16", gpu_memory_utilization=0.9,
              max_model_len=4096, trust_remote_code=True)
    queries = [_trim(q) for q in df["query"].fillna("").astype(str).tolist()]
    sc = llm.generate([SOLVE_PROMPT.format(q=q) for q in queries],
                      SamplingParams(n=cfg.llm_k_samples, temperature=cfg.llm_temperature,
                                     top_p=0.95, max_tokens=cfg.llm_max_new_tokens))
    jd = llm.generate([JUDGE_PROMPT.format(q=q) for q in queries],
                      SamplingParams(n=1, temperature=0.0, max_tokens=64))
    sc_lists = [[o.text for o in s.outputs] for s in sc]
    judge_texts = [j.outputs[0].text for j in jd]
    return sc_lists, judge_texts


def _hf_generate(model, tok, prompts, n, max_new, do_sample, temperature, bs):
    """Batched left-padded generation; returns a list (per prompt) of n decoded completions."""
    import torch
    out_lists = []
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        msgs = [tok.apply_chat_template([{"role": "user", "content": p}],
                                        tokenize=False, add_generation_prompt=True) for p in chunk]
        enc = tok(msgs, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        kwargs = dict(max_new_tokens=max_new, num_return_sequences=n, pad_token_id=tok.pad_token_id)
        if do_sample:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kwargs.update(do_sample=False)
        with torch.no_grad():
            gen = model.generate(**enc, **kwargs)
        new = gen[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new, skip_special_tokens=True)
        for j in range(len(chunk)):
            out_lists.append(texts[j * n:(j + 1) * n])
        print(f"  hf gen {min(i + bs, len(prompts))}/{len(prompts)}", flush=True)
    return out_lists


def _compute_hf(cfg, df):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.llm_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.llm_id, torch_dtype=torch.bfloat16,
                                                 device_map="auto")
    model.eval()
    queries = [_trim(q) for q in df["query"].fillna("").astype(str).tolist()]
    bs = getattr(cfg, "hf_batch_size", 16)
    sc_lists = _hf_generate(model, tok, [SOLVE_PROMPT.format(q=q) for q in queries],
                            cfg.llm_k_samples, cfg.llm_max_new_tokens, True, cfg.llm_temperature, bs)
    judge_lists = _hf_generate(model, tok, [JUDGE_PROMPT.format(q=q) for q in queries],
                               1, 64, False, 0.0, bs)
    judge_texts = [g[0] if g else "" for g in judge_lists]
    return sc_lists, judge_texts


def compute_llm_features(cfg, df: pd.DataFrame, split: str, backend: str = "vllm") -> pd.DataFrame:
    """Difficulty features per query. backend='vllm' (fast, needs matching CUDA) or 'hf'
    (transformers; rides on Colab's existing torch, no CUDA matching). Cached to parquet."""
    cache = cfg.cache_dir / f"llm_feats_{split}_{len(df)}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    sc_lists, judge_texts = (_compute_hf if backend == "hf" else _compute_vllm)(cfg, df)
    out = _features_frame(sc_lists, judge_texts)
    out.to_parquet(cache)
    return out

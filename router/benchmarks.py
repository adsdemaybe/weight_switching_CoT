"""Unified MCQ loader over two benchmarks, normalized to one row schema.

  * MILU (ai4bharat) — 11 Indic languages. The canonical repo is gated; the
    `murthyrudra/milu-cleaned` mirror carries the same per-language configs
    and identical fields, so it is used by default with the gated original
    tried first (so this transparently upgrades if access is granted).
  * MMMLU (openai) — the non-Indic set: zh, ja, ko, de, fr, es, it, pt, ar.

Both are 4-option multiple choice, so they normalize cleanly onto:
    {question, options[4], answer (letter A-D), lang (code), lang_name,
     subject, source}
"""
import json
import os

from .script_detect import INDIC_LANGS, MMMLU_LANGS

MILU_PRIMARY = "ai4bharat/MILU"
MILU_MIRROR = "murthyrudra/milu-cleaned"
MMMLU_REPO = "openai/MMMLU"

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "milu_stub.jsonl")

# Three disjoint row regions of a single-split benchmark, so nothing that
# tunes or trains is ever scored on itself:
#   validation  [0,            TEST_OFFSET)  — sweep/dev tuning + three-arm dev
#   test        [TEST_OFFSET,  EVAL_RESERVE) — held-out FINAL confirmation only
#   train       [EVAL_RESERVE, ...)          — adapter fine-tuning
# `validation` and `test` limits must each stay <= TEST_OFFSET to not overlap.
EVAL_RESERVE = 100
TEST_OFFSET = 50

# Every language the router can be evaluated on, in one table.
ALL_LANGS = [
    # (dataset config name, short code, benchmark)
    *[(name, code, "milu") for name, code in INDIC_LANGS.items()],
    *[(name, code, "mmmlu") for name, code in MMMLU_LANGS.items()],
]


def _norm_milu(row, lang_name, code):
    """MILU stores the answer as the literal string 'option3'."""
    options = [row["option1"], row["option2"], row["option3"], row["option4"]]
    target = str(row["target"]).strip()
    if target.startswith("option"):
        idx = int(target.replace("option", "")) - 1
    else:
        idx = int(target) - 1
    return {
        "question": row["question"],
        "options": options,
        "answer": "ABCD"[idx],
        "lang": code,
        "lang_name": lang_name,
        "subject": row.get("subject", ""),
        "source": "milu",
    }


def _norm_mmmlu(row, lang_name, code):
    """MMMLU stores options in columns A-D and the answer as a letter."""
    return {
        "question": row["Question"],
        "options": [row["A"], row["B"], row["C"], row["D"]],
        "answer": str(row["Answer"]).strip().upper(),
        "lang": code,
        "lang_name": lang_name,
        "subject": row.get("Subject", ""),
        "source": "mmmlu",
    }


def _load_hf(repo, config, span):
    from datasets import load_dataset
    return load_dataset(repo, config, split=span)


def _milu_span(split, limit):
    # MILU has separate `validation` and `test` splits. Training draws from the
    # `test` split; eval/held-out both come from `validation`, at disjoint
    # offsets so the sweep-tuning rows and the final-confirmation rows differ.
    if split == "train":
        return f"test[:{limit}]"
    if split == "test":
        return f"validation[{TEST_OFFSET}:{TEST_OFFSET + limit}]"
    return f"validation[:{limit}]"          # validation (sweep/dev)


def _mmmlu_span(split, limit):
    # MMMLU ships only `test`; carve three disjoint regions out of it.
    if split == "train":
        return f"test[{EVAL_RESERVE}:{EVAL_RESERVE + limit}]"
    if split == "test":
        return f"test[{TEST_OFFSET}:{TEST_OFFSET + limit}]"
    return f"test[:{limit}]"                 # validation (sweep/dev)


def _load_milu(lang_name, code, split, limit):
    span = _milu_span(split, limit)
    for repo in (MILU_PRIMARY, MILU_MIRROR):
        try:
            ds = _load_hf(repo, lang_name, span)
            return [_norm_milu(r, lang_name, code) for r in ds], repo
        except Exception:
            continue
    return _fixture_rows(lang_name, code, split, limit), "fixture"


def _load_mmmlu(lang_name, code, split, limit):
    from datasets import load_dataset
    span = _mmmlu_span(split, limit)
    try:
        ds = load_dataset(MMMLU_REPO, lang_name, split=span)
        return [_norm_mmmlu(r, lang_name, code) for r in ds], MMMLU_REPO
    except Exception:
        return [], "unavailable"


def _fixture_rows(lang_name, code, split, limit):
    """Offline fallback used only if both MILU sources are unreachable."""
    if not os.path.isfile(FIXTURE_PATH):
        return []
    rows = []
    with open(FIXTURE_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r["language"] == lang_name and r.get("split", "validation") == split:
                rows.append(_norm_milu(r, lang_name, code))
    return rows[:limit]


def load_language(lang_name, code, benchmark, split="validation", limit=10):
    if benchmark == "milu":
        return _load_milu(lang_name, code, split, limit)
    return _load_mmmlu(lang_name, code, split, limit)


def load_eval_set(limit_per_lang=5, split="validation", langs=None):
    """Return (rows, sources) across every configured language."""
    selected = langs or ALL_LANGS
    rows, sources = [], {}
    for lang_name, code, benchmark in selected:
        got, src = load_language(lang_name, code, benchmark, split, limit_per_lang)
        rows.extend(got)
        sources[code] = src
    return rows, sources

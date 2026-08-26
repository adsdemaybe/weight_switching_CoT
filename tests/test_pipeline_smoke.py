"""End-to-end smoke test on a tiny model.

Exercises the real generation loop, the real evaluate() and the real config
plumbing, so integration bugs surface without waiting on the full base model.
Accuracy is meaningless here — only that every arm runs and the accounting
is self-consistent.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate import evaluate
from router.config import RouterConfig

TINY = "sshleifer/tiny-gpt2"


class TinyManager:
    """Same surface as ExpertManager, backed by a tiny model."""

    def __init__(self):
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(TINY)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # tiny-gpt2 has no chat template; supply a minimal one.
        self.tokenizer.chat_template = (
            "{% for m in messages %}{{ m['content'] }}\n{% endfor %}"
        )
        base = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
        cfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                         target_modules=["c_attn"], task_type="CAUSAL_LM")
        self.model = get_peft_model(base, cfg, adapter_name="_base")
        self.loaded = {"_base"}
        for name in ("hi", "zh", "en"):
            self.model.add_adapter(name, self.model.peft_config["_base"])
            self.loaded.add(name)
        self.model.eval()
        self.active = "_base"

    def available_experts(self):
        return sorted(self.loaded - {"_base"})

    def set_active(self, lang):
        target = lang if lang in self.loaded else "_base"
        if target != self.active:
            self.model.set_adapter(target)
            self.active = target
        return self.active


ITEMS = [
    {"question": "भारत की राजधानी क्या है?",
     "options": ["मुंबई", "नई दिल्ली", "कोलकाता", "चेन्नई"],
     "answer": "B", "lang": "hi", "lang_name": "Hindi", "subject": "geo",
     "source": "test"},
    {"question": "中国的首都是什么？",
     "options": ["上海", "北京", "广州", "深圳"],
     "answer": "B", "lang": "zh", "lang_name": "ZH_CN", "subject": "geo",
     "source": "test"},
    {"question": "What is the capital of India?",
     "options": ["Mumbai", "New Delhi", "Kolkata", "Chennai"],
     "answer": "B", "lang": "en", "lang_name": "English", "subject": "geo",
     "source": "test"},
]


@pytest.fixture(scope="module")
def manager():
    return TinyManager()


@pytest.fixture(scope="module")
def cfg():
    return RouterConfig(max_new_tokens=12, check_every=2)


def test_baseline_arm_runs(manager, cfg):
    r = evaluate(manager, ITEMS, route=False, cfg=cfg)
    assert r["n"] == len(ITEMS)
    assert 0.0 <= r["accuracy"] <= 1.0
    # No routing means no switches and therefore no notional prefill saving.
    assert r["avg_switches"] == 0
    assert r["avg_prefill_saved"] == 0


def test_router_arm_runs(manager, cfg):
    r = evaluate(manager, ITEMS, route=True, cfg=cfg)
    assert r["n"] == len(ITEMS)
    assert set(r["per_lang"]) == {"hi", "zh", "en"}


def test_recompute_arm_runs(manager, cfg):
    import dataclasses
    r = evaluate(manager, ITEMS, route=True,
                 cfg=dataclasses.replace(cfg, recompute_on_switch=True))
    assert r["n"] == len(ITEMS)


def test_token_accounting_is_consistent(manager, cfg):
    """Dispatch must never be cheaper than unified, and the saving must equal
    the stated difference."""
    r = evaluate(manager, ITEMS, route=True, cfg=cfg)
    assert r["avg_dispatch_processed"] >= r["avg_total_processed"] - 1e-9
    assert r["avg_prefill_saved"] >= 0
    assert 0.0 <= r["token_saving_ratio"] < 1.0


def test_baseline_and_router_see_identical_items(manager, cfg):
    a = evaluate(manager, ITEMS, route=False, cfg=cfg, per_item=True)
    b = evaluate(manager, ITEMS, route=True, cfg=cfg, per_item=True)
    assert [i["gold"] for i in a["items"]] == [i["gold"] for i in b["items"]]
    assert [i["lang"] for i in a["items"]] == [i["lang"] for i in b["items"]]


def test_max_new_tokens_respected(manager):
    r = evaluate(manager, ITEMS, route=True, cfg=RouterConfig(max_new_tokens=5))
    assert r["avg_gen_tokens"] <= 5


def test_routing_does_not_collapse_generation(manager, cfg):
    """Guard against adapters that destroy instruction-following.

    Over-trained experts (lr=1e-4 for 60 steps) made routed runs emit a bare
    "Answer: C" in 3 tokens while the baseline still produced full reasoning.
    That looks like a huge token saving in the metrics but is just a broken
    model, so a large collapse in generated length is treated as a failure
    rather than a win.
    """
    base = evaluate(manager, ITEMS, route=False, cfg=cfg)
    routed = evaluate(manager, ITEMS, route=True, cfg=cfg)
    assert routed["avg_gen_tokens"] >= 0.4 * base["avg_gen_tokens"], (
        f"routed generation collapsed: {routed['avg_gen_tokens']:.1f} vs "
        f"baseline {base['avg_gen_tokens']:.1f} tokens"
    )


def test_seed_text_selects_initial_expert(manager, cfg):
    """The initial expert must come from the question, not the templated
    prompt — the latter is mostly English scaffolding and misroutes."""
    from router.generation import build_prompt, generate_unified
    q = "中国的首都是什么？"
    prompt = build_prompt(manager.tokenizer, q, ["上海", "北京", "广州", "深圳"])
    r = generate_unified(manager, prompt, cfg=cfg, route=True, seed_text=q)
    assert r["switches"] == [] or r["switches"][0]["from"] == "zh"

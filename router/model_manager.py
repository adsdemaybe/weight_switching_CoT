"""Shared-base + hot-swappable LoRA experts.

The trick that makes "switch model weights mid-generation" compatible with
"maintain a single continuous token history": only the LoRA deltas move, the
frozen base transformer (and therefore every cached key/value already
computed) never changes. Swapping the active adapter is a dict-pointer flip
(`PeftModel.set_adapter`), not a reload, so `past_key_values` computed under
one expert stays valid input to the next.
"""
import os
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (LoraConfig, PeftModel, get_peft_model,
                  get_peft_model_state_dict, set_peft_model_state_dict)

# Overridable so the same code serves the small laptop base and the larger
# GB10 base. Training and generation MUST use the same value or the LoRA
# adapters attach to a different embedding geometry than they were fit on.
BASE_MODEL = os.environ.get("WSC_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "..", "adapters")


def _device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype(device):
    """Half precision on CUDA, fp32 elsewhere.

    bf16 where supported (Ampere+). On Turing (T4, sm_75) bf16 is unavailable
    and falling back to fp32 costs roughly 8x — T4 does ~8 TFLOPS fp32 against
    ~65 TFLOPS fp16 — which turns a 30-minute evaluation into an overnight one.
    So T4 uses fp16.

    MPS stays fp32 deliberately: half precision there has been a source of
    silent numerical drift, and this experiment turns on comparing logits
    across an adapter swap.

    Override with WSC_DTYPE=float32|float16|bfloat16 if a run looks numerically
    suspect.
    """
    override = os.environ.get("WSC_DTYPE")
    if override:
        return getattr(torch, override)
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


class ExpertManager:
    """Owns one base model and a registry of per-language LoRA adapters."""

    def __init__(self, base_model=BASE_MODEL, adapter_dir=ADAPTER_DIR,
                 device=None, dtype=None):
        self.device = device or _device()
        self.dtype = dtype or _dtype(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=self.dtype
        ).to(self.device)
        base.eval()

        # Seed with a throwaway adapter so the model is a PeftModel from the
        # start; every real language adapter gets added alongside it.
        #
        # `embed_tokens` is in the target set deliberately. Adapting only
        # q_proj/v_proj leaves the token embeddings frozen and shared, so an
        # "expert" would have no language-specific representation of tokens at
        # all. LoRA on the embedding gives each language its own embedding
        # delta (~1.2M params at r=8) while keeping the tokenizer and the
        # cache geometry identical — which is what lets the KV cache still be
        # shared across a switch. Swapping in a genuinely different model per
        # language would give better embeddings but different tokenization and
        # different cache shapes, which would forbid cache reuse entirely.
        # We want the language-specific embedding delta to apply on BOTH the
        # read side (embed_tokens) and the output-scoring side (lm_head) — the
        # expert should score tokens in its language differently, not just read
        # them differently. HOW that is achieved depends on the base:
        #   - Small Qwen (0.5B/1.5B) ties embed_tokens to lm_head, so a single
        #     embed_tokens LoRA plus `ensure_weight_tying` already reaches the
        #     output head — adding lm_head would double-target the same tensor.
        #   - 7B+ Qwen does NOT tie (separate lm_head matrix), so
        #     `ensure_weight_tying` finds nothing and warns; the output side
        #     stays un-adapted unless lm_head is targeted explicitly.
        tied = getattr(base.config, "tie_word_embeddings", False)
        targets = ["q_proj", "v_proj", "embed_tokens"]
        if not tied:
            targets.append("lm_head")
        seed_cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0,
            target_modules=targets,
            task_type="CAUSAL_LM",
            ensure_weight_tying=tied,
        )
        self.model = get_peft_model(base, seed_cfg, adapter_name="_base")
        self.model.eval()
        self.loaded = {"_base"}
        self.active = "_base"

        self.seed_cfg = seed_cfg
        self.adapter_dir = adapter_dir
        if os.path.isdir(adapter_dir):
            for lang in sorted(os.listdir(adapter_dir)):
                path = os.path.join(adapter_dir, lang, "lora.safetensors")
                if os.path.isfile(path):
                    self.load_expert(lang, path)

    def save_expert(self, code, path=None):
        """Persist only the LoRA tensors.

        `save_pretrained` would also write the tied base layers — with
        `ensure_weight_tying` that means the full 233M embed_tokens *and* the
        233M lm_head alongside a 2.3M delta, i.e. ~1.7GB per language. Those
        base weights are identical for every expert and already on disk with
        the base model, so only the delta is stored.
        """
        path = path or os.path.join(self.adapter_dir, code, "lora.safetensors")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = get_peft_model_state_dict(self.model, adapter_name=code)
        lora_only = {k: v.detach().cpu().contiguous()
                     for k, v in state.items() if "lora_" in k}
        save_file(lora_only, path)
        return path

    def load_expert(self, code, path=None):
        path = path or os.path.join(self.adapter_dir, code, "lora.safetensors")
        if code not in self.model.peft_config:
            self.model.add_adapter(code, self.seed_cfg)
        state = load_file(path)
        set_peft_model_state_dict(self.model, state, adapter_name=code)
        self.loaded.add(code)
        return code

    def available_experts(self):
        return sorted(self.loaded - {"_base"})

    def set_active(self, lang):
        """Hot-swap the active expert. No-op (and no cache impact) if `lang`
        has no trained adapter yet — falls back to the unmodified base."""
        target = lang if lang in self.loaded else "_base"
        if target != self.active:
            self.model.set_adapter(target)
            self.active = target
        return self.active

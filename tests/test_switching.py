"""Mechanism tests for the weight-switching architecture.

These run on a tiny model, so they verify the *mechanism* — cache continuity
across a swap, real weight deltas, grad isolation — without needing the real
base model. This is the part of the design most likely to be silently wrong.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

TINY = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def tiny():
    tok = AutoTokenizer.from_pretrained(TINY)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32)
    cfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                     target_modules=["c_attn"], task_type="CAUSAL_LM")
    model = get_peft_model(base, cfg, adapter_name="_base")
    for name in ("hi", "zh"):
        model.add_adapter(name, model.peft_config["_base"])
    model.eval()
    return tok, model


def test_base_adapter_is_identity(tiny):
    """The baseline arm must be the genuine unmodified model, so the seed
    adapter's B matrices have to be zero (making its delta exactly zero)."""
    _, model = tiny
    zeros = [p for n, p in model.named_parameters()
             if "lora_B" in n and "_base" in n]
    assert zeros, "no _base lora_B parameters found"
    assert all(float(p.abs().sum()) == 0.0 for p in zeros)


def test_grad_isolation(tiny):
    """Training one expert must not drag gradients into the others."""
    _, model = tiny
    for active in ("hi", "zh"):
        model.set_adapter(active)
        trainable = {n.split(".")[-2] for n, p in model.named_parameters()
                     if p.requires_grad}
        assert trainable == {active}, f"active={active} but trainable={trainable}"


def test_cache_survives_adapter_swap(tiny):
    """The load-bearing property: keys/values computed under one expert must
    remain valid input after swapping to another, so the token history is
    never re-encoded."""
    tok, model = tiny
    ids = tok("the capital of india is", return_tensors="pt").input_ids

    model.set_adapter("hi")
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax())

        model.set_adapter("zh")
        cont = model(input_ids=torch.tensor([[nxt]]),
                     past_key_values=past, use_cache=True)

    assert cont.logits.shape[:2] == (1, 1)
    assert torch.isfinite(cont.logits).all()


def test_swapping_changes_output_once_experts_differ(tiny):
    """A swap has to actually change the computation — otherwise the router is
    measuring nothing. Perturb one adapter and confirm logits move."""
    tok, model = tiny
    ids = tok("hello world", return_tensors="pt").input_ids

    model.set_adapter("_base")
    with torch.no_grad():
        base_logits = model(input_ids=ids).logits.clone()

    # Give "hi" a non-zero delta, the way training would.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n and "hi" in n:
                p.add_(torch.randn_like(p) * 0.5)

    model.set_adapter("hi")
    with torch.no_grad():
        hi_logits = model(input_ids=ids).logits

    assert not torch.allclose(base_logits, hi_logits), \
        "adapter swap did not change the output"

    # And switching back must restore the base behaviour exactly.
    model.set_adapter("_base")
    with torch.no_grad():
        back = model(input_ids=ids).logits
    assert torch.allclose(base_logits, back, atol=1e-6)


def test_cache_reuse_matches_full_forward(tiny):
    """Incremental decoding with a retained cache must agree with a single
    full-sequence forward pass, when no swap intervenes. This guards the
    cache plumbing itself."""
    tok, model = tiny
    model.set_adapter("_base")
    ids = tok("the capital of india is new delhi", return_tensors="pt").input_ids

    with torch.no_grad():
        full = model(input_ids=ids).logits[:, -1, :]

        step = model(input_ids=ids[:, :-1], use_cache=True)
        inc = model(input_ids=ids[:, -1:],
                    past_key_values=step.past_key_values).logits[:, -1, :]

    assert torch.allclose(full, inc, atol=1e-4), \
        "incremental cache path diverges from full forward"


def test_odia_adapter_name_avoids_lora_collision(tmp_path):
    """The Odia code 'or' is a substring of the literal 'lora_', which makes
    PEFT silently fail to load that adapter's weights (it warns
    "Adapter name 'or' should not be contained in the prefix 'lora_'") and
    leaves the expert at zero-init — an untrained adapter that scores like the
    bare base. `peft_name` lifts every expert out of that collision.

    This reproduces the manager's exact save/load (get/set_peft_model_state_dict
    + safetensors) for 'or' and asserts the trained delta actually lands."""
    from safetensors.torch import load_file, save_file
    from peft import get_peft_model_state_dict, set_peft_model_state_dict

    from router.model_manager import peft_name

    assert peft_name("or") == "e_or"
    assert peft_name("or") not in "lora_"      # the whole point

    aname = peft_name("or")
    cfg = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                     target_modules=["c_attn"], task_type="CAUSAL_LM")

    model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32),
        cfg, adapter_name="_base")
    model.add_adapter(aname, model.peft_config["_base"])
    # Simulate training: give this expert a non-zero delta (lora_B starts at 0).
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n and aname in n:
                p.add_(torch.randn_like(p) * 0.5)

    state = get_peft_model_state_dict(model, adapter_name=aname)
    lora_only = {k: v.detach().cpu().contiguous()
                 for k, v in state.items() if "lora_" in k}
    saved_sum = sum(float(v.abs().sum()) for k, v in lora_only.items()
                    if "lora_B" in k)
    assert saved_sum > 0
    path = str(tmp_path / "lora.safetensors")
    save_file(lora_only, path)

    # Fresh model, load the file back under the same safe name.
    model2 = get_peft_model(
        AutoModelForCausalLM.from_pretrained(TINY, dtype=torch.float32),
        cfg, adapter_name="_base")
    model2.add_adapter(aname, model2.peft_config["_base"])
    set_peft_model_state_dict(model2, load_file(path), adapter_name=aname)

    loaded_sum = sum(float(p.abs().sum()) for n, p in model2.named_parameters()
                     if "lora_B" in n and aname in n)
    assert loaded_sum > 0, "Odia adapter loaded as zero — the lora_ collision"
    assert abs(loaded_sum - saved_sum) < 1e-3, \
        "loaded Odia weights do not match what was saved"

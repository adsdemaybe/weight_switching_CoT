"""Fine-tune one small LoRA adapter per language.

Objective: plain causal language modelling over text *in that language*.

This is deliberately not answer-format supervision. Training on targets like
"The answer is B." would teach the adapters to skip reasoning, and the eval
prompt asks for step-by-step CoT — so the adapters would suppress the very
behaviour being measured, deflating generated-token counts in a way that
looks like a token-efficiency win but is really just truncated reasoning.

Training on in-language text instead makes each adapter a *language*
specialist rather than a *task* specialist, which is what the routing
architecture actually claims to exploit, and leaves the CoT format entirely
to the shared base.
"""
import os
import torch
from torch.optim import AdamW

from .benchmarks import ALL_LANGS, load_language
from .model_manager import ExpertManager, peft_name


def _language_text(row):
    """Question and options as a plain in-language passage. No prompt
    scaffolding, no answer marker — just the language itself."""
    return row["question"] + "\n" + "\n".join(row["options"])


def _batch(tokenizer, rows, device, max_len=256):
    out = []
    for row in rows:
        ids = tokenizer(_language_text(row), return_tensors="pt",
                        truncation=True, max_length=max_len).input_ids
        if ids.shape[1] < 8:      # too short to carry signal
            continue
        out.append(ids.to(device))
    return out


def train_one_language(manager, code, rows, steps=60, lr=1e-5):
    # lr matters more than it looks. At 1e-4 for 60 steps the adapters destroy
    # the base model's instruction-following: routed generations collapsed to a
    # bare "Answer: C" in 3 tokens while the baseline still produced full CoT.
    # That reads as a ~98% token saving but is really just a broken model.
    # 1e-5 leaves the chat/CoT behaviour intact — verified before training.
    model = manager.model

    aname = peft_name(code)
    if aname not in model.peft_config:
        model.add_adapter(aname, model.peft_config["_base"])
    model.set_adapter(aname)
    manager.active = code
    model.train()

    examples = _batch(manager.tokenizer, rows, manager.device)
    if not examples:
        return None, []

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=lr)

    losses = []
    for step in range(steps):
        ids = examples[step % len(examples)]
        out = model(input_ids=ids, labels=ids)   # standard LM loss
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(out.loss.detach().item())

    model.eval()
    save_path = manager.save_expert(code)
    manager.loaded.add(code)
    return save_path, losses


def train_all(steps=60, n_train=48, force=False):
    """Resumable: languages whose adapter is already on disk are skipped, so
    an interrupted run continues by simply being restarted. Pass force=True to
    retrain everything from scratch."""
    manager = ExpertManager()
    trained = {}
    for lang_name, code, benchmark in ALL_LANGS:
        existing = os.path.join(manager.adapter_dir, code, "lora.safetensors")
        if os.path.isfile(existing) and not force:
            print(f"{code:3s} ({lang_name:10s}) already trained — skipping", flush=True)
            continue

        rows, source = load_language(lang_name, code, benchmark,
                                     split="train", limit=n_train)
        if len(rows) < 4:
            print(f"skip {code}: only {len(rows)} train rows ({source})", flush=True)
            continue
        path, losses = train_one_language(manager, code, rows, steps=steps)
        if path is None:
            print(f"skip {code}: no usable examples", flush=True)
            continue
        first = sum(losses[:5]) / len(losses[:5])
        last = sum(losses[-5:]) / len(losses[-5:])
        trained[code] = {"n": len(rows), "source": source,
                         "loss_first": first, "loss_last": last}
        print(f"{code:3s} ({lang_name:10s}) n={len(rows):3d} "
              f"loss {first:6.3f} -> {last:6.3f}", flush=True)
    return trained


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--n-train", type=int, default=48)
    ap.add_argument("--force", action="store_true",
                    help="retrain languages that already have an adapter")
    a = ap.parse_args()
    train_all(steps=a.steps, n_train=a.n_train, force=a.force)

"""Autoregressive loop over a single continuous token history.

The architectural claim being measured here is about *prefill*, not about
generated tokens. Two ways to use language-specific experts:

  dispatch    detect the language, hand the conversation to a separate
              specialist model, and re-encode the whole context so that model
              can see it. Every switch re-pays prefill over the full prefix.

  unified     keep one token history and one KV cache, and swap only the LoRA
              delta in place. The frozen base never changes, so keys/values
              already computed stay valid and prefill is paid exactly once.

`generate_unified` runs the second and additionally accounts for what the
first would have cost on the same trajectory, so the saving is measured rather
than asserted.
"""
import re
import torch

from .config import DEFAULT
from .script_detect import detect_language, language_shifted

ANSWER_RE = re.compile(r"answer\s*[:\-]?\s*\(?([A-D])\)?", re.IGNORECASE)


def build_prompt(tokenizer, question, options):
    opt_lines = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options))
    user = (
        f"{question}\n{opt_lines}\n\n"
        "Think step by step, then on the final line write exactly "
        "'Answer: <letter>'."
    )
    messages = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_unified(manager, prompt, cfg=None, route=True, seed_text=None):
    """Greedy decode. With `route` set, the language-shift signal hot-swaps the
    active expert mid-generation; with it unset the identical loop stays pinned
    to the base weights (the baseline arm).

    `seed_text` picks the *initial* expert. It matters: the full prompt is
    mostly chat-template markup and an English instruction, so detecting on it
    routes on the scaffolding rather than on the question — a Gujarati item
    was landing on the French expert. Callers pass the raw question.
    """
    cfg = cfg or DEFAULT
    tok = manager.tokenizer
    device = manager.device

    eos_ids = set(tok.eos_token_id if isinstance(tok.eos_token_id, list)
                  else [tok.eos_token_id])
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end >= 0:
        eos_ids.add(im_end)

    input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    if route:
        manager.set_active(detect_language(seed_text or prompt) or "_base")
    else:
        manager.set_active("_base")
    cur_lang = manager.active

    generated_ids = []
    text_buffer = ""
    switch_log = []
    pending_lang, pending_count = None, 0
    recompute_events = 0

    # Prefill cost actually paid: the prompt, once.
    prefill_tokens = prompt_len
    # What a dispatch architecture would have paid: the prompt once, plus a
    # re-encode of everything generated so far at each switch.
    dispatch_prefill_tokens = prompt_len

    with torch.no_grad():
        out = manager.model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        next_token_logits = out.logits[:, -1, :]

        for step in range(cfg.max_new_tokens):
            next_id = int(torch.argmax(next_token_logits, dim=-1).item())
            if next_id in eos_ids:
                break

            generated_ids.append(next_id)
            text_buffer += tok.decode([next_id])

            if cfg.stop_on_answer and ANSWER_RE.search(text_buffer):
                break

            if route and (step + 1) % cfg.check_every == 0:
                window = text_buffer[-cfg.window_chars:]
                shift = language_shifted(cur_lang, window, cfg)

                if shift is None:
                    pending_lang, pending_count = None, 0
                else:
                    # Require the same verdict on consecutive checks before
                    # committing, so a single noisy window can't cause a swap.
                    if shift == pending_lang:
                        pending_count += 1
                    else:
                        pending_lang, pending_count = shift, 1

                    if pending_count >= cfg.switch_patience:
                        switch_log.append(
                            {"step": step, "from": cur_lang, "to": shift}
                        )
                        manager.set_active(shift)
                        cur_lang = manager.active
                        pending_lang, pending_count = None, 0
                        # A dispatch design would re-encode the whole prefix
                        # here; the unified design reuses the live cache.
                        dispatch_prefill_tokens += prompt_len + len(generated_ids)

                        if cfg.recompute_on_switch:
                            # Rebuild the cache under the new expert so it
                            # attends over a history it encoded itself. Exact,
                            # but pays the re-prefill the unified design avoids.
                            full = torch.cat(
                                [input_ids,
                                 torch.tensor([generated_ids], device=device)],
                                dim=1,
                            )
                            out = manager.model(input_ids=full, use_cache=True)
                            past = out.past_key_values
                            next_token_logits = out.logits[:, -1, :]
                            prefill_tokens += full.shape[1]
                            recompute_events += 1
                            continue

            out = manager.model(
                input_ids=torch.tensor([[next_id]], device=device),
                past_key_values=past, use_cache=True,
            )
            past = out.past_key_values
            next_token_logits = out.logits[:, -1, :]

    n_gen = len(generated_ids)
    return {
        "text": tok.decode(generated_ids),
        "num_tokens": n_gen,
        "prompt_tokens": prompt_len,
        "prefill_tokens": prefill_tokens,
        "dispatch_prefill_tokens": dispatch_prefill_tokens,
        "prefill_saved": dispatch_prefill_tokens - prefill_tokens,
        "total_processed": prefill_tokens + n_gen,
        "dispatch_total_processed": dispatch_prefill_tokens + n_gen,
        "switches": switch_log,
        "recompute_events": recompute_events,
        "final_lang": cur_lang,
    }


def extract_answer(text):
    m = ANSWER_RE.search(text)
    return m.group(1).upper() if m else None

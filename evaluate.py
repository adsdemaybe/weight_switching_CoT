"""Shared evaluation routine used by both pipeline.py and optimize.py."""
import collections
import sys
import time

from router.config import DEFAULT
from router.generation import build_prompt, generate_unified, extract_answer


def evaluate(manager, items, route=True, cfg=None, per_item=False,
             label="", progress_every=10):
    """Run the eval set.

    Progress is printed periodically: a 1500-generation sweep otherwise looks
    identical to a hang for hours, which is how the first Kaggle run was lost.
    Set progress_every=0 to silence it.
    """
    cfg = cfg or DEFAULT
    correct = 0
    tot_gen = tot_proc = tot_dispatch = tot_saved = tot_switch = 0
    per_lang = collections.defaultdict(
        lambda: {"n": 0, "correct": 0, "gen": 0, "proc": 0, "switches": 0}
    )
    items_out = []
    started = time.time()

    for idx, row in enumerate(items, 1):
        if progress_every and (idx == 1 or idx % progress_every == 0):
            elapsed = time.time() - started
            rate = (idx - 1) / elapsed if elapsed > 0 and idx > 1 else 0
            eta = (len(items) - idx) / rate if rate > 0 else float("nan")
            print(f"  [{label or ('router' if route else 'baseline')}] "
                  f"{idx}/{len(items)}  "
                  f"acc={correct / max(idx - 1, 1):.3f}  "
                  f"{rate * 60:.1f} items/min  eta {eta / 60:.0f}m",
                  flush=True, file=sys.stderr)
        prompt = build_prompt(manager.tokenizer, row["question"], row["options"])
        # Seed the initial expert from the question itself, not the templated
        # prompt (which is dominated by English instruction scaffolding).
        r = generate_unified(manager, prompt, cfg=cfg, route=route,
                             seed_text=row["question"])
        pred = extract_answer(r["text"])
        hit = int(pred == row["answer"])

        correct += hit
        tot_gen += r["num_tokens"]
        tot_proc += r["total_processed"]
        tot_dispatch += r["dispatch_total_processed"]
        tot_saved += r["prefill_saved"]
        tot_switch += len(r["switches"])

        st = per_lang[row["lang"]]
        st["n"] += 1
        st["correct"] += hit
        st["gen"] += r["num_tokens"]
        st["proc"] += r["total_processed"]
        st["switches"] += len(r["switches"])

        if per_item:
            items_out.append({
                "lang": row["lang"], "gold": row["answer"], "pred": pred,
                "correct": hit, "switches": r["switches"],
                "text": r["text"][:400],
            })

    n = len(items) or 1
    out = {
        "accuracy": correct / n,
        "avg_gen_tokens": tot_gen / n,
        "avg_total_processed": tot_proc / n,
        "avg_dispatch_processed": tot_dispatch / n,
        "avg_prefill_saved": tot_saved / n,
        "token_saving_ratio": (tot_dispatch - tot_proc) / tot_dispatch if tot_dispatch else 0.0,
        "avg_switches": tot_switch / n,
        "n": len(items),
        "unanswered": sum(1 for i in items_out if i["pred"] is None) if per_item else None,
        "per_lang": {
            k: {
                "acc": v["correct"] / v["n"],
                "avg_gen": v["gen"] / v["n"],
                "avg_proc": v["proc"] / v["n"],
                "avg_switches": v["switches"] / v["n"],
                "n": v["n"],
            }
            for k, v in sorted(per_lang.items())
        },
    }
    if per_item:
        out["items"] = items_out
    return out


def summarize(result):
    """Drop the bulky per-item payload for sweep logs."""
    return {k: v for k, v in result.items() if k != "items"}

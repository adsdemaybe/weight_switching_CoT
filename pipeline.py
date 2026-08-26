"""Evaluation entrypoint.

Compares two configurations on the same items, same prompts, same decoding:

  baseline  one fixed set of weights for the whole generation (no routing)
  router    unified CoT with the active LoRA expert hot-swapped mid-generation
            whenever the language-shift signal fires, over one continuous
            token history and one continuous KV cache

Benchmarks: MILU (11 Indic languages) + MMMLU (9 non-Indic), both 4-option MCQ.

Run: python3 pipeline.py
Scores land in results/history.json; the run reports whether the last two runs
have plateaued on accuracy and token count.
"""
import argparse
import json
import os
import sys

from evaluate import evaluate, summarize
from router.benchmarks import load_eval_set
from router.config import RouterConfig, DEFAULT
from router.model_manager import ExpertManager

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
HISTORY_PATH = os.path.join(RESULTS_DIR, "history.json")
BEST_CONFIG_PATH = os.path.join(RESULTS_DIR, "sweep.json")
ACC_EPS = 0.01
TOK_EPS = 1.0
PLATEAU_RUNS = 2


def load_best_config():
    """Prefer the config the sweep settled on, if one exists."""
    if os.path.isfile(BEST_CONFIG_PATH):
        try:
            with open(BEST_CONFIG_PATH) as f:
                return RouterConfig(**json.load(f)["best"]), "sweep"
        except Exception:
            pass
    return DEFAULT, "default"


def load_history():
    if os.path.isfile(HISTORY_PATH):
        with open(HISTORY_PATH) as f:
            return json.load(f)
    return []


def save_history(history):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def check_plateau(history):
    if len(history) < PLATEAU_RUNS:
        return False
    for a, b in zip(history[-PLATEAU_RUNS:], history[-PLATEAU_RUNS + 1:]):
        if abs(a["router"]["accuracy"] - b["router"]["accuracy"]) > ACC_EPS:
            return False
        if abs(a["router"]["avg_total_processed"] - b["router"]["avg_total_processed"]) > TOK_EPS:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    items, sources = load_eval_set(limit_per_lang=args.per_lang)
    if not items:
        print("No eval items available. Aborting.")
        return 1
    langs = sorted({r["lang"] for r in items})
    print(f"Eval set: {len(items)} items / {len(langs)} languages: {' '.join(langs)}")

    cfg, cfg_src = load_best_config()
    print(f"Router config ({cfg_src}): {cfg.to_dict()}")

    manager = ExpertManager()
    print(f"Device: {manager.device}")
    print(f"Trained experts: {manager.available_experts() or '(none — base only)'}\n")

    import dataclasses
    recompute_cfg = dataclasses.replace(cfg, recompute_on_switch=True)

    baseline = evaluate(manager, items, route=False, cfg=cfg,
                        per_item=args.verbose, label="baseline")
    router = evaluate(manager, items, route=True, cfg=cfg,
                      per_item=args.verbose, label="router")
    # Third arm: same routing decisions, but the cache is rebuilt under the new
    # expert on each switch. Isolates the cost of reusing a history that a
    # different expert encoded. Slowest arm by far — it re-prefills the whole
    # prefix on every switch.
    exact = evaluate(manager, items, route=True, cfg=recompute_cfg,
                     per_item=args.verbose, label="recomputed")

    print(f"Baseline (no routing)     : acc={baseline['accuracy']:.3f}  "
          f"gen={baseline['avg_gen_tokens']:.1f}  processed={baseline['avg_total_processed']:.1f}")
    print(f"Router   (shared cache)   : acc={router['accuracy']:.3f}  "
          f"gen={router['avg_gen_tokens']:.1f}  processed={router['avg_total_processed']:.1f}  "
          f"switches={router['avg_switches']:.2f}")
    print(f"Router   (recomputed)     : acc={exact['accuracy']:.3f}  "
          f"gen={exact['avg_gen_tokens']:.1f}  processed={exact['avg_total_processed']:.1f}  "
          f"switches={exact['avg_switches']:.2f}")

    print(f"\nDecomposition:")
    print(f"  value of experts   (recomputed - baseline): "
          f"{exact['accuracy'] - baseline['accuracy']:+.3f} acc")
    print(f"  cost of shared cache (router - recomputed): "
          f"{router['accuracy'] - exact['accuracy']:+.3f} acc, "
          f"{router['avg_total_processed'] - exact['avg_total_processed']:+.1f} tokens")
    print(f"  net vs baseline      (router - baseline)  : "
          f"{router['accuracy'] - baseline['accuracy']:+.3f} acc, "
          f"{router['avg_total_processed'] - baseline['avg_total_processed']:+.1f} tokens")

    # The architectural claim: a dispatch design re-encodes the prefix on every
    # switch, the unified design does not.
    print(f"\nToken efficiency vs a re-encoding dispatch design:")
    print(f"  unified processed/item : {router['avg_total_processed']:.1f}")
    print(f"  dispatch processed/item: {router['avg_dispatch_processed']:.1f}")
    print(f"  prefill saved/item     : {router['avg_prefill_saved']:.1f} "
          f"({router['token_saving_ratio'] * 100:.1f}%)")

    print("\nper-language (baseline acc -> router acc, switches):")
    for lang in langs:
        b = baseline["per_lang"].get(lang, {})
        r = router["per_lang"].get(lang, {})
        print(f"  {lang:3s} n={b.get('n', 0):3d}  {b.get('acc', 0):.2f} -> {r.get('acc', 0):.2f}"
              f"   sw={r.get('avg_switches', 0):.2f}")

    history = load_history()
    history.append({
        "config": cfg.to_dict(),
        "config_source": cfg_src,
        "baseline": summarize(baseline),
        "router": summarize(router),
        "router_recomputed": summarize(exact),
        "sources": sources,
        "experts": manager.available_experts(),
        "per_lang_items": args.per_lang,
    })
    save_history(history)

    plateaued = check_plateau(history)
    print(f"\nRun {len(history)} recorded. Plateau reached: {plateaued}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Outer optimization loop over router mechanics.

Runs the eval once per candidate RouterConfig, keeps the best, and stops when
the best score stops improving for `--patience` consecutive rounds — i.e. the
benchmark score has plateaued.

The model is loaded once and reused across every candidate, so a sweep costs
one model load plus N evals.

Run: python3 optimize.py --per-lang 5
"""
import argparse
import itertools
import json
import os
import sys

from router.benchmarks import load_eval_set
from router.config import RouterConfig
from router.model_manager import ExpertManager
from evaluate import evaluate, summarize

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SWEEP_PATH = os.path.join(RESULTS_DIR, "sweep.json")

# Candidate values per knob. Kept small on purpose: each combination is a full
# eval pass, so the grid is the expensive part.
GRID = {
    "check_every": [2, 4, 8],
    "window_chars": [64, 128, 32],
    "switch_patience": [1, 2],
    "latin_min_conf": [0.90, 0.99],
}


def is_viable(cfg):
    """Reject configurations that cannot express the behaviour being swept.

    `window_chars < latin_min_chars` silently disables Latin detection
    entirely: the detector never sees enough Latin characters to return a
    verdict. Since the model reasons in English, nearly every switch is *into*
    `en`, so such a config collapses to ~0 switches and scores identically to
    baseline. A first sweep wasted its whole budget on this corner and then
    declared a "plateau" at the resulting flat score.
    """
    if cfg.window_chars < cfg.latin_min_chars:
        return False, (f"window_chars={cfg.window_chars} < "
                       f"latin_min_chars={cfg.latin_min_chars}: "
                       f"Latin script undetectable, routing disabled")
    return True, None


def candidates():
    """Viable grid points. Dropped combinations are returned too, so the
    caller can report them rather than silently narrowing the search."""
    keys = list(GRID)
    good, dropped = [], []
    for combo in itertools.product(*(GRID[k] for k in keys)):
        cfg = RouterConfig(**dict(zip(keys, combo)))
        ok, why = is_viable(cfg)
        (good if ok else dropped).append(cfg if ok else (cfg, why))
    return good, dropped


def score_of(result):
    """Primary objective is accuracy; ties break toward fewer tokens
    processed, since token efficiency is the secondary goal."""
    return (result["accuracy"], -result["avg_total_processed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-lang", type=int, default=3)
    ap.add_argument("--patience", type=int, default=3,
                    help="rounds without improvement before declaring plateau")
    ap.add_argument("--max-candidates", type=int, default=None)
    args = ap.parse_args()

    items, sources = load_eval_set(limit_per_lang=args.per_lang)
    if not items:
        print("No eval items available. Aborting.")
        return 1
    langs = sorted({r["lang"] for r in items})
    print(f"Eval set: {len(items)} items / {len(langs)} languages")

    manager = ExpertManager()
    print(f"Experts: {manager.available_experts() or '(none — base only)'}")

    cands, dropped = candidates()
    for cfg, why in dropped:
        print(f"  dropped {cfg.check_every}/{cfg.window_chars}/"
              f"{cfg.switch_patience}/{cfg.latin_min_conf}: {why}")
    if dropped:
        print(f"  ({len(dropped)} unviable configurations excluded)\n")
    if args.max_candidates:
        cands = cands[: args.max_candidates]
    print(f"Sweeping {len(cands)} router configurations\n")

    history = []
    best, best_score, stale = None, None, 0
    distinct_scores = set()

    for i, cfg in enumerate(cands, 1):
        result = evaluate(manager, items, route=True, cfg=cfg,
                          label=f"cfg{i}")
        s = score_of(result)
        improved = best_score is None or s > best_score

        if improved:
            best, best_score, stale = cfg, s, 0
        else:
            # Identical scores mean the knob under test did nothing, which is
            # not evidence of a plateau — the earlier sweep "converged" purely
            # because a degenerate corner of the grid produced four identical
            # numbers in a row. Only count a genuinely new-but-worse score
            # against the patience budget.
            if round(s[0], 4) in distinct_scores:
                print(f"    (score unchanged — not counted toward plateau)")
            else:
                stale += 1
        distinct_scores.add(round(s[0], 4))

        history.append({"config": cfg.to_dict(), "result": summarize(result),
                        "improved": improved})
        print(f"[{i:>2}/{len(cands)}] acc={result['accuracy']:.3f} "
              f"tok={result['avg_total_processed']:.0f} "
              f"sw={result['avg_switches']:.2f} "
              f"{'*BEST*' if improved else f'stale {stale}/{args.patience}'}"
              f"  {cfg.to_dict()}")

        if stale >= args.patience:
            print(f"\nPlateau: no improvement in {args.patience} consecutive rounds.")
            break

    # Held-out confirmation. `best` was SELECTED by its validation score across
    # many configs, so that score is optimistic — selection fits validation
    # noise. Re-score the winner once on a `test` split used for NEITHER
    # training NOR tuning; this is the honest generalization number and the one
    # to report. A positive validation-minus-test gap is the selection bias.
    held_out = None
    test_items, test_sources = load_eval_set(limit_per_lang=args.per_lang,
                                             split="test")
    if test_items:
        tr = evaluate(manager, test_items, route=True, cfg=best,
                      label="held-out-test")
        held_out = summarize(tr)
        gap = best_score[0] - tr["accuracy"]
        print(f"\nHeld-out TEST (winning config on unseen data): "
              f"acc={tr['accuracy']:.3f}  tok={tr['avg_total_processed']:.0f}  "
              f"sw={tr['avg_switches']:.2f}")
        print(f"  validation acc {best_score[0]:.3f} -> test acc "
              f"{tr['accuracy']:.3f}  (selection bias {gap:+.3f})")
    else:
        print("\nHeld-out TEST set empty (fixture mode?) — skipping confirmation.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(SWEEP_PATH, "w") as f:
        json.dump({"sources": sources, "best": best.to_dict(),
                   "best_score": {"accuracy": best_score[0],
                                  "avg_total_processed": -best_score[1]},
                   "held_out_test": held_out,
                   "test_sources": test_sources if test_items else None,
                   "history": history}, f, indent=2, ensure_ascii=False)

    print(f"\nBest config: {best.to_dict()}")
    print(f"Best (validation) accuracy: {best_score[0]:.3f}  "
          f"avg_total_processed: {-best_score[1]:.0f}")
    if held_out:
        print(f"Held-out test accuracy : {held_out['accuracy']:.3f}  "
              f"(report THIS as the generalization result)")
    print(f"Sweep written to {SWEEP_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

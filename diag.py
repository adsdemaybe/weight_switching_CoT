"""Minimal generation diagnostic: does the 7B emit 'Answer: <letter>' so
stop_on_answer fires and extract_answer works? Keeps GPU exposure tiny."""
import os, time
os.environ.setdefault("WSC_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

from router.benchmarks import load_eval_set
from router.model_manager import ExpertManager
from router.generation import build_prompt, generate_unified, extract_answer
from router.config import DEFAULT

items, _ = load_eval_set(limit_per_lang=1)
by = {r["lang"]: r for r in items}
m = ExpertManager()

for lang in ["en", "zh"]:
    r = by.get(lang)
    if r is None:
        continue
    t0 = time.time()
    prompt = build_prompt(m.tokenizer, r["question"], r["options"])
    out = generate_unified(m, prompt, cfg=DEFAULT, route=False,
                           seed_text=r["question"])
    dt = time.time() - t0
    pred = extract_answer(out["text"])
    print(f"--- {lang} gold={r['answer']} pred={pred} "
          f"tokens={out['num_tokens']} time={dt:.1f}s")
    print("TAIL:", repr(out["text"][-160:]))
print("DIAG_DONE")

"""Generate a self-contained Kaggle/Colab notebook from the current source.

The notebook embeds every source file verbatim, so it needs no git remote, no
uploaded dataset, and no credentials — open it, turn on the GPU, run all.

Generating it from the real files (rather than maintaining a parallel copy)
means the notebook cannot silently drift from the code it is supposed to run.

    python3 notebooks/build_notebook.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "run_on_gpu.ipynb")

SOURCE_FILES = [
    "requirements.txt",
    "router/__init__.py",
    "router/config.py",
    "router/script_detect.py",
    "router/model_manager.py",
    "router/generation.py",
    "router/benchmarks.py",
    "router/train_experts.py",
    "evaluate.py",
    "pipeline.py",
    "optimize.py",
    "tests/test_router.py",
    "tests/test_switching.py",
    "tests/test_pipeline_smoke.py",
    "data/milu_stub.jsonl",
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(True)}


def write_file_cell(paths):
    """One cell that writes several source files, using a heredoc-free approach
    so file content containing quotes or backslashes survives intact."""
    blobs = {}
    for rel in paths:
        full = os.path.join(ROOT, rel)
        if not os.path.isfile(full):
            continue
        with open(full, encoding="utf-8") as f:
            blobs[rel] = f.read()

    body = [
        "import os, json",
        "",
        "FILES = json.loads(r'''" + json.dumps(blobs, ensure_ascii=False) + "''')",
        "",
        "for path, content in FILES.items():",
        "    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)",
        "    with open(path, 'w', encoding='utf-8') as f:",
        "        f.write(content)",
        "print(f'wrote {len(FILES)} files')",
    ]
    return code("\n".join(body))


CELLS = [
    md("""# Token-Level Linguistic Router — GPU run

Trains 20 per-language LoRA experts, runs the three-arm evaluation, and sweeps
the router config to a plateau.

**Works on Kaggle and Colab.** Before running:

- **Kaggle**: Settings → Accelerator → **GPU T4 x2**, and Settings →
  **Internet: On** (needed to pull the base model and benchmarks).
  **Do not pick P100** — it is sm_60 and Kaggle's PyTorch only ships sm_70+
  kernels, so it fails. Cell 1 checks this and stops early.
- **Colab**: Runtime → Change runtime type → **T4 GPU**.

Then Run All. Everything is self-contained — no git remote, no credentials.
"""),

    md("## 1. Environment check"),
    code("""import os, sys, subprocess

gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                      '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
print(gpu or 'NO GPU DETECTED — enable the accelerator in settings, then rerun')

IN_KAGGLE = os.path.isdir('/kaggle')
IN_COLAB = os.path.isdir('/content') and not IN_KAGGLE
WORK = '/kaggle/working' if IN_KAGGLE else ('/content/work' if IN_COLAB else '.')
os.makedirs(WORK, exist_ok=True)
os.chdir(WORK)
print('platform:', 'kaggle' if IN_KAGGLE else 'colab' if IN_COLAB else 'local')
print('workdir :', os.getcwd())

# Fail here rather than 20 minutes in. Kaggle still offers P100 (sm_60), but
# its PyTorch build only ships kernels for sm_70+. Selecting P100 produces a
# pile of warnings and then dies on the first real kernel launch, which is a
# much more confusing failure than this assertion.
try:
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
        arches = torch.cuda.get_arch_list()
        print(f'device  : {name} (sm_{major}{minor})')
        if f'sm_{major}{minor}' not in arches:
            raise SystemExit(
                f'\\nINCOMPATIBLE GPU: {name} is sm_{major}{minor}, but this '
                f'PyTorch supports {arches}.\\n'
                f'Fix: right sidebar -> Accelerator -> "GPU T4 x2" (T4 is '
                f'sm_75), then rerun.\\n'
                f'P100 is sm_60 and will not work without rebuilding PyTorch.'
            )
    else:
        print('WARNING: CUDA not available — this will run on CPU and be very slow.')
except ImportError:
    pass   # torch arrives with the install cell below"""),

    md("## 2. Dependencies"),
    code("""# Kaggle preinstalls torchao 0.10.0. PEFT's is_torchao_available() *raises*
# on a version below 0.16 instead of returning False, so building any LoRA
# module dies with "Found an incompatible version of torchao". Nothing here
# uses torchao, and with it absent that check returns False cleanly — so the
# fix is to remove it rather than chase a compatible build.
%pip uninstall -q -y torchao
%pip install -q -U transformers peft datasets accelerate safetensors langid pytest

import importlib.util
import torch, transformers, peft
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
print('bf16 supported:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
print('transformers', transformers.__version__, '| peft', peft.__version__)
print('torchao present:', importlib.util.find_spec('torchao') is not None, '(should be False)')

# Prove the LoRA path actually builds before spending 10 minutes on training.
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
_m = get_peft_model(
    AutoModelForCausalLM.from_pretrained('sshleifer/tiny-gpt2', dtype=torch.float32),
    LoraConfig(r=4, target_modules=['c_attn'], task_type='CAUSAL_LM'),
    adapter_name='_smoke')
print('LoRA construction OK')"""),

    md("""## 3. Write the source

Embedded verbatim from the repo by `notebooks/build_notebook.py`."""),
    write_file_cell(SOURCE_FILES),

    md("""## 4. Tests

These run on a tiny model and need no GPU. They cover the load-bearing
properties: the KV cache stays valid across an adapter swap, the incremental
decode path matches a full forward, the baseline adapter is exactly identity,
gradients isolate to the active adapter, and routed generation does not
collapse (a guard against adapters that destroy instruction-following and
thereby fake a token-efficiency win).

**Do not skip this cell.** It exercises the same PEFT/LoRA construction path
that training uses, on a tiny model, in seconds — so environment breakages
surface here rather than after a long model download."""),
    code("!python -m pytest tests/ -q"),

    md("""## 5. Train the 20 language experts

Plain causal-LM loss over in-language text — deliberately *not* answer-format
supervision, which would teach the adapters to skip the reasoning the eval is
measuring.

`lr=1e-5` is load-bearing: at `1e-4` for 60 steps the adapters destroy the
base model's instruction-following, and routed runs collapse to a bare
`Answer: C` in 3 tokens. That scores as a ~98% token saving but is just a
broken model.

Resumable — already-trained languages are skipped, so re-running after a
disconnect continues where it stopped."""),
    code("!python -m router.train_experts --steps 60 --n-train 48"),

    md("""## 6. Three-arm evaluation

| arm | routing | cache on switch |
|---|---|---|
| baseline | off | n/a |
| router | on | retained (cheap, approximate) |
| router_recomputed | on | rebuilt under the new expert (exact, re-pays prefill) |

The third arm exists because retaining the cache means the new expert attends
over a history a *different* expert encoded. Without it, a flat result is
uninterpretable — you cannot tell whether the experts are worthless or the
cache shortcut ate the gains.

Cost scales as `per_lang x 20 languages x 3 arms` generations, each up to 512
tokens. `--per-lang 5` is 300 generations (~20-40 min on a T4) and is enough to
see whether the numbers are sane. Raise it to 15-25 for a reportable result
once a small run has come back clean — below ~10 per language the accuracy
figures are mostly noise.

The `recomputed` arm is much slower than the other two: it re-prefills the
entire prefix on every expert switch, which is exactly the cost being
measured.

Progress prints to stderr every 10 items with a rate and ETA, so a long run is
distinguishable from a hang."""),
    code("!python pipeline.py --per-lang 5"),

    md("""## 7. Sweep the router to a plateau

Sweeps `check_every` x `window_chars` x `switch_patience` x `latin_min_conf`,
keeping the best and stopping once the score stops improving."""),
    code("!python optimize.py --per-lang 15 --patience 3"),

    md("## 8. Results"),
    code("""import json, os
for name in ['results/history.json', 'results/sweep.json']:
    if os.path.isfile(name):
        d = json.load(open(name))
        print('='*70); print(name)
        if name.endswith('history.json'):
            r = d[-1]
            for arm in ['baseline', 'router', 'router_recomputed']:
                if arm in r:
                    a = r[arm]
                    print(f"  {arm:20s} acc={a['accuracy']:.3f} "
                          f"gen={a['avg_gen_tokens']:.1f} proc={a['avg_total_processed']:.1f}")
            b, rt = r['baseline'], r['router']
            ex = r.get('router_recomputed', rt)
            print(f"  value of experts    (recomputed - baseline): {ex['accuracy']-b['accuracy']:+.3f}")
            print(f"  cost of shared cache (router - recomputed) : {rt['accuracy']-ex['accuracy']:+.3f}")
            print(f"  net                  (router - baseline)   : {rt['accuracy']-b['accuracy']:+.3f}")
            print(f"  prefill saved/item vs dispatch design      : {rt['avg_prefill_saved']:.1f} "
                  f"({rt['token_saving_ratio']*100:.1f}%)")
        else:
            print('  best config :', d['best'])
            print('  best score  :', d['best_score'])
            print('  candidates  :', len(d['history']))
    else:
        print(name, 'MISSING')"""),

    md("""### Sanity check before believing any efficiency number

Two of the three bugs found while building this failed *toward* a flattering
token-efficiency result. Confirm the reasoning traces are intact — routed
generations should look like real CoT, not a bare answer line."""),
    code("""import json
h = json.load(open('results/history.json'))[-1]
b, r = h['baseline']['avg_gen_tokens'], h['router']['avg_gen_tokens']
print(f'baseline {b:.1f} tokens vs routed {r:.1f} tokens  (ratio {r/b:.2f})')
print('OK — reasoning preserved' if r >= 0.4*b else
      'SUSPECT — routed generation collapsed; the "saving" is a broken model')"""),

    md("## 9. Package results for download"),
    code("""!zip -qr results.zip results adapters && ls -lh results.zip
# Kaggle: appears under the notebook's Output tab.
# Colab: uncomment to download directly.
# from google.colab import files; files.download('results.zip')"""),
]


def main():
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    size = os.path.getsize(OUT)
    print(f"wrote {OUT} ({size/1024:.0f} KB, {len(CELLS)} cells, "
          f"{len(SOURCE_FILES)} source files embedded)")


if __name__ == "__main__":
    main()

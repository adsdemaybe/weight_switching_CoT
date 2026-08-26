# Running on Kaggle or Colab

`run_on_gpu.ipynb` is self-contained: it embeds every source file, so it needs
no git remote, no uploaded dataset, and no credentials. Open it, enable the
GPU, Run All.

Free T4 beats an M3 Pro badly here — generation is the bottleneck and CUDA runs
the base model in half precision. What takes hours locally is minutes there.

## Kaggle (recommended)

Better for this job than Colab: longer sessions, more reliable GPU allocation,
and `/kaggle/working` persists as a downloadable output artifact.

1. https://kaggle.com/code → **New Notebook** → File → **Upload Notebook** →
   `run_on_gpu.ipynb`
2. Right sidebar → **Accelerator: GPU T4 x2** (or P100)
3. Right sidebar → **Internet: On** — required, the notebook pulls the base
   model and the benchmarks from Hugging Face. Kaggle may ask you to verify a
   phone number before it lets you enable this.
4. **Run All**
5. Results land in the **Output** tab as `results.zip`

Quota: ~30 GPU-hours/week, 9h per session. The whole run is far inside that.

## Colab

1. https://colab.research.google.com → **Upload** → `run_on_gpu.ipynb`
2. Runtime → Change runtime type → **T4 GPU**
3. **Run All**
4. Last cell downloads `results.zip` (uncomment the `files.download` line)

Colab disconnects idle sessions aggressively. Training is resumable — rerun
the training cell and it skips languages already done — but `/content` is wiped
on a full runtime reset, so mount Drive if you want to survive that.

## Regenerating

The notebook is built from the real source, so it cannot silently drift:

```bash
python3 notebooks/build_notebook.py
```

Rerun this after any code change and re-upload.

## Runtime expectations

| stage | ~T4 |
|---|---|
| deps + model download | 3–5 min |
| train 20 experts | 5–10 min |
| three-arm eval, `--per-lang 25` | 20–40 min |
| sweep, `--per-lang 15` | 30–60 min |

Lower `--per-lang` if you are short on session time. Below ~10 items per
language the accuracy numbers are mostly noise and not worth reporting.

## Reading the output

The eval reports three arms and decomposes the result:

```
value of experts     = recomputed - baseline
cost of shared cache = router     - recomputed
net                  = router     - baseline
```

The third arm matters. Retaining the KV cache across a switch means the new
expert attends over a history a *different* expert encoded. Without measuring
the recomputed arm, a flat net result is uninterpretable — you cannot tell
whether the experts are worthless or whether the cache shortcut consumed the
gains.

**Be suspicious of a large token-efficiency win.** Two of the three bugs found
while building this failed *toward* a flattering efficiency number — adapters
that destroyed instruction-following collapsed generation to a bare
`Answer: C`, which scores as a ~98% saving. The last cell checks routed
generation length against baseline for exactly this reason.

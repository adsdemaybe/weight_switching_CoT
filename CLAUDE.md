# Current Objective
- Build, test, and optimize an internal Token-Level Linguistic Router inside an autoregressive text generation pipeline. The system must dynamically switch model weights mid-generation inside the thinking state based on language-as-intent signals, minimizing token usage on cross-lingual tasks. Iterate using the pre-validated AI4Bharat MILU / Indic LLM Arena benchmark dataset until the accuracy score and token-efficiency gains hit a definitive ceiling and stop changing.

# How the Router Hot-Swaps Experts (mechanism)
Two independent parts: the TRIGGER (when to switch) and the SWAP (how).

TRIGGER — `router/script_detect.py`, driven from `router/generation.py`:
- Every `check_every=4` generated tokens, look at the last `window_chars=64`
  chars of generated text and run `detect_language()`.
- Detector: count Unicode scripts in the window. Kana present -> ja; Hangul
  present -> ko (these beat the dominant-script rule). Else dominant script
  wins: Han -> zh, Devanagari -> hi/mr via closed-class function words, each
  Brahmic script -> its language. Latin needs >=40 chars AND langid conf >=0.90
  or it returns None.
- None = "not enough evidence" -> HOLD current expert, never switch on noise.
- Switch fires only when detected lang != active expert and it persists for
  `switch_patience=1` checks. `language_shifted()` returns the new code.

SWAP — `router/model_manager.py`:
- One frozen base (Qwen2.5-1.5B) + all 20 LoRA adapters preloaded in one
  `PeftModel`. Switch = `self.model.set_adapter(code)`. Pure pointer flip:
  no reload, no disk read, no new model — just which LoRA deltas are active.

CACHE CONTINUITY (the whole point) — `router/generation.py`:
- Weights flip but `past_key_values` is carried across the swap UNCHANGED. The
  new expert attends over history the old expert encoded; the next token is
  computed under the new adapter using the old cache. Prefix is never rebuilt
  = the 44.2% prefill saving. The `recompute_on_switch=True` arm instead
  rebuilds the cache under the new expert on every switch (exact but re-pays
  full prefill) and exists only to measure what the shortcut costs.

# GB10 Training/Serving Machine (set up & used 2026-08-26)
- Host: `ssh advaith@promaxgb10-6c88` (Tailscale SSH, user `advaith`). NOTE:
  Tailscale SSH uses a `check` re-auth window — access was revoked mid-session
  once and needed the user to re-`tailscale up`. Detached jobs survive it.
- HW: NVIDIA GB10 Grace Blackwell (sm_121), driver 580, CUDA 13.0, aarch64,
  121GB UNIFIED memory (CPU+GPU share it), 3.4T free. `nvidia-smi` reports
  memory as "Not Supported" here — do not rely on it for OOM checks.
- Base model bumped 1.5B -> **Qwen2.5-7B-Instruct** for both serve and train.
  Set via `WSC_BASE_MODEL` (model_manager reads it). Train and generate MUST
  use the same value or the LoRA attaches to a different embedding geometry.

## Environment (two isolated uv venvs, both on uv-MANAGED Python 3.12.14)
- `~/weight_switching_CoT/.venv`      training: torch 2.13.0+cu130 (aarch64),
  transformers 5.16.1, peft 0.20.0.
- `~/weight_switching_CoT/.venv-vllm` serving:  vllm 0.28.0 + torch 2.13.0+cu130.
- WHY managed Python, not system: system `/usr/bin/python3.12` has no dev
  headers and there is NO sudo. Both torch inductor AND vLLM JIT-compile a
  `cuda_utils.c` at runtime and need `Python.h`. uv's managed CPython bundles
  headers, so recreating the venvs with `uv venv --python-preference
  only-managed` fixes it with no root. (`Python.h: No such file or directory`.)
- vLLM also shells out to `ninja` for that JIT; `ninja` must be on PATH of the
  server process. `serve_vllm.sh` prepends `.venv-vllm/bin` to PATH for this.

## Scripts on the box
- `serve_vllm.sh`  serves Qwen2.5-7B on :8000 (`--served-model-name qwen`,
  `--gpu-memory-utilization 0.45` so training can coexist). Verified up:
  `/v1/models` returns `qwen`, KV cache 39.6 GiB, ~90x concurrency.
- `train.sh`       `WSC_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct python -m
  router.train_experts --steps 60 --n-train 48`. Detach with
  `setsid bash -c "./train.sh > ~/train.log 2>&1" </dev/null &`.

## RESULT of the 7B training run (2026-08-26)
- All 20 adapters trained fresh vs the 7B base. ~20MB each (LoRA-only save path
  held; NOT the 1.7GB regression). 5.01M trainable params/adapter.
- Losses (first5->last5 avg; noisy because the two windows are different
  examples, not the same batch): gu 1.20->0.78, hi 1.91->1.38, kn 1.28->0.99,
  en 2.92->2.00, ko 3.12->2.29, ar 2.75->2.10, most others down similarly.
  bn/ml ticked up — within the example-variance of this weak 60-step fit.
- 24/24 tests pass on the box after the change below.

## Design fix forced by 7B: lm_head LoRA (model_manager.py)
- Qwen2.5-7B does NOT tie embed_tokens to lm_head (the 0.5B/1.5B DO). So the
  old `embed_tokens` + `ensure_weight_tying=True` left the OUTPUT head
  un-adapted on 7B — the per-language embedding delta reached reading but not
  scoring, and peft warned "no tied modules were found".
- Fix: branch on `base.config.tie_word_embeddings`. Tied -> keep embed_tokens +
  ensure_weight_tying (delta reaches lm_head via the shared tensor). Untied ->
  add `lm_head` to target_modules explicitly and drop ensure_weight_tying.
  Verified on 7B: targets = {q,v,embed,lm_head}, lm_head LoRA present.

## Running the eval/sweep on GB10 (in progress 2026-08-26)
- The box is SHARED (another user runs a gaussian-splatting job on the GPU).
  Be a good neighbour: run the ~16GB in-process eval, and do NOT also hold the
  ~56GB vLLM server up during heavy compute. Stop vLLM for the sweep, restart
  after. `serve_vllm.sh` uses `--gpu-memory-utilization 0.45` for this reason.
- The eval is a per-token Python greedy loop on 7B (one forward per token — the
  price of in-process mid-gen switching; vLLM's batching is unavailable to it).
  ~2-3 items/min, so 100 items x 3 arms is ~hours and a full sweep is longer.
  `stop_on_answer` DOES fire (breaks on the `Answer:` line), so gens aren't
  running the full 512 — the slowness is inherent per-token overhead.
- Current run (detached, `~/run.log`): `pipeline.py --per-lang 5` then
  `optimize.py --per-lang 3 --patience 2`, `WSC_BASE_MODEL=...7B-Instruct`.
- The old 1.5B numbers earlier in this file are STALE for the 7B base.
- vLLM serves the base only; the routing cache-sharing swap still runs
  in-process via peft (vLLM cannot do it — see Notes/Constraints).

# Build & Test Commands
- Build: `python3 -m pip install -r requirements.txt` (or framework setup equivalent)
- Test: `python3 pipeline.py` (Executes the evaluation loop, prints token efficiency and benchmark accuracy scores)

# Development Rules
1. Before starting a new turn, read this file to see what passed or failed.
2. If tests fail, do not write new features. Fix the error immediately.
3. Keep track of what you have finished under the Progress section below.
4. When `python3 pipeline.py` yields 0 errors and the benchmark evaluation scores hit a plateau (stop changing across iterations), stop the execution loop and declare completion.

# Progress Tracking
- [x] Initialize Environment (venv, torch 2.13 + MPS, transformers 5.14, peft 0.20)
- [x] Benchmark data access
      - `ai4bharat/MILU` is GATED for this account; requests via API are 403.
        Using `murthyrudra/milu-cleaned`, an ungated mirror with identical
        per-language configs and fields. `benchmarks.py` still tries the
        canonical repo first, so this upgrades automatically if access lands.
      - Non-Indic languages come from `openai/MMMLU` (ungated, official).
      - 20 languages total / 8 scripts: bn gu hi kn ml mr or pa ta te en
        (MILU) + zh ja ko de fr es it pt ar (MMMLU).
- [x] Token-Level Linguistic Router (`router/script_detect.py`) — 14/14 on the
      detection test. Unicode script is the primary signal; two families need
      more: Kana/Hangul presence decides ja/ko over co-occurring Han, and
      Latin-script languages fall through to langid with a 40-char / 0.90-conf
      floor. hi vs mr uses closed-class function words, because langid
      actually inverts that pair on short text.
- [x] Expert manager (`router/model_manager.py`) — one frozen base, per-language
      LoRA adapters swapped via `set_adapter`. Verified: grads isolate to the
      active adapter, and the `_base` adapter is zero-init in `lora_B`, so the
      baseline arm is the true unmodified model.
- [x] Unified continuous token history (`router/generation.py`) — verified that
      `past_key_values` computed under one adapter feeds the next after a
      swap, so the cache is carried across switches, never rebuilt.
- [x] Base model downloaded (Qwen2.5-1.5B-Instruct) and verified on MPS
- [x] Per-language token embeddings — `embed_tokens` added to the LoRA target
      set with `ensure_weight_tying=True`. Without this the experts shared one
      frozen embedding matrix and had no language-specific token
      representation at all. 2.32M trainable params per adapter.
- [x] Test suite — 20 tests, no base model needed. Covers cache continuity
      across a swap, incremental-vs-full-forward agreement, `_base` identity,
      grad isolation, detector behaviour, token-accounting consistency, and a
      full three-arm end-to-end run on tiny-gpt2.
- [x] Train per-language LoRA experts — DONE. First 20 on 1.5B (Kaggle).
      Retrained all 20 fresh on GB10 vs **Qwen2.5-7B-Instruct** (2026-08-26),
      now with the lm_head LoRA fix for the untied 7B. ~20MB each, on the GB10
      at `~/weight_switching_CoT/adapters/`. See the GB10 section below.
- [ ] Benchmark execution & baseline metrics capture
- [ ] Optimization loop until plateau

# Resume Here — on Kaggle (preferred) or Colab

Local runs are stopped at user request (laptop heat). Run it on a free GPU:

  Upload `notebooks/run_on_gpu.ipynb` to Kaggle, set Accelerator=GPU T4 x2 and
  Internet=On, Run All. See `notebooks/README.md`.

The notebook embeds all source verbatim — no git remote, no credentials, no
uploaded dataset needed. Regenerate with `python3 notebooks/build_notebook.py`
after any code change, then re-upload. Verified: extracting the embedded files
into a clean dir and running pytest gives 22/22.

(Training now runs on the GB10 box — see the GB10 section below. The Kaggle
notebook path is kept as a fallback. The AWS path was removed 2026-08-26.)

All 20 adapters ARE trained and on disk (lr=1e-5). Adapters are ~9.3MB each;
if one comes out at ~1.7GB the LoRA-only save path has regressed — see the
`save_expert` docstring.

# RESULTS — first real run (Kaggle T4, n=100, 5 items/language)

`results/history.json`. Three arms, same items, same decoding:

  baseline (no routing)   acc 0.340   gen 303.2   processed 471.8   sw 0.00
  router   (shared cache) acc 0.300   gen 297.6   processed 466.2   sw 1.50
  router   (recomputed)   acc 0.290   gen 304.2   processed 826.8   sw 1.47

  value of experts    (recomputed - baseline) = -0.050
  cost of shared cache (router - recomputed)  = +0.010
  net                  (router - baseline)    = -0.040

NOT SIGNIFICANT: diff -0.040, SE 0.066, z = -0.61, 95% CI about [-0.17, +0.09].
n=100 cannot resolve a 4-point difference. Do not report this as "routing
hurts" — report it as "no detectable effect at this sample size".

The one solid positive: 466 processed/item vs 835 for a re-encoding dispatch
design = 44.2% prefill saved. That is the architectural claim and it holds.
The recomputed arm costs 827 tokens to buy -0.01 accuracy, i.e. paying full
price for cache exactness bought nothing.

Generation length ratio 0.98 (297.6 vs 303.2) — reasoning intact, so this is
NOT the collapse failure mode. The experts simply do not help.

# The sweep from that run is INVALID — results/sweep_INVALID.json
`window_chars=32 < latin_min_chars=40` makes Latin script undetectable. The
model reasons in English, so nearly every switch is *into* `en`; that config
collapsed switching from 1.50 to 0.03 and scored flat. itertools ordering hit
that corner first, all 4 candidates tied, and the patience counter declared a
"plateau" at 0.263 having never tried window_chars 64 or 128.

Fixed in `optimize.py`: `is_viable()` drops such configs and logs them,
window_chars ordering now puts 64 first, and identical scores no longer count
toward the patience budget (a knob that does nothing is not evidence of
convergence). 24 of 36 grid points are viable. Regression tests added.

A VALID SWEEP HAS STILL NOT BEEN RUN.

# Bugs found by actually running it (all fixed, all would have faked a result)
1. Router seeded the initial expert from the full templated prompt, which is
   mostly chat markup + an English instruction — a Gujarati item routed to the
   French expert. Fixed: seed from the raw question (`seed_text`).
2. lr=1e-4 x 60 steps destroyed instruction-following. Routed runs collapsed
   to a bare "Answer: C" in 3 tokens while baseline produced full CoT — which
   scores as a ~98% token saving. Fixed: lr=1e-5, verified CoT survives.
   Regression test added (routed length must stay >=40% of baseline).
3. max_new_tokens=200 truncated generation mid-CoT, so no answer line was ever
   emitted and every arm would have scored ~0. Fixed: 512.

Note that (2) and (3) both fail *toward* a flattering efficiency number. Be
suspicious of any large token-efficiency win that is not accompanied by intact
reasoning traces.
4. Odia adapter silently loaded as ZERO. The code `or` is a substring of the
   literal `lora_`, so PEFT's key handling collided and `set_peft_model_state_dict`
   never wrote the weights — the Odia "expert" was the bare base, untrained.
   PEFT only *warns* ("Adapter name 'or' should not be contained in the prefix
   'lora_'"), so it passed silently through the 1.5B Kaggle run too. Fixed:
   `peft_name(code)` maps every expert to `e_<code>` at all add/get/set/
   set_adapter sites (model_manager + train_experts). Saved safetensors are
   name-independent (canonical keys), so the 20 existing files load unchanged —
   no re-save. Verified on 7B: 0 collision warnings, `or` lora_B sum 403.3
   (was 0). Regression test `test_odia_adapter_name_avoids_lora_collision`.

# Notes / Constraints
- Mid-generation weight switching over a shared KV cache requires in-process
  tensor access. Confirmed from vLLM's docs that `LoRARequest` is bound at
  request-submission time — a different adapter means a separate request, i.e.
  fresh prefill and fresh cache, which is exactly the cost this design avoids.
  Ollama and hosted APIs expose no cache at all. vLLM also has no arm64 macOS
  wheel, but that is the lesser problem.
- Separate per-language *models* would give better embeddings but different
  tokenizers and different cache geometry, so the cache could not be shared
  and every switch would re-pay full prefill. Embedding-LoRA over one frozen
  base is the strongest version of per-language embeddings that still permits
  the single continuous token history the objective requires. These two
  requirements genuinely trade off.
- Expert training uses plain LM loss over in-language text, NOT answer-format
  supervision. Training on "The answer is B." would teach the adapters to skip
  reasoning while the eval asks for CoT — deflating token counts in a way that
  mimics an efficiency win but is really truncated reasoning.
- The adapters get ~60 steps over ~48 passages each. Loss moves (e.g. Gujarati
  1.311 -> 0.842) but this is not a serious fine-tune. Report results as
  "does the mechanism work", not "are these experts competitive".

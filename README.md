# Token-Level Linguistic Router

An autoregressive generation wrapper that treats a **language shift in the
generated text as a routing signal**, and hot-swaps a language-specific LoRA
expert mid-generation — over a **single continuous token history and a single
KV cache**.

```
query ──▶ prefill (once) ──▶ decode ──┬─▶ token
                                      │
                     every N tokens:  ├─▶ detect language of trailing window
                                      │      │
                                      │      └─▶ shifted? set_adapter(lang)
                                      │           (cache is NOT rebuilt)
                                      └─▶ continue with same past_key_values
```

## Why the weights are LoRA deltas and not separate models

The requirement "switch the underlying model weights mid-generation" and the
requirement "maintain a single continuous token history" are in tension. A
genuinely different model per language has a different tokenizer and different
cache geometry, so its keys/values are not interchangeable — every switch would
force a full re-encode of the prefix.

Keeping one frozen base and swapping only a LoRA delta resolves this: the base
never changes, so every key/value already computed stays valid, and
`set_adapter` is a pointer flip rather than a reload.

`embed_tokens` is included in the LoRA target set (with `ensure_weight_tying`,
since this base ties embeddings to `lm_head`). Without it the "experts" would
share one frozen embedding matrix and have no language-specific token
representation at all. This is the strongest form of per-language embeddings
that still permits a shared cache.

## Why not vLLM / Ollama / a hosted API

None of them can express this. vLLM binds a `LoRARequest` at request-submission
time — the adapter is fixed for the lifetime of a request, and using a different
one means a **separate request**, i.e. a fresh prefill and a fresh cache. Ollama
and hosted APIs are text-in/text-out and expose no cache at all. Mid-generation
swapping over a retained cache needs in-process tensor access.

(vLLM additionally has no arm64 macOS wheel, but that is the lesser problem.)

## The measurement

Three arms, same items, same prompts, same decoding:

| arm | routing | cache on switch |
|---|---|---|
| `baseline` | off — base weights throughout | n/a |
| `router` | on | **retained** (cheap, approximate) |
| `router_recomputed` | on | rebuilt under the new expert (exact, re-pays prefill) |

This decomposes the result:

```
value of experts     = recomputed - baseline
cost of shared cache = router     - recomputed
net                  = router     - baseline
```

The third arm exists because retaining the cache means the new expert attends
over a history that a *different* expert encoded. Without separating these, a
flat net result is uninterpretable — you cannot tell whether the experts are
worthless or whether the cache shortcut consumed the gains.

**Token accounting** measures prefill, not generated tokens. The architecture
does not change how much text is produced; it changes how many times the prefix
is encoded. `dispatch_prefill_tokens` tracks what a re-encoding design would
have paid on the same trajectory, so the saving is measured rather than asserted.

## Router

Unicode script is the primary signal — deterministic, no model call. Two
families need more:

- **CJK**: Japanese mixes Han with Kana and is often *majority* Han, so a plain
  dominant-script rule sends it to Chinese. Kana/Hangul presence is treated as
  decisive.
- **Latin**: en/de/fr/es/it/pt share one block, so script is uninformative.
  These fall through to `langid` behind a 40-char / 0.90-confidence floor.
- **hi vs mr**: both Devanagari. `langid` *inverts* this pair on short text, so
  it uses closed-class function words (copulas, negators, genitive markers)
  plus the Marathi-only character ळ.

Insufficient evidence returns `None`, which means "hold the current expert"
rather than "fall back to a default" — otherwise the router thrashes on
whitespace and digits.

## Benchmarks

20 languages / 8 scripts, all 4-option MCQ.

- **MILU** (11 Indic: bn gu hi kn ml mr or pa ta te en). `ai4bharat/MILU` is
  gated; `benchmarks.py` tries it first and falls back to the ungated
  `murthyrudra/milu-cleaned` mirror, which carries identical per-language
  configs and fields.
- **MMMLU** (9 non-Indic: zh ja ko de fr es it pt ar), `openai/MMMLU`.

Train and eval slices are disjoint by construction (MILU: `validation` vs
`test`; MMMLU: a fixed `EVAL_RESERVE` offset).

## Expert training

Plain causal-LM loss over text *in that language*. Deliberately **not**
answer-format supervision — training on `"The answer is B."` would teach the
adapters to suppress reasoning while the eval prompt asks for CoT, deflating
token counts in a way that mimics an efficiency win but is really just
truncated reasoning. The adapters are language specialists; CoT format is left
to the shared base.

## Layout

```
router/
  script_detect.py   language-shift signal
  config.py          sweepable router hyperparameters
  model_manager.py   frozen base + hot-swappable LoRA experts
  generation.py      decode loop, cache continuity, token accounting
  benchmarks.py      MILU + MMMLU, normalized to one schema
  train_experts.py   per-language LoRA training
evaluate.py          shared eval used by both entrypoints
pipeline.py          three-arm comparison            -> results/history.json
optimize.py          router config sweep to plateau  -> results/sweep.json
tests/               20 tests, no base model required
```

## Run

```bash
python3 -m pip install -r requirements.txt
python3 -m router.train_experts     # train the per-language adapters
python3 pipeline.py                 # three-arm evaluation
python3 optimize.py                 # sweep router mechanics to plateau
python3 -m pytest tests/ -q
```

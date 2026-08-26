"""Router hyperparameters in one place, so the optimization loop can sweep
them without editing module constants."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class RouterConfig:
    # How often (in generated tokens) the language signal is evaluated.
    # Lower = more responsive, more detector calls.
    check_every: int = 4

    # Trailing decoded characters the detector sees. Too short starves the
    # Latin classifier; too long makes the router laggy after a real shift.
    window_chars: int = 64

    # Minimum script-bearing characters before any verdict is offered.
    min_chars: int = 4

    # Latin-script languages are not separable by script, so they need more
    # evidence and a confidence floor before a switch is allowed.
    latin_min_chars: int = 40
    latin_min_conf: float = 0.90

    # Require the same new language to be observed on this many consecutive
    # checks before actually swapping. Suppresses one-window flickers.
    switch_patience: int = 1

    # Stop as soon as the answer line is emitted.
    stop_on_answer: bool = True

    # MILU/MMMLU items need real reasoning before the answer line. At 200 the
    # model was still mid-CoT when it hit the cap, so extraction returned None
    # and every arm scored ~0 — a meaningless comparison. 512 lets most items
    # actually reach "Answer: <letter>".
    max_new_tokens: int = 512

    # When an expert switch happens, the keys/values already in the cache were
    # computed by the *previous* expert, so the new expert attends over a
    # history it did not encode. Setting this rebuilds the cache under the new
    # weights instead — exact, but it re-pays prefill on every switch, which
    # is precisely the cost the unified design exists to avoid. Off by default;
    # the eval turns it on for one arm to price the shortcut.
    recompute_on_switch: bool = False

    def to_dict(self):
        return asdict(self)


DEFAULT = RouterConfig()

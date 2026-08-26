"""Tests that need no base model — detector behaviour and data normalization."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from router.config import RouterConfig
from router.script_detect import detect_language, language_shifted

CASES = [
    ("中国的首都是北京，人口很多", "zh"),
    ("日本の首都は東京です", "ja"),
    # Majority-Han Japanese: the Kana rule must beat the dominant-script rule.
    ("東京は日本の首都で、人口が多い", "ja"),
    ("대한민국의 수도는 서울입니다", "ko"),
    ("العاصمة هي الرياض والمدينة كبيرة", "ar"),
    ("भारत की राजधानी दिल्ली है", "hi"),
    ("मी मराठी बोलतो आणि लिहितो चांगले आहे", "mr"),
    ("হে একটি ভাল উদাহরণ এবং আমি জানি না", "bn"),
    ("இந்தியாவின் தலைநகரம் எது", "ta"),
    ("ಕರ್ನಾಟಕದ ರಾಜಧಾನಿ ಯಾವುದು", "kn"),
    ("The capital of India is New Delhi and it is a very large city", "en"),
    ("Die Hauptstadt von Deutschland ist Berlin und sie ist sehr schoen", "de"),
    ("La capitale de la France est Paris et elle est tres belle ville", "fr"),
]

UNDECIDED = ["", "   ", "42", "3.14159", "A. B. C.", "short"]


def test_detection():
    fails = []
    for text, expected in CASES:
        got = detect_language(text)
        if got != expected:
            fails.append(f"  {text[:40]!r} -> {got} (expected {expected})")
    assert not fails, "detection failures:\n" + "\n".join(fails)


def test_undecided_returns_none():
    """Too little evidence must yield None so the router holds the current
    expert rather than defaulting to one."""
    for text in UNDECIDED:
        assert detect_language(text) is None, f"{text!r} should be undecided"


def test_latin_needs_more_evidence_than_unique_script():
    """A short Latin fragment is undecidable, but the same length of a
    script-unique language is decidable."""
    assert detect_language("The capital") is None
    assert detect_language("中国的首都是北京") == "zh"


def test_no_shift_when_language_unchanged():
    text = "The capital of India is New Delhi and it is a very large city"
    assert language_shifted("en", text) is None


def test_shift_detected_on_change():
    text = "The capital of India is New Delhi and it is a very large city"
    assert language_shifted("hi", text) == "en"


def test_config_threshold_is_respected():
    """Raising the Latin confidence floor must make borderline text undecided."""
    strict = RouterConfig(latin_min_conf=1.01)  # unreachable by construction
    text = "The capital of India is New Delhi and it is a very large city"
    assert language_shifted("hi", text, strict) is None


def test_answer_extraction():
    from router.generation import extract_answer
    assert extract_answer("blah\nAnswer: C") == "C"
    assert extract_answer("Answer: (B)") == "B"
    assert extract_answer("answer - d") == "D"
    assert extract_answer("no answer here") is None


def test_benchmark_normalization():
    from router.benchmarks import _norm_milu, _norm_mmmlu
    milu = _norm_milu(
        {"question": "q", "option1": "a", "option2": "b", "option3": "c",
         "option4": "d", "target": "option3", "subject": "s"},
        "Hindi", "hi")
    assert milu["answer"] == "C" and milu["options"][2] == "c"

    mmmlu = _norm_mmmlu(
        {"Question": "q", "A": "a", "B": "b", "C": "c", "D": "d",
         "Answer": "B", "Subject": "s"},
        "DE_DE", "de")
    assert mmmlu["answer"] == "B" and mmmlu["options"][1] == "b"


def test_sweep_excludes_configs_that_disable_routing():
    """A window shorter than the Latin evidence floor makes Latin
    undetectable. Since the model reasons in English, that silently collapses
    routing to ~0 switches and the whole config scores as baseline — which a
    first sweep then mistook for a plateau."""
    from optimize import candidates, is_viable
    from router.config import RouterConfig

    ok, why = is_viable(RouterConfig(window_chars=32, latin_min_chars=40))
    assert not ok and "undetectable" in why

    assert is_viable(RouterConfig(window_chars=64, latin_min_chars=40))[0]

    good, dropped = candidates()
    assert dropped, "grid should exclude at least one unviable config"
    for cfg in good:
        assert cfg.window_chars >= cfg.latin_min_chars


def test_latin_undetectable_below_floor():
    """The underlying reason the above configs are dead: a Latin window
    shorter than the floor yields no verdict at all."""
    text = "The capital of India is New Delhi"      # < 40 Latin chars
    assert detect_language(text, latin_min_chars=40) is None
    assert detect_language(text, latin_min_chars=10) == "en"


def test_milu_numeric_target_form():
    """The fixture uses '2'; the real dataset uses 'option2'. Both must work."""
    from router.benchmarks import _norm_milu
    row = {"question": "q", "option1": "a", "option2": "b", "option3": "c",
           "option4": "d", "target": "2", "subject": "s"}
    assert _norm_milu(row, "Hindi", "hi")["answer"] == "B"

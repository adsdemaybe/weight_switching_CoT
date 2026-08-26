"""Language-shift signal detection over a rolling window of generated text.

Design: Unicode script is the primary signal — it is deterministic, costs no
model call, and for most of the target set it is decisive on its own. Two
script families need extra work:

  * CJK: Japanese mixes Han with Kana, and a Japanese sentence is often
    *majority* Han, so plain "dominant script wins" would misroute it to
    Chinese. Presence of Kana (or Hangul) is therefore treated as decisive
    regardless of how much Han sits alongside it.
  * Latin: en/de/fr/es/it/pt all occupy the same block, so script carries no
    information. Those fall through to a statistical classifier (langid),
    which needs a longer window and a confidence floor to be usable.

Anything the detector is not confident about returns None, which callers read
as "keep the current expert" rather than "switch to a default".
"""
from langid.langid import LanguageIdentifier, model as _langid_model

# --- language registry -------------------------------------------------
# Indic set comes from MILU (ai4bharat), the rest from MMMLU (openai).
# `name` is the dataset config key, `code` is what adapter files are keyed on.

INDIC_LANGS = {
    "Bengali": "bn", "Gujarati": "gu", "Hindi": "hi", "Kannada": "kn",
    "Malayalam": "ml", "Marathi": "mr", "Odia": "or", "Punjabi": "pa",
    "Tamil": "ta", "Telugu": "te", "English": "en",
}

# MMMLU config name -> short code
MMMLU_LANGS = {
    "ZH_CN": "zh", "JA_JP": "ja", "KO_KR": "ko", "DE_DE": "de",
    "FR_FR": "fr", "ES_LA": "es", "IT_IT": "it", "PT_BR": "pt",
    "AR_XY": "ar",
}

LANG_CODE = dict(INDIC_LANGS)

# Languages that share the Latin block and must be separated statistically.
LATIN_LANGS = ["en", "de", "fr", "es", "it", "pt"]
# Languages that share the Devanagari block.
DEVANAGARI_LANGS = ["hi", "mr"]

# (start, end) inclusive Unicode codepoint ranges per script block.
SCRIPT_RANGES = {
    "bn": [(0x0980, 0x09FF)],           # Bengali
    "gu": [(0x0A80, 0x0AFF)],           # Gujarati
    "kn": [(0x0C80, 0x0CFF)],           # Kannada
    "ml": [(0x0D00, 0x0D7F)],           # Malayalam
    "or": [(0x0B00, 0x0B7F)],           # Odia
    "pa": [(0x0A00, 0x0A7F)],           # Gurmukhi / Punjabi
    "ta": [(0x0B80, 0x0BFF)],           # Tamil
    "te": [(0x0C00, 0x0C7F)],           # Telugu
    "ar": [(0x0600, 0x06FF), (0x0750, 0x077F)],   # Arabic
    "ru": [(0x0400, 0x04FF)],           # Cyrillic
    "devanagari": [(0x0900, 0x097F)],   # Hindi + Marathi
    "han": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],  # Chinese, also used in ja/ko
    "kana": [(0x3040, 0x309F), (0x30A0, 0x30FF)],  # Hiragana + Katakana -> ja
    "hangul": [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],  # ko
    "latin": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
}

# Independent classifiers so the two ambiguous families don't clobber each
# other's language restriction (langid's restriction is global per-instance).
_latin_id = LanguageIdentifier.from_modelstring(_langid_model, norm_probs=True)
_latin_id.set_languages(LATIN_LANGS)

_deva_id = LanguageIdentifier.from_modelstring(_langid_model, norm_probs=True)
_deva_id.set_languages(DEVANAGARI_LANGS)

# langid separates hi/mr poorly (it inverts the pair on short text), so that
# one case uses closed-class function words instead — copulas, conjunctions,
# negators and genitive markers, which differ sharply between the two and are
# frequent enough to show up in a short window.
HINDI_CUES = {
    "है", "हैं", "और", "मैं", "नहीं", "का", "की", "के", "यह", "वह",
    "को", "से", "में", "कि", "था", "थे", "थी", "हुआ", "करना", "गया",
}
MARATHI_CUES = {
    "आहे", "आहेत", "आणि", "मी", "नाही", "चा", "ची", "चे", "हे", "ते",
    "मला", "त्या", "होते", "केले", "या", "व", "असे", "काय", "पण",
}
# U+0933 (ळ) is standard in Marathi and effectively absent from Hindi.
MARATHI_CHAR = "ळ"

_DEVA_PUNCT = "।,.?!\"'()[]{}:;०१२३४५६७८९"


def _disambiguate_devanagari(text):
    tokens = [t.strip(_DEVA_PUNCT) for t in text.split()]
    hi_score = sum(1 for t in tokens if t in HINDI_CUES)
    mr_score = sum(1 for t in tokens if t in MARATHI_CUES)
    if MARATHI_CHAR in text:
        mr_score += 2
    if hi_score != mr_score:
        return "hi" if hi_score > mr_score else "mr"
    # No lexical evidence either way — fall back to the statistical model, and
    # if that is also unsure, report undecided rather than guessing.
    lang, conf = _deva_id.classify(text)
    if conf >= 0.95 and lang in DEVANAGARI_LANGS:
        return lang
    return None

# Latin needs far more evidence than an alphabet-unique script does.
LATIN_MIN_CHARS = 40
LATIN_MIN_CONF = 0.90


def _char_script(ch):
    cp = ord(ch)
    for lang, ranges in SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return lang
    return None


def script_counts(text):
    counts = {}
    for ch in text:
        s = _char_script(ch)
        if s is not None:
            counts[s] = counts.get(s, 0) + 1
    return counts


def detect_language(text, min_chars=4, latin_min_chars=LATIN_MIN_CHARS,
                    latin_min_conf=LATIN_MIN_CONF):
    """Return a short language code for `text`, or None if undecidable.

    None means "not enough evidence" — callers keep the active expert instead
    of falling back to a default, so the router doesn't thrash on whitespace,
    digits, or a few shared punctuation characters.
    """
    counts = script_counts(text)
    total = sum(counts.values())
    if total < min_chars:
        return None

    # CJK disambiguation runs before the generic dominant-script rule:
    # Kana and Hangul are exclusive to Japanese and Korean respectively, and
    # their presence outweighs any amount of co-occurring Han.
    if counts.get("kana", 0) > 0:
        return "ja"
    if counts.get("hangul", 0) > 0:
        return "ko"

    dominant = max(counts, key=counts.get)

    if dominant == "han":
        return "zh"

    if dominant == "devanagari":
        return _disambiguate_devanagari(text)

    if dominant == "latin":
        # Script carries no signal here; require a longer window and a
        # confident classifier verdict, else report "undecided".
        if counts["latin"] < latin_min_chars:
            return None
        lang, conf = _latin_id.classify(text)
        if conf < latin_min_conf:
            return None
        return lang

    return dominant


def language_shifted(prev_lang, window_text, cfg=None):
    """Router decision: return the new language code if the active-expert
    language should change, else None."""
    from .config import DEFAULT
    cfg = cfg or DEFAULT
    new_lang = detect_language(
        window_text,
        min_chars=cfg.min_chars,
        latin_min_chars=cfg.latin_min_chars,
        latin_min_conf=cfg.latin_min_conf,
    )
    if new_lang is None or new_lang == prev_lang:
        return None
    return new_lang

#!/usr/bin/env python3
"""
NER using LLM with Prompt Engineering
Reads spoken-text transcriptions from Reference.jsonl and uses an LLM to
detect named entities. Outputs predictions to Hypothesis.jsonl.

Usage:
    # LLM-based (requires OPENAI_API_KEY env variable):
    python ner_llm.py

    # Rule-based fallback (no API key needed):
    python ner_llm.py --use-rules

    # Specify custom input/output files:
    python ner_llm.py --input Reference.jsonl --output Hypothesis.jsonl
"""

import json
import re
import os
import sys
import time
import argparse
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Prompt templates for LLM-based NER
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Named Entity Recognition (NER) expert specialising in \
spoken text from Singapore. The texts are speech recognition outputs and may contain \
spoken-form numbers, emails, and identifiers.

Identify every named entity in the user's text and classify it as one of:
  PERSON       - Individual names (e.g. "Ram", "Boon", "Beatrice", "#tan#")
  ORG          - Organisations, companies, brands (e.g. "POSB", "Gmail", "VISA")
  GPE          - Geopolitical entities: countries, cities (e.g. "Singapore", "Japan")
  LOC          - Non-GPE locations (e.g. "Little India", "Botanic Gardens")
  DATE         - Date expressions (e.g. "yesterday", "second september", "friday")
  TIME         - Time expressions (e.g. "afternoon", "two P_M", "eleven A_M")
  CARDINAL     - Cardinal numbers as words (e.g. "three", "seven pax", "forty five")
  MONEY        - Monetary amounts (e.g. "ten K", "fifty dollar sing")
  EMAIL        - Email addresses – standard or spelled-out spoken form
                 (e.g. "user@gmail.com", "t x 1 r z a four seven at yahoo dot com")
  PHONE        - Phone numbers – digits, hyphenated, or spelled-out spoken form
                 (e.g. "8372-1289", "+65 65841894", "nine one three four seven three zero five")
  CAR_PLATE    - Vehicle registration plates (e.g. "SY8792H", "SBC 770 X", "SDP222 U")
  NRIC         - Singapore NRIC / IC numbers (e.g. "S0768363", "T6404940I", "282I")
  PASSPORT_NUM - Passport numbers (e.g. "K9644461Z", "K8608363 G")
  CREDIT_CARD  - 16-digit credit card numbers, possibly grouped (e.g. "4273 5141 9211 0013")
  BANK_ACCOUNT - Bank account numbers (e.g. "236-38502-7", "349 572108 1")
  ACCOUNT_NUM  - Other account/reference numbers (e.g. "772 09111 0", "764-917253-8")

Rules:
- Return ONLY a JSON object with a single key "entities".
- Each entity is [start_char, end_char, TYPE] using 0-based indices on the original text.
- end_char is the index of the character AFTER the last character (Python slice convention).
- Include ALL entity occurrences; omit nothing.
- Do NOT overlap entities.
- If no entities exist return {"entities": []}.
"""

FEW_SHOT_EXAMPLES = [
    {
        "text": "yes, Forstest car is SY8792H",
        "entities": [[21, 28, "CAR_PLATE"]],
    },
    {
        "text": "Boon contact number is 8372-1289 but he rarely use this number",
        "entities": [[0, 4, "PERSON"], [23, 32, "PHONE"]],
    },
    {
        "text": (
            "0321-9329-7066-0001 is my brother Tan's card, please call him to "
            "verify and his IC number S0768363 and email is boonchang90@gmail.com"
        ),
        "entities": [
            [0, 19, "CREDIT_CARD"],
            [34, 37, "PERSON"],
            [90, 98, "NRIC"],
            [112, 133, "EMAIL"],
        ],
    },
    {
        "text": "t a n b b at gmail dot com is BB's Gmail, do you know ?",
        "entities": [[0, 26, "EMAIL"], [35, 40, "ORG"]],
    },
    {
        "text": (
            "he do not really know the passport number K8608363 G "
            "my passport and my account number is 772 09111 0"
        ),
        "entities": [[42, 52, "PASSPORT_NUM"], [90, 101, "ACCOUNT_NUM"]],
    },
]


def build_messages(text: str) -> List[dict]:
    """Build the OpenAI messages list for a single text."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["text"]})
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps({"entities": ex["entities"]}),
            }
        )
    messages.append({"role": "user", "content": text})
    return messages


# ---------------------------------------------------------------------------
# LLM inference
# ---------------------------------------------------------------------------


def predict_with_llm(
    text: str, client, model: str = "gpt-4o-mini", retries: int = 3
) -> List[List]:
    """Call OpenAI chat completion and parse entity list."""
    messages = build_messages(text)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            parsed = json.loads(raw)
            entities = parsed.get("entities", [])
            # Validate and clean: keep only well-formed triples
            valid = []
            for e in entities:
                if (
                    isinstance(e, (list, tuple))
                    and len(e) == 3
                    and isinstance(e[0], int)
                    and isinstance(e[1], int)
                    and isinstance(e[2], str)
                    and 0 <= e[0] < e[1] <= len(text)
                ):
                    valid.append([e[0], e[1], e[2]])
            return valid
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2**attempt)
            else:
                print(f"  [WARN] LLM failed for text: {text[:60]}... — {exc}")
                return []
    return []


# ---------------------------------------------------------------------------
# Rule-based NER (used when no API key is available)
# ---------------------------------------------------------------------------

# Sliding window sizes for context-based entity detection
_NRIC_CTX_WINDOW = 60        # chars before a partial NRIC to check for NRIC context
_SPOKEN_NRIC_CTX_WINDOW = 80 # chars before spoken NRIC partial
_BANK_CTX_WINDOW = 80        # chars before a bank number to check context
_EMAIL_LOCAL_WINDOW = 15     # chars after "may" to check if it is used as a date
_CARDINAL_BEFORE = 10        # chars before a cardinal word for qty context
_CARDINAL_AFTER = 25         # chars after a cardinal word for qty context

# Known organisations
_ORGS = {
    "gmail", "yahoo", "hotmail", "outlook", "posb", "dbs", "ocbc", "uob",
    "visa", "master", "mastercard", "american express", "singtel", "nea",
    "abc", "a b c", "abb", "a b b", "nus", "ntu", "ntuc", "sp", "altair",
    "n_t_u_c", "cold storage",
}

# Countries / cities (GPE)
_GPES = {
    "singapore", "taiwan", "korea", "japan", "osaka", "seoul", "busan",
    "jeju", "hokkaido", "india", "china", "australia", "malaysia",
}

# Named locations (LOC)
_LOCS = {
    "little india", "botanic gardens", "chinatown", "orchard road",
    "sentosa", "changi",
}

# Day / month words for DATE detection
_DAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
}
_MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
}
_DATE_WORDS = {
    "yesterday", "today", "tomorrow", "tonight",
    "last month", "next month", "this month",
    "last week", "next week", "this week",
}
_ORDINALS = {
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
    "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth",
    "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth",
    "nineteenth", "twentieth", "twenty-first", "twenty first",
    "thirty-first", "thirty first",
}

# Time words
_TIME_WORDS = {"morning", "afternoon", "evening", "night", "midnight", "noon"}

# Cardinal number words
_CARDINALS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand",
}


def _find_all(pattern: str, text: str, flags: int = 0) -> List[Tuple[int, int, str]]:
    """Return list of (start, end, matched_string) for all non-overlapping matches."""
    results = []
    for m in re.finditer(pattern, text, flags):
        results.append((m.start(), m.end(), m.group()))
    return results


def _overlaps(start: int, end: int, spans: List[Tuple[int, int]]) -> bool:
    for s, e in spans:
        if start < e and end > s:
            return True
    return False


def _add(
    entities: List[List],
    taken: List[Tuple[int, int]],
    start: int,
    end: int,
    label: str,
) -> None:
    """Add entity if it does not overlap any existing span."""
    if not _overlaps(start, end, taken):
        entities.append([start, end, label])
        taken.append((start, end))


def _extract_standard_email(text: str) -> List[Tuple[int, int]]:
    pat = r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    return [(m.start(), m.end()) for m in re.finditer(pat, text)]


def _extract_spoken_email(text: str) -> List[Tuple[int, int]]:
    """
    Spoken-form email: "<local-part> at <domain> dot <tld> [dot <tld2>]"
    Handles:
      - Single-letter spaced: "r e m y at outlook dot sg"
      - Name-based:          "julia tan at G mail dot com"
      - Compound domain:     "at G mail dot com" (two-word domain)
    """
    _EMAIL_STOP = {
        "email", "mail", "from", "is", "are", "was", "were", "to",
        "please", "can", "have", "just", "may", "ok", "okay", "sure",
        "the", "an", "of", "in", "on", "with", "that", "this",
        "my", "your", "his", "her", "our", "their", "and", "or", "but",
        "not", "no", "yes", "hi", "hello", "so", "then", "check",
        "verify", "send", "note", "correct", "me", "it", "we", "they",
        "you", "he", "she", "am", "be", "been", "has", "had",
        "contact", "reach", "via", "at",  # "at" must stop (avoid double-at)
        # Common email domain words should stop the scan — they are not part of local part
        "gmail", "yahoo", "hotmail", "outlook", "singtel",
    }

    # Match "at <domain> dot <tld>" where domain may be 1 or 2 words (e.g. "G mail")
    at_pat = re.compile(
        r"\bat\s+([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s+dot\s+([a-zA-Z]+)"
        r"(?:\s+dot\s+([a-zA-Z]+))?",
        re.IGNORECASE,
    )

    results = []
    for m in at_pat.finditer(text):
        at_pos = m.start()
        end_pos = m.end()

        prefix = text[:at_pos].rstrip()
        if not prefix:
            continue
        tokens = prefix.split()
        local_tokens: List[str] = []
        for tok in reversed(tokens):
            tok_clean = tok.lower().strip(".,!?;:#")
            if len(tok_clean) == 1:
                # Single-character tokens are always valid email parts
                local_tokens.insert(0, tok)
                continue
            if tok_clean in _EMAIL_STOP:
                break
            if not tok_clean.replace("_", "").replace("-", "").replace(".", "").isalnum():
                break
            local_tokens.insert(0, tok)

        if not local_tokens:
            continue

        local_str = " ".join(local_tokens)
        start = prefix.rfind(local_str)
        if start == -1:
            continue

        results.append((start, end_pos))

    return results


def _extract_phone(text: str) -> List[Tuple[int, int]]:
    """Detect phone numbers: formatted digits and spoken-form."""
    results = []
    # +65 xxxxxxxx
    for m in re.finditer(r"\+\d{2}\s*\d{7,8}", text):
        results.append((m.start(), m.end()))
    # xxxx-xxxx (Singapore 8-digit with hyphen)
    for m in re.finditer(r"\b\d{4}[-]\d{4,5}\b", text):
        results.append((m.start(), m.end()))
    # 8-digit run: must be standalone
    for m in re.finditer(r"\b\d{8}\b", text):
        results.append((m.start(), m.end()))
    # Spoken phone: sequence of digit-words ~8 in a row
    spoken_digits = (
        r"\b(?:(?:zero|one|two|three|four|five|six|seven|eight|nine)\s+){6,9}"
        r"(?:zero|one|two|three|four|five|six|seven|eight|nine)\b"
    )
    for m in re.finditer(spoken_digits, text, re.IGNORECASE):
        results.append((m.start(), m.end()))
    return results


def _extract_car_plate(text: str) -> List[Tuple[int, int]]:
    """Singapore car plate: 1-3 letters + 1-4 digits + 1 letter, optional spaces."""
    results = []

    def _is_plate(matched: str) -> bool:
        digits_in = re.sub(r"[^0-9]", "", matched)
        letters_in = re.sub(r"[^A-Z]", "", matched)
        # Exclude NRIC/passport: 1-letter prefix + 7 digits + 1 letter (no space variant)
        if len(digits_in) == 7 and len(letters_in) == 2:
            return False
        return len(digits_in) >= 1 and len(letters_in) >= 2

    # Compact (no spaces): SY8792H, SBC700X, SHA300J, SB133E
    for m in re.finditer(r"\b[A-Z]{1,3}\d{1,4}[A-Z]\b", text):
        if _is_plate(m.group()):
            results.append((m.start(), m.end()))

    # Trailing-space variant: SB133 E, SDP222 U, SY8792 H, SNQ4567 A
    for m in re.finditer(r"\b[A-Z]{1,3}\d{1,4}\s[A-Z]\b", text):
        matched = m.group()
        digits_in = re.sub(r"[^0-9]", "", matched)
        letters_in = re.sub(r"[^A-Z]", "", matched)
        # Exclude NRIC/passport with space: 1-letter + 7-digits + space + letter
        if len(digits_in) == 7 and len(letters_in) == 2:
            continue
        if len(digits_in) >= 1 and len(letters_in) >= 2:
            results.append((m.start(), m.end()))

    # Leading-space variant: SBC 770 X, SMA 476G
    for m in re.finditer(r"\b[A-Z]{1,3}\s\d{1,4}\s?[A-Z]\b", text):
        if _is_plate(m.group()):
            results.append((m.start(), m.end()))

    return results


def _extract_nric(text: str) -> List[Tuple[int, int]]:
    """
    Singapore NRIC: [STFG] + 7 digits + letter, optional space before last letter.
    Also partial: last-4-digits-form like "zero zero seven A".
    """
    results = []
    # Full NRIC
    for m in re.finditer(r"\b[STFG]\d{7}\s?[A-Z]\b", text):
        results.append((m.start(), m.end()))
    # Partial NRIC "4 chars ending in uppercase letter" mentioned after IC/NRIC context
    # e.g. "282I", "867H", "565M", "eight one one G"
    partial_digit = r"\b\d{3,4}[A-Z]\b"
    for m in re.finditer(partial_digit, text):
        # Check preceding context mentions IC/NRIC
        ctx = text[max(0, m.start() - _NRIC_CTX_WINDOW) : m.start()].lower()
        if re.search(r"\b(nric|ic|i_c|n_r_i_c|digit)\b", ctx):
            results.append((m.start(), m.end()))
    # Spoken partial: "eight one one G", "zero zero seven A"
    spoken_partial = (
        r"\b(?:(?:zero|one|two|three|four|five|six|seven|eight|nine)\s+){2,3}"
        r"[A-Z]\b"
    )
    for m in re.finditer(spoken_partial, text):
        ctx = text[max(0, m.start() - _SPOKEN_NRIC_CTX_WINDOW) : m.start()].lower()
        if re.search(r"\b(nric|ic|i_c|n_r_i_c|digit)\b", ctx):
            results.append((m.start(), m.end()))
    return results


def _extract_passport(text: str) -> List[Tuple[int, int]]:
    """Passport number: letter + 7 digits + letter, optional space or space-separated."""
    results = []
    # Compact: K9644461Z
    for m in re.finditer(r"\b[A-Z]\d{7}[A-Z]\b", text):
        results.append((m.start(), m.end()))
    # With space before last letter: K8608363 G
    for m in re.finditer(r"\b[A-Z]\d{7}\s[A-Z]\b", text):
        results.append((m.start(), m.end()))
    # Space-separated: K 4213904 U
    for m in re.finditer(r"\b[A-Z]\s\d{7}\s[A-Z]\b", text):
        results.append((m.start(), m.end()))
    return results


def _extract_credit_card(text: str) -> List[Tuple[int, int]]:
    """16-digit credit card in 4 groups of 4."""
    pat = r"\b\d{4}[\s\-]\d{4}[\s\-]\d{4}[\s\-]\d{4}\b"
    return [(m.start(), m.end()) for m in re.finditer(pat, text)]


def _extract_bank_account(text: str, taken: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Bank account / account numbers. Labelled BANK_ACCOUNT.
    Formats:
      - Hyphenated 3-{4..8}-{1..2}:  236-38502-7, 888-20011-0
      - Space-separated 3-3-3-1:     921 372 579 2  (UOB 4-part)
      - Space-separated 3-{5..8}-1:  349 572108 1 / 111 385092 7
    Larger spans take priority over sub-spans (e.g. 4-part > 3-part sub-match).
    """
    candidates = []
    # 4-part space-separated: 3-3-3-1 (longest; check first)
    for m in re.finditer(r"\b\d{3}\s\d{3}\s\d{3}\s\d\b", text):
        candidates.append((m.start(), m.end()))
    # Hyphenated 3-{4..8}-{1..2}
    for m in re.finditer(r"\b\d{3}-\d{4,8}-\d{1,2}\b", text):
        candidates.append((m.start(), m.end()))
    # Space-separated 3-{5..8}-{1..2}
    for m in re.finditer(r"\b\d{3}\s\d{5,8}\s\d{1,2}\b", text):
        candidates.append((m.start(), m.end()))

    # Deduplicate: prefer longer spans; skip if a longer overlapping span is already added
    # Sort by length descending so we process longer spans first
    candidates.sort(key=lambda x: -(x[1] - x[0]))

    local_taken: List[Tuple[int, int]] = []
    results = []
    for s, e in candidates:
        if _overlaps(s, e, taken) or _overlaps(s, e, local_taken):
            continue
        results.append((s, e, True))
        local_taken.append((s, e))
    return results


def _find_keyword_spans(
    text: str, keywords: set, label: str, taken: List[Tuple[int, int]]
) -> List[List]:
    """Find longest-first keyword matches (case-insensitive)."""
    entities = []
    lower = text.lower()
    for kw in sorted(keywords, key=len, reverse=True):
        start = 0
        while True:
            idx = lower.find(kw, start)
            if idx == -1:
                break
            end = idx + len(kw)
            # Word-boundary check
            before_ok = idx == 0 or not text[idx - 1].isalnum()
            after_ok = end == len(text) or not text[end].isalnum()
            if before_ok and after_ok and not _overlaps(idx, end, taken):
                entities.append([idx, end, label])
                taken.append((idx, end))
            start = idx + 1
    return entities


def _extract_date_expressions(text: str, taken: List[Tuple[int, int]]) -> List[List]:
    """Find date-related spans."""
    entities = []
    lower = text.lower()

    # 1. Month names (standalone) — reference annotates months individually, not combined
    for tok in sorted(_MONTHS, key=len, reverse=True):
        # Skip "may" when used as modal verb
        if tok == "may":
            for m in re.finditer(r"\bmay\b", lower):
                if _overlaps(m.start(), m.end(), taken):
                    continue
                after = lower[m.end(): m.end() + _EMAIL_LOCAL_WINDOW]
                if not re.match(r"\s+\d|\s+(next|last|this|nineteen|twenty|two|three)", after):
                    continue
                entities.append([m.start(), m.end(), "DATE"])
                taken.append((m.start(), m.end()))
            continue
        start = 0
        while True:
            idx = lower.find(tok, start)
            if idx == -1:
                break
            end = idx + len(tok)
            before_ok = idx == 0 or not text[idx - 1].isalnum()
            after_ok = end == len(text) or not text[end].isalnum()
            if before_ok and after_ok and not _overlaps(idx, end, taken):
                entities.append([idx, end, "DATE"])
                taken.append((idx, end))
            start = idx + 1

    # 2. Day names
    for tok in sorted(_DAYS, key=len, reverse=True):
        start = 0
        while True:
            idx = lower.find(tok, start)
            if idx == -1:
                break
            end = idx + len(tok)
            before_ok = idx == 0 or not text[idx - 1].isalnum()
            after_ok = end == len(text) or not text[end].isalnum()
            if before_ok and after_ok and not _overlaps(idx, end, taken):
                entities.append([idx, end, "DATE"])
                taken.append((idx, end))
            start = idx + 1

    # 3. Simple date words (no "last/this month" — reference doesn't annotate these)
    for tok in {"yesterday", "today", "tomorrow", "tonight"}:
        start = 0
        while True:
            idx = lower.find(tok, start)
            if idx == -1:
                break
            end = idx + len(tok)
            before_ok = idx == 0 or not text[idx - 1].isalnum()
            after_ok = end == len(text) or not text[end].isalnum()
            if before_ok and after_ok and not _overlaps(idx, end, taken):
                entities.append([idx, end, "DATE"])
                taken.append((idx, end))
            start = idx + 1

    # 4. Day-of-week with next/last (but NOT "this/last month" — ref doesn't annotate those)
    for m in re.finditer(
        r"\b(next|last)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        lower,
    ):
        if not _overlaps(m.start(), m.end(), taken):
            entities.append([m.start(), m.end(), "DATE"])
            taken.append((m.start(), m.end()))

    return entities


def _extract_time_expressions(text: str, taken: List[Tuple[int, int]]) -> List[List]:
    entities = []
    lower = text.lower()

    # Number word + A_M / P_M  (e.g. "two P_M", "eleven A_M")
    # But NOT "one am I" (am as a verb) — ensure not followed by "I", "we", etc.
    num_words = (
        "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
        "|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty"
    )
    for m in re.finditer(
        rf"\b({num_words})\s+[ap]_?m\b", lower
    ):
        # Check what follows — if it's "i", "we", "you", etc., it's a verb
        after = lower[m.end():m.end() + 5].strip()
        if re.match(r"^(i|we|you|they|he|she)\b", after):
            continue
        if not _overlaps(m.start(), m.end(), taken):
            entities.append([m.start(), m.end(), "TIME"])
            taken.append((m.start(), m.end()))

    # Single time-of-day words — but only when not followed by "is", "from" etc.
    for tok in _TIME_WORDS:
        idx = lower.find(tok)
        while idx != -1:
            end = idx + len(tok)
            before_ok = idx == 0 or not text[idx - 1].isalnum()
            after_ok = end == len(text) or not text[end].isalnum()
            if before_ok and after_ok and not _overlaps(idx, end, taken):
                entities.append([idx, end, "TIME"])
                taken.append((idx, end))
            idx = lower.find(tok, idx + 1)

    return entities


def _extract_cardinals(text: str, taken: List[Tuple[int, int]]) -> List[List]:
    """Extract cardinal number words in quantitative contexts."""
    entities = []
    lower = text.lower()

    card_word = "|".join(sorted(_CARDINALS, key=len, reverse=True))

    # "about/around [cardinal]" pattern (e.g. "about two", "around two-thirty")
    for m in re.finditer(
        rf"\b(?:about|around)\s+(?:{card_word})(?:[-\s]+(?:{card_word}))*\b", lower
    ):
        if not _overlaps(m.start(), m.end(), taken):
            entities.append([m.start(), m.end(), "CARDINAL"])
            taken.append((m.start(), m.end()))

    # Multi-word runs (e.g. "forty five", "three three", "two-thirty")
    _year_starters = {"nineteen", "twenty"}
    multi_pat = rf"\b(?:{card_word})(?:[-\s]+(?:{card_word}))+\b"
    for m in re.finditer(multi_pat, lower):
        span_words = re.split(r"[\s\-]+", m.group())
        if span_words[0] in _year_starters and len(span_words) >= 3:
            continue
        if not _overlaps(m.start(), m.end(), taken):
            entities.append([m.start(), m.end(), "CARDINAL"])
            taken.append((m.start(), m.end()))

    # Single cardinal words — only in quantitative context
    qty_context = re.compile(
        r"\b(pax|person|people|guests?|members?|rooms?|nights?|beds?|"
        r"master|twin|single|double|star|"
        r"suitcases?|bags?|items?|types?|cards?|blocks?|view|views?"
        r"|cans?|percent|%|pieces?|banking|children|child|kids?)\b",
        re.IGNORECASE,
    )
    for m in re.finditer(rf"\b({card_word})\b", lower):
        if _overlaps(m.start(), m.end(), taken):
            continue
        window = lower[max(0, m.start() - _CARDINAL_BEFORE): m.end() + _CARDINAL_AFTER]
        if qty_context.search(window):
            entities.append([m.start(), m.end(), "CARDINAL"])
            taken.append((m.start(), m.end()))

    return entities


def _extract_money(text: str, taken: List[Tuple[int, int]]) -> List[List]:
    entities = []
    lower = text.lower()
    # "ten K", "fifty dollar", "X dollar sing", "$X"
    patterns = [
        r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty"
        r"|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)"
        r"\s+(?:k|thousand|dollar|dollars|sgd|cents?|sing)\b",
        r"\$[\d,]+(?:\.\d{2})?",
    ]
    for pat in patterns:
        for m in re.finditer(pat, lower):
            if not _overlaps(m.start(), m.end(), taken):
                entities.append([m.start(), m.end(), "MONEY"])
                taken.append((m.start(), m.end()))
    return entities


def _extract_persons(text: str, taken: List[Tuple[int, int]]) -> List[List]:
    """
    Extract person names using several heuristics:
      - After title words: mister/miss/mr/mrs/ms/madam + name(s)
      - Context phrases: "this is X", "I'm X", "hi X", "my name is X"
      - Hashtag names: #name# (or leading-word #name#)
      - Capitalised proper nouns (not sentence-start, not known ORG/GPE/LOC)
      - Possessive: "X's" at sentence start
    """
    entities = []
    lower = text.lower()

    # Words that are definitely NOT person names (common English words)
    _NOT_NAME = {
        "the", "a", "an", "is", "are", "was", "here", "there", "my", "your",
        "his", "her", "our", "their", "this", "that", "these", "those",
        "it", "its", "we", "they", "he", "she", "i", "you", "me",
        "please", "ok", "okay", "sure", "yes", "no", "not",
        "nric", "ic", "email", "phone", "contact", "card", "bank",
        "account", "number", "passport", "car", "plate",
        "hotel", "booking", "insurance", "travel", "agency", "tour", "trip",
        "office", "home", "visa", "master", "new", "old", "good", "day",
        "time", "need", "lost", "got", "know", "different", "correct",
        "planning", "travelling", "traveling", "calling", "going", "doing",
        "directly", "another", "also", "when", "then", "thank", "thanks",
        "inside", "outside", "before", "after", "with", "without", "for",
        "from", "about", "into", "onto", "upon", "already", "because",
        "right", "left", "while", "just", "still", "only", "even",
        "however", "therefore", "furthermore", "actually", "basically",
        "definitely", "really", "very", "quite", "more", "most", "less",
        "like", "such", "every", "each", "any", "some", "all",
        "first", "second", "third", "last", "next", "same", "other",
        # common verbs that look capitalised at sentence start:
        "heres", "need", "found", "given", "noted", "if", "help",
        "take", "make", "find", "keep", "let", "put",
    } | _ORGS | _GPES | _LOCS

    # 1. Title + name(s):
    #    - "mister" / "miss" -> reference wants just the NAME, not title
    #    - "Mr." / "Mr" / "madam" -> reference includes the title in the span
    for m in re.finditer(
        r"\b(mister|miss|mr\.?|mrs\.?|ms\.?|madam)\s+([A-Za-z]+)(?:\s+([A-Za-z]+))?",
        text,
        re.IGNORECASE,
    ):
        title = m.group(1).lower().rstrip(".")
        name1 = m.group(2)
        name2 = m.group(3)
        if name2 and name2.lower() in _NOT_NAME:
            name2 = None

        if title in ("mister", "miss"):
            # Reference wants just the name without the title
            s = m.start(2)
            e = (m.start(3) + len(name2)) if name2 else m.end(2)
        else:
            # Mr. / madam / mrs. / ms. -> include title
            s = m.start()
            e = (m.end(3) if name2 else m.end(2))

        if not _overlaps(s, e, taken):
            entities.append([s, e, "PERSON"])
            taken.append((s, e))

    # 2. Hashtag names: "john #tan#" or standalone "#tan#"
    #    Reference uses the span without the trailing '#':
    #    "janice #Teh" -> [126:137] and "#Wei Ling#" interior "Wei Ling" -> [145:153]
    for m in re.finditer(r"(?:([A-Za-z]+)\s+)?(#([^#]+)#)", text):
        leading = m.group(1)
        inner = m.group(3).strip()
        inner_s = m.start(3)
        inner_e = m.end(3)
        if leading and leading.lower() not in _NOT_NAME:
            # "john #tan#" -> span from leading word start to end of inner name
            s = m.start(1)
            e = inner_e
        else:
            # standalone "#choo#" -> just the inner name
            s = inner_s
            e = inner_e
        if s < e and not _overlaps(s, e, taken):
            entities.append([s, e, "PERSON"])
            taken.append((s, e))

    # 3. Context phrases — only capture plausible short name tokens (<=15 chars)
    _context_stoppers = _NOT_NAME | {
        "know", "have", "call", "reach", "meet", "speak", "said",
        "see", "send", "get", "try", "go", "come",
    }
    for m in re.finditer(
        r"\b(?:this\s+is|i'?m|hi|hello|name\s+is)\s+([A-Za-z]{2,15})\b",
        text,
        re.IGNORECASE,
    ):
        name = m.group(1)
        ns, ne = m.start(1), m.end(1)
        if name.lower() not in _context_stoppers and not _overlaps(ns, ne, taken):
            entities.append([ns, ne, "PERSON"])
            taken.append((ns, ne))

    # 4. Possessive at position 0: "Chandra's", "Vijay's", "Boon's", "Hong's"
    for m in re.finditer(r"^([A-Z][a-z]{1,})'s?\b", text):
        ns, ne = m.start(1), m.end(1)
        if text[ns: ne].lower() not in _NOT_NAME and not _overlaps(ns, ne, taken):
            entities.append([ns, ne, "PERSON"])
            taken.append((ns, ne))

    # 5. Capital name at position 0 followed by comma (strong signal: "Ravi, this could be")
    m0 = re.match(r"^([A-Z][a-z]{1,}),", text)
    if m0:
        ns, ne = m0.start(1), m0.end(1)
        if text[ns: ne].lower() not in _NOT_NAME and not _overlaps(ns, ne, taken):
            entities.append([ns, ne, "PERSON"])
            taken.append((ns, ne))

    # 6. Capital names mid-sentence (not at position 0, not in excluded list)
    for m in re.finditer(r"\b[A-Z][a-z]{1,}\b", text):
        word = m.group()
        s, e = m.start(), m.end()
        if s == 0:
            continue
        if word.lower() in _NOT_NAME:
            continue
        if _overlaps(s, e, taken):
            continue
        if s > 0 and text[s - 1].isalpha():
            continue
        entities.append([s, e, "PERSON"])
        taken.append((s, e))

    return entities


def predict_with_rules(text: str) -> List[List]:
    """Rule-based NER pipeline."""
    entities: List[List] = []
    taken: List[Tuple[int, int]] = []  # already claimed spans

    # Priority order: structured IDs first, then free-text entities

    # 1. Credit card (16 digits in 4 groups)
    for s, e in _extract_credit_card(text):
        _add(entities, taken, s, e, "CREDIT_CARD")

    # 2. Bank account / account number (context-classified together)
    for s, e, is_bank in _extract_bank_account(text, taken):
        if not _overlaps(s, e, taken):
            label = "BANK_ACCOUNT" if is_bank else "ACCOUNT_NUM"
            entities.append([s, e, label])
            taken.append((s, e))

    # 3. NRIC
    for s, e in _extract_nric(text):
        _add(entities, taken, s, e, "NRIC")

    # 4. Passport
    for s, e in _extract_passport(text):
        _add(entities, taken, s, e, "PASSPORT_NUM")

    # 5. Car plate
    for s, e in _extract_car_plate(text):
        _add(entities, taken, s, e, "CAR_PLATE")

    # 6. Standard email
    for s, e in _extract_standard_email(text):
        _add(entities, taken, s, e, "EMAIL")

    # 7. Spoken email
    for s, e in _extract_spoken_email(text):
        _add(entities, taken, s, e, "EMAIL")

    # 8. Phone
    for s, e in _extract_phone(text):
        _add(entities, taken, s, e, "PHONE")

    # 9. ORG (known brands/banks)
    for ent in _find_keyword_spans(text, _ORGS, "ORG", taken):
        entities.append(ent)

    # 10. LOC (before GPE so 2-word locations block component GPEs)
    for ent in _find_keyword_spans(text, _LOCS, "LOC", taken):
        entities.append(ent)

    # 11. GPE
    for ent in _find_keyword_spans(text, _GPES, "GPE", taken):
        entities.append(ent)

    # 12. DATE
    for ent in _extract_date_expressions(text, taken):
        entities.append(ent)

    # 13. MONEY (before CARDINAL — "fifty dollar" must be MONEY, not CARDINAL "fifty")
    for ent in _extract_money(text, taken):
        entities.append(ent)

    # 14. CARDINAL (before TIME so "about two P_M" => CARDINAL "about two")
    for ent in _extract_cardinals(text, taken):
        entities.append(ent)

    # 15. TIME
    for ent in _extract_time_expressions(text, taken):
        entities.append(ent)

    # 16. PERSON (last – most ambiguous)
    for ent in _extract_persons(text, taken):
        entities.append(ent)

    # Sort by start position
    entities.sort(key=lambda x: x[0])
    return entities


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    input_path: str,
    output_path: str,
    use_rules: bool = False,
    model: str = "gpt-4o-mini",
    delay: float = 0.5,
) -> None:
    client = None
    if not use_rules:
        try:
            from openai import OpenAI  # type: ignore

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                print(
                    "[INFO] OPENAI_API_KEY not set. Falling back to rule-based NER."
                )
                use_rules = True
            else:
                client = OpenAI(api_key=api_key)
                print(f"[INFO] Using OpenAI model: {model}")
        except ImportError:
            print("[INFO] openai package not installed. Falling back to rule-based NER.")
            use_rules = True

    if use_rules:
        print("[INFO] Using rule-based NER.")

    results = []
    with open(input_path, "r", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh if line.strip()]

    total = len(lines)
    for i, record in enumerate(lines, 1):
        line_id = record["id"]
        text = record["text"]

        if use_rules:
            entities = predict_with_rules(text)
        else:
            entities = predict_with_llm(text, client, model=model)
            if delay > 0:
                time.sleep(delay)

        out_record = {"id": line_id, "text": text, "entities": entities}
        results.append(out_record)

        if i % 10 == 0 or i == total:
            print(f"  Processed {i}/{total} ...")

    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in results:
            fh.write(json.dumps(rec) + "\n")

    print(f"[INFO] Wrote {len(results)} records to {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NER on spoken-text using LLM prompt engineering."
    )
    parser.add_argument(
        "--input",
        default="Reference.jsonl",
        help="Path to input JSONL file (default: Reference.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="Hypothesis.jsonl",
        help="Path to output JSONL file (default: Hypothesis.jsonl)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--use-rules",
        action="store_true",
        help="Use rule-based NER instead of LLM (no API key required)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between API calls (default: 0.5)",
    )
    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        use_rules=args.use_rules,
        model=args.model,
        delay=args.delay,
    )


if __name__ == "__main__":
    main()

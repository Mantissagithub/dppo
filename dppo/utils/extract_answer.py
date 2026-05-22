import re


FINAL_RE = re.compile(r"####\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)")
NUMBER_RE = re.compile(r"([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)")


def normalize_number(text):
    return text.replace(",", "").strip()


def extract_final_answer(text):
    match = FINAL_RE.search(text)
    if match:
        return normalize_number(match.group(1))
    numbers = NUMBER_RE.findall(text)
    if not numbers:
        return None
    return normalize_number(numbers[-1])

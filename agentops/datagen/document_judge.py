"""Judge a Document Agent answer against the known answer.

No LLM is used, so the label is cheap, fast and repeatable. It is a keyword
judge: good for numbers, names and dates, and for yes/no and "not in the
document" answers. It can be slightly strict on long free-text answers - the
export step keeps the raw answer in the database so labels can be re-checked.
"""

import re

REFUSAL_PHRASES = (
    "could not find", "couldn't find", "can't find", "cannot find", "not mentioned",
    "not specified", "not provided", "not stated", "not available", "not included",
    "does not contain", "doesn't contain", "does not mention", "doesn't mention",
    "does not specify", "does not provide", "do not contain", "no information",
    "no mention", "not found", "not present", "not listed", "unavailable",
    "does not include", "no details", "not explicitly", "not addressed", "unable to find",
    "does not state", "no such", "not covered in", "isn't mentioned", "is not in",
    "cannot determine", "cannot provide", "can't determine", "unable to determine",
    "cannot be determined", "no reference", "is mentioned", "zikr nahi", "nahi mila",
    "nahi mili", "nahi hai", "jaankari nahi", "cannot calculate", "can not be calculated",
)

NUMBER_WORDS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
                "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def normalise(text) -> str:
    text = str(text or "").lower()
    text = text.replace("₹", " ").replace("*", " ").replace("`", " ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)          # 22,000 -> 22000
    text = re.sub(r"[^\w\s.\-/@:]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def with_number_words(text: str) -> str:
    return " ".join(NUMBER_WORDS.get(word, word) for word in text.split())


def refused(answer) -> bool:
    text = normalise(answer)
    if any(phrase in text for phrase in REFUSAL_PHRASES):
        return True
    # "no parking space is mentioned", "no information about ..."
    return re.search(r"\bno\b[^.]{0,70}\b(mention|mentioned|information|record|details|reference)\b", text) is not None


def yes_no(answer):
    """'yes', 'no' or None, read from the start of the answer."""
    head = normalise(answer)[:200]
    first_yes = re.search(r"\b(yes|haan|han)\b", head)
    first_no = re.search(r"\b(no|nahi|not allowed|prohibited|not covered|cannot|is not|are not|does not|not available)\b", head)
    if first_yes and (not first_no or first_yes.start() < first_no.start()):
        return "yes"
    if first_no:
        return "no"
    return None


def contains(text: str, wanted: str) -> bool:
    """Whole word / whole number match.
    - "6" must not match "16" or "2.6"
    - a long number (3+ digits) may be followed by decimals: 166666 in 166666.67
    - a plain word may end with s/es/ed/ing: "beep" matches "beeps"
    """
    escaped = re.escape(wanted)
    if wanted.isdigit() and len(wanted) >= 3:
        pattern = r"(?<![\w])(?<!\d\.)" + escaped + r"(?!\w)"
    elif wanted.isalpha():
        pattern = r"(?<![\w])" + escaped + r"(?:s|es|ed|ing)?(?!\w)"
    else:
        pattern = r"(?<![\w])(?<!\d\.)" + escaped + r"(?!\w)(?!\.\d)"
    return re.search(pattern, text) is not None


ZERO_WORDS = ("no co-payment", "no copayment", "zero", "none", "nil", "not applicable")


def piece_found(text: str, text_numbers: str, piece: str, question_type: str) -> bool:
    wanted = normalise(piece)
    if contains(text, wanted) or contains(text_numbers, with_number_words(wanted)):
        return True
    if wanted == "0" and any(word in text for word in ZERO_WORDS):
        return True
    # Rounded calculation: 91666.67 -> 91,667 is also right.
    if question_type == "Calculation" and wanted.isdigit() and len(wanted) >= 3:
        return contains(text, str(int(wanted) + 1))
    return False


def answer_correct(question_type: str, expected: list, answer) -> bool:
    text = normalise(answer)
    if not text:
        return False

    if question_type == "Unanswerable":
        return refused(answer)

    if question_type == "Yes No":
        return yes_no(answer) == (expected[0] if expected else None)

    if not expected:
        return False

    # An answer that says "not found" cannot be a correct answer to a
    # question whose answer IS in the document.
    # Every expected piece must be in the answer (numbers/names/dates),
    # as a whole word/number (so "6" does not match "16" or "2.6").
    text_numbers = with_number_words(text)
    all_found = all(piece_found(text, text_numbers, p, question_type) for p in expected)
    if not all_found:
        return False

    # The answer opens with "not found" but may still END with the right
    # answer ("... Answer: 385 units"). Accept only if the closing part has it.
    opens_with_refusal = refused(text[:250]) or re.match(
        r"(based on the retrieved context, )?(there is|i could) no", text)
    if opens_with_refusal:
        tail = text[-250:]
        tail_numbers = with_number_words(tail)
        return all(piece_found(tail, tail_numbers, p, question_type) for p in expected)
    return True


def source_correct(doc_file: str, sources) -> bool:
    return doc_file in (sources or [])
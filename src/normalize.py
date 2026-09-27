import re
import unicodedata
from unidecode import unidecode
from typing import Optional


# Common business/legal terms.
# We keep these as tokens rather than deleting them aggressively.
BUSINESS_SUFFIXES = {
    "llc", "inc", "incorporated", "corp", "corporation",
    "co", "company", "ltd", "limited", "pvt", "private",
    "plc", "llp", "lp", "llc", "gmbh", "sa", "sarl",
    "pte", "pty", "bv", "nv"
}


def unicode_normalize(text: Optional[str]) -> str:
    """
    Unicode normalization while preserving non-Latin scripts.

    Example:
        accented French characters remain meaningful.
        Hindi/Tamil/etc. are NOT converted to ASCII.
    """
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)

    return text

def transliterated_text(text: str) -> str:
    if text is None:
        return ""

    return normalize_text(
        unidecode(str(text))
    )

def accent_fold(text: Optional[str]) -> str:
    """
    Remove Latin diacritics while preserving non-Latin scripts.

    Examples:
        Cátering -> Catering
        Públic   -> Public

    Hindi/Tamil/etc. are left intact.
    """
    if text is None:
        return ""

    text = unicode_normalize(text)

    result = []

    for ch in text:
        # Decompose the character.
        decomposed = unicodedata.normalize("NFD", ch)

        for dch in decomposed:
            # Unicode combining marks are removed.
            if unicodedata.category(dch) != "Mn":
                result.append(dch)

    return "".join(result)

def accent_folded_text(text: Optional[str]) -> str:
    """
    Case-normalized + accent-folded representation.
    """
    text = accent_fold(text)
    text = text.casefold()
    text = normalize_punctuation(text)
    text = normalize_whitespace(text)

    return text


def normalize_case(text: str) -> str:
    """Unicode-aware lowercase."""
    return text.casefold()


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_punctuation(text: str) -> str:
    """
    Replace punctuation/separators with spaces.

    We intentionally don't remove characters outright because
    doing so can accidentally concatenate useful tokens.
    """
    text = re.sub(r"[\.,;:!?()\[\]{}<>\"'`~@#$%^&*_+=|\\/]+", " ", text)
    text = re.sub(r"[-–—]+", " ", text)

    return normalize_whitespace(text)


def normalize_text(text: Optional[str]) -> str:
    """
    General Unicode-safe normalization.
    """
    if text is None:
        return ""

    text = unicode_normalize(text)
    text = normalize_case(text)
    text = normalize_punctuation(text)
    text = normalize_whitespace(text)

    return text


def alphanumeric_form(text: Optional[str]) -> str:
    """
    Compact representation containing Unicode letters/numbers.

    IMPORTANT:
    This is Unicode-aware; it does NOT use ASCII encoding.
    """
    text = normalize_text(text)

    chars = []

    for ch in text:
        category = unicodedata.category(ch)

        if category.startswith(("L", "N")):
            chars.append(ch)

    return "".join(chars)


def token_form(text: Optional[str]) -> list[str]:
    """
    Return normalized tokens.
    """
    text = normalize_text(text)

    if not text:
        return []

    return text.split()


def sorted_token_form(text: Optional[str]) -> str:
    """
    Token-order-insensitive representation.

    Useful when addresses/business names reorder tokens.
    """
    tokens = token_form(text)

    return " ".join(sorted(tokens))


def character_ngrams(text: Optional[str], n: int = 3) -> set[str]:
    """
    Unicode-safe character n-grams.

    Spaces are retained after normalization so that word boundaries
    still contribute information.
    """
    text = normalize_text(text)

    if not text:
        return set()

    padded = f" {text} "

    if len(padded) < n:
        return {padded}

    return {
        padded[i:i + n]
        for i in range(len(padded) - n + 1)
    }


def extract_numbers(text: Optional[str]) -> list[str]:
    """
    Extract numeric components.

    Addresses often contain highly informative numbers.
    """
    text = normalize_text(text)

    return re.findall(r"\d+(?:[./-]\d+)*", text)


def number_signature(text: Optional[str]) -> str:
    """
    Sorted unique numeric components.
    """
    nums = extract_numbers(text)

    return " ".join(sorted(set(nums)))


def business_tokens(text: Optional[str]) -> list[str]:
    """
    Business-name tokens with legal suffixes retained.

    We don't delete suffixes here because they may still be useful
    as features. A later feature can ignore them when appropriate.
    """
    return token_form(text)


def content_tokens(text: Optional[str]) -> list[str]:
    """
    Remove common legal/business suffixes from token representation.

    This is an additional representation, not a replacement.
    """
    tokens = business_tokens(text)

    return [
        token
        for token in tokens
        if token not in BUSINESS_SUFFIXES
    ]


def content_token_form(text: Optional[str]) -> str:
    return " ".join(content_tokens(text))


def normalize_name(text: Optional[str]) -> dict:
    normalized = normalize_text(text)
    folded = accent_folded_text(text)

    return {
        "raw": "" if text is None else str(text),
        "norm": normalized,
        "accent_folded": folded,
        "alnum": alphanumeric_form(normalized),
        "alnum_folded": alphanumeric_form(folded),
        "tokens": token_form(normalized),
        "sorted_tokens": sorted_token_form(normalized),
        "content_tokens": content_token_form(normalized),
        "numbers": number_signature(normalized),
    }

def normalize_address(text: Optional[str]) -> dict:
    normalized = normalize_text(text)
    folded = accent_folded_text(text)

    return {
        "raw": "" if text is None else str(text),
        "norm": normalized,
        "accent_folded": folded,
        "alnum": alphanumeric_form(normalized),
        "alnum_folded": alphanumeric_form(folded),
        "tokens": token_form(normalized),
        "sorted_tokens": sorted_token_form(normalized),
        "numbers": number_signature(normalized),
    }

if __name__ == "__main__":
    examples = [
        "Physical Therapy Global",
        "PHYSICAL THERAPY GLOBAL",
        "157 Beechwood Street, Warsaw",
        "157 BEECH WOOD STREET, WARSAW",
        "Anand Investment Private Limited",
        "ஆனந்த் இன்வெஸ்ட்மெண்ட் பிரைவேட்",
        "Urban Management Limited",
        "अर्बन मैनेजमेंट लिमिटेड",
        "2408 Lexington Ave Drive Company",
        "35 Commons Dr., Unit 429",
    ]

    print("=" * 80)
    print("NORMALIZATION TEST")
    print("=" * 80)

    for text in examples:
        print("\nRAW:")
        print(text)

        result = normalize_name(text)

        print("NORM:")
        print(result["norm"])

        print("ALNUM:")
        print(result["alnum"])

        print("TOKENS:")
        print(result["tokens"])

        print("CONTENT TOKENS:")
        print(result["content_tokens"])

        print("NUMBERS:")
        print(result["numbers"])

        print("CHAR 3-GRAMS:")
        print(list(character_ngrams(text, 3))[:10])
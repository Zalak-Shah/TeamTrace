"""
============================================================
Universal Name Extraction
Passport / Visa / National ID / Driving Licence
============================================================
Strategy (in priority order):
  1. MRZ parsing        -> Passport, Visa (most reliable, fixed format)
  2. Label anchoring     -> ID Card, Driving Licence, anything with
                             "Name:", "Surname:", "Given Name(s):" etc.
  3. Safe heuristic guess -> last resort, big stopword list
============================================================
"""

import re
from typing import Optional


# ============================================================
# 1. MRZ PARSING (Passport / Visa)
# ============================================================
# MRZ (Machine Readable Zone) lines look like:
#   P<INDSHARMA<<RAHUL<<<<<<<<<<<<<<<<<<<<<<<<<<
#   1234567890IND9001015M3001017<<<<<<<<<<<<<<02
#
# Format: P<CCCSURNAME<<GIVEN<NAMES<<<<<<<<<<<<<<<<
# '<' is filler/separator, '<<' separates surname from given names.

MRZ_LINE_RE = re.compile(r"^[A-Z0-9<]{28,44}$")


def _clean_mrz_name_field(raw: str) -> str:
    """Convert an MRZ name field ('SHARMA<<RAHUL<KUMAR') into 'Rahul Kumar Sharma' style text."""
    raw = raw.strip("<")
    parts = [p for p in raw.split("<") if p]
    return " ".join(word.capitalize() for word in parts)


def extract_name_from_mrz(text: str) -> Optional[str]:
    """
    Scans OCR text for MRZ-style lines and extracts surname + given names.
    Works for TD3 (passport, 2 lines x 44 chars) and TD1/TD2 (ID/visa variants).
    """

    candidate_lines = []

    for line in text.splitlines():

        cleaned = line.strip().replace(" ", "")

        # OCR sometimes drops/adds a stray char; allow a little slack
        if len(cleaned) < 20:
            continue

        # MRZ lines are mostly A-Z, 0-9 and '<'
        letters_ok = re.fullmatch(r"[A-Z0-9<]+", cleaned)

        if letters_ok and "<" in cleaned:
            candidate_lines.append(cleaned)

    if not candidate_lines:
        return None

    # The name line is the one starting with a document code letter
    # (P for passport, V for visa, I/A/C for ID cards) followed by '<'
    # and containing '<<' (surname/given-name separator).
    for line in candidate_lines:

        if re.match(r"^[PVIAC][A-Z<]", line) and "<<" in line:

            # Strip leading document code + issuing country (first 5 chars: e.g. "P<IND")
            match = re.match(r"^[A-Z]<[A-Z]{3}(.+)$", line)

            name_field = match.group(1) if match else line[5:]

            if "<<" not in name_field:
                continue

            surname_part, _, given_part = name_field.partition("<<")

            surname = _clean_mrz_name_field(surname_part)
            given = _clean_mrz_name_field(given_part)

            if surname and given:
                return f"{given} {surname}"

            if surname:
                return surname

    return None


# ============================================================
# 2. LABEL-ANCHORED EXTRACTION (ID Card / Driving Licence / generic)
# ============================================================

NAME_LABELS = (
    "given name",
    "given names",
    "full name",
    "surname",
    "holder's name",
    "name of holder",
    "licence holder",
    "license holder",
    "name",
)

# Other field labels that can appear right after / merged with the name value.
# We stop capturing as soon as one of these shows up, instead of swallowing
# it into the name (this is what caused "S/W/D" to get merged into the name).
NEXT_FIELD_LABELS = (
    "s/w/d", "s/o", "w/o", "d/o", "c/o",
    "dob", "date of birth",
    "address", "bg", "blood group",
    "licence no", "license no", "dl no",
    "validity", "issue date", "date of issue",
    "authorisation", "authorization",
)

VALUE_RE = re.compile(r"[A-Za-z][A-Za-z .'-]{1,49}")

# Matches everything up to (but not including) the next field label, if present.
_STOP_PATTERN = re.compile(
    r"(.*?)(?:\b(?:" + "|".join(re.escape(lbl) for lbl in NEXT_FIELD_LABELS) + r")\b.*)?$",
    re.IGNORECASE,
)


def _trim_to_next_field(value: str) -> str:
    """Cuts off a captured value as soon as another field label appears in it."""
    match = _STOP_PATTERN.match(value)
    trimmed = match.group(1) if match else value
    return trimmed.strip(" :-\t")


def extract_name_from_labels(text: str) -> Optional[str]:
    """
    Looks for a line containing a name label (e.g. "Name:", "Given Name(s):")
    and extracts the value either after the colon on the same line,
    or from the very next non-empty line. Stops capturing as soon as
    another field label (S/W/D, DOB, Address, etc.) appears, so merged
    OCR lines don't pull in the wrong text.
    """

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for index, line in enumerate(lines):

        lower = line.lower()

        for label in NAME_LABELS:

            if label not in lower:
                continue

            # avoid matching lines like "Name of Issuing Authority"
            if "authority" in lower or "department" in lower:
                continue

            # avoid accidentally matching the S/W/D line itself
            if any(rel in lower for rel in ("s/w/d", "s/o", "w/o", "d/o", "c/o")):
                continue

            # Try value on the same line, after the label
            after_label = re.sub(
                rf".*\b{re.escape(label)}\b\s*[:\-]?\s*",
                "",
                line,
                flags=re.IGNORECASE,
            ).strip()

            after_label = _trim_to_next_field(after_label)

            if after_label and VALUE_RE.fullmatch(after_label) and len(after_label.split()) <= 5:
                return after_label

            # Otherwise, try the next line
            if index + 1 < len(lines):

                next_line = _trim_to_next_field(lines[index + 1])

                if next_line and VALUE_RE.fullmatch(next_line) and len(next_line.split()) <= 5:
                    return next_line

    return None


# ============================================================
# 3. SAFE FALLBACK GUESS (last resort)
# ============================================================

SKIP_WORDS = {
    "GOVERNMENT", "INDIA", "REPUBLIC", "INCOME", "TAX", "DEPARTMENT",
    "UNIQUE", "IDENTIFICATION", "AUTHORITY", "TRANSPORT", "LICENCE",
    "LICENSE", "AADHAAR", "PASSPORT", "NATIONALITY", "VISA", "PERMIT",
    "VALID", "EXPIRY", "SURNAME", "GIVEN", "NAME", "DATE", "BIRTH",
    "ISSUE", "ISSUED", "UNTIL", "FROM", "TO", "ADDRESS", "STATE",
    "CITY", "COUNTRY", "SEX", "GENDER", "BLOOD", "GROUP", "TYPE",
    "CLASS", "CATEGORY", "NUMBER", "NO", "CARD", "IDENTITY", "NATIONAL",
    "ENTRY", "ENTRIES", "PLACE", "OF", "DISTRICT", "PIN", "CODE",
    "MINISTRY", "OFFICE", "HOLDER", "SIGNATURE", "PHOTO", "PHOTOGRAPH",
    "APPLICATION", "REFERENCE", "CATEGORY", "VEHICLE", "ENDORSEMENT",
    "RESTRICTION", "CONDITION", "AUTHORIZED", "AUTHORISED", "REGION",
    "PROVINCE", "TERRITORY", "REPUBLIC", "FEDERAL", "STATUS", "DOCUMENT",
}


def extract_name_fallback(text: str) -> Optional[str]:

    for line in text.splitlines():

        clean = line.strip()

        if not (4 <= len(clean) <= 50):
            continue

        if not re.fullmatch(r"[A-Za-z][A-Za-z .'-]+", clean):
            continue

        words = {word.upper() for word in clean.split()}

        if words & SKIP_WORDS:
            continue

        if len(words) <= 5:
            return clean

    return None


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def guess_name(text: str) -> Optional[str]:
    """
    Universal name extraction across Passport, Visa, National ID,
    Driving Licence and other document types.

    Order of attempts:
      1. MRZ (passport / visa)
      2. Label-anchored (ID / licence / generic)
      3. Safe heuristic fallback
    """

    name = extract_name_from_mrz(text)

    if name:
        return name

    name = extract_name_from_labels(text)

    if name:
        return name

    return extract_name_fallback(text)

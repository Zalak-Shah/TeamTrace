"""
============================================================
TeamTrace / BorderShield AI - Backend
OCR.space + OpenCV + SQLite Identity Intelligence
============================================================

FULL REPLACEMENT MAIN.PY

Improved:
- Visa detection
- Passport MRZ detection
- Visa date extraction
- OCR normalization
- Expiry date detection
- Document number extraction
- Name extraction
- Visual analysis
- Risk scoring
- Dashboard API
- Identity intelligence API
============================================================
"""

import os
import re
import sqlite3
from datetime import date, datetime
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from database import (
    initialize_database,
    create_or_get_identity,
    save_screening,
    get_identity_intelligence,
    add_audit_log,
)

from name_extraction import guess_name


# ============================================================
# SETUP
# ============================================================

load_dotenv()

OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY")

if not OCR_SPACE_API_KEY:
    print("WARNING: OCR_SPACE_API_KEY is not set in .env")


app = FastAPI(
    title="BorderShield AI Backend",
    version="2.0.0",
)

initialize_database()


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# FILE SETTINGS
# ============================================================

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "application/pdf",
}

MAX_FILE_SIZE_MB = 10


# ============================================================
# RISK SCORING
# ============================================================

WEIGHTS = {
    "CRITICAL_MISMATCH": 40,
    "INVALID_NUMBER_FORMAT": 25,
    "EXPIRED_DOCUMENT": 20,
    "MISSING_REQUIRED_FIELD": 15,
    "LOW_OCR_CONFIDENCE": 10,
    "VISUAL_ISSUE": 10,
}


def severity_for_weight(weight: int) -> str:

    if weight >= 25:
        return "HIGH"

    if weight >= 10:
        return "MEDIUM"

    return "LOW"


def status_for_score(score: int) -> str:

    if score <= 20:
        return "LOW RISK"

    if score <= 50:
        return "REVIEW REQUIRED"

    if score <= 75:
        return "SUSPICIOUS"

    return "HIGH RISK"


# ============================================================
# REGEX PATTERNS
# ============================================================

AADHAAR_NUMBER_RE = re.compile(
    r"\b\d{4}\s?\d{4}\s?\d{4}\b"
)


PAN_NUMBER_RE = re.compile(
    r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    re.IGNORECASE,
)


PASSPORT_NUMBER_RE = re.compile(
    r"\b[A-Z]{1,2}\s?\d{6,8}\b",
    re.IGNORECASE,
)


DL_NUMBER_RE = re.compile(
    r"\b[A-Z]{2}[-\s]?\d{2}[-\s]?\d{4,11}\b",
    re.IGNORECASE,
)


# ============================================================
# VISA NUMBER PATTERNS
# ============================================================

VISA_NUMBER_PATTERNS = [

    # Example:
    # ABC123456
    re.compile(
        r"\b[A-Z]{1,4}\d{5,10}\b",
        re.IGNORECASE,
    ),

    # Example:
    # 123456789
    re.compile(
        r"\b\d{7,12}\b"
    ),

    # Example:
    # A1234567
    re.compile(
        r"\b[A-Z]\d{6,9}\b",
        re.IGNORECASE,
    ),
]


# ============================================================
# DATE REGEX
# ============================================================

DATE_RE = re.compile(
    r"""
    \b(
        \d{2}[/-]\d{2}[/-]\d{4}
        |
        \d{4}[/-]\d{2}[/-]\d{2}
        |
        \d{1,2}\s+
        (?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)
        \s+\d{4}
        |
        \d{1,2}\s+
        (?:STY|LUT|MAR|KWI|MAJ|CZE|LIP|SIE|WRZ|PAZ|PAŹ|LIS|GRU)
        \s*/?\s*
        (?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)
        \s+\d{4}
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ============================================================
# DOCUMENT KEYWORDS
# ============================================================

KEYWORDS = {

    "Aadhaar": [
        "unique identification authority",
        "aadhaar",
        "uidai",
        "government of india",
        "enrolment",
        "vid",
    ],

    "PAN Card": [
        "income tax department",
        "permanent account number",
        "pan card",
        "income tax",
    ],

    "Passport": [
        "passport",
        "nationality",
        "surname",
        "given name",
        "place of birth",
        "date of birth",
        "date of issue",
        "date of expiry",
        "type",
        "sex",
        "issuing authority",
    ],

    "Visa": [
        "visa",
        "visa number",
        "visa no",
        "visa type",
        "type of visa",
        "entry",
        "entries",
        "valid from",
        "valid until",
        "valid till",
        "valid through",
        "duration of stay",
        "number of entries",
        "issued by",
        "schengen",
        "schengen states",
        "etats schengen",
        "états schengen",
        "vfs",
        "consulate",
        "embassy",
    ],

    "National ID": [
        "national identity",
        "identity card",
        "national id",
        "identification card",
    ],

    "Driving Licence": [
        "driving licence",
        "driving license",
        "transport department",
        "licence no",
        "license no",
        "dl no",
    ],

    "Permit": [
        "permit",
        "validity",
        "issued by",
        "permit number",
        "permission",
    ],
}


# ============================================================
# MONTH MAPPING
# ============================================================

MONTHS = {

    "JAN": 1,
    "STY": 1,

    "FEB": 2,
    "LUT": 2,

    "MAR": 3,

    "APR": 4,
    "KWI": 4,

    "MAY": 5,
    "MAJ": 5,

    "JUN": 6,
    "CZE": 6,

    "JUL": 7,
    "LIP": 7,

    "AUG": 8,

    "SEP": 9,
    "WRZ": 9,

    "OCT": 10,
    "PAZ": 10,
    "PAŹ": 10,

    "NOV": 11,
    "LIS": 11,

    "DEC": 12,
    "GRU": 12,
}


# ============================================================
# OCR NORMALIZATION
# ============================================================

def normalize_ocr_text(text: str) -> str:

    if not text:
        return ""

    text = text.replace("\r", "\n")

    # Common OCR substitutions
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u00a0": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def compact_text(text: str) -> str:

    if not text:
        return ""

    return re.sub(
        r"[^A-Z0-9]",
        "",
        text.upper(),
    )


# ============================================================
# DATE PARSING
# ============================================================

def to_date(value: Optional[str]) -> Optional[date]:

    if not value:
        return None

    value = value.strip().upper()

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    # Normalize separators
    value = value.replace(
        " / ",
        "/",
    )

    # --------------------------------------------------------
    # Numeric formats
    # --------------------------------------------------------

    for fmt in (
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
    ):

        try:

            return datetime.strptime(
                value,
                fmt,
            ).date()

        except ValueError:
            pass

    # --------------------------------------------------------
    # Text month
    # --------------------------------------------------------

    match = re.fullmatch(
        r"(\d{1,2})\s+([A-ZĄĆĘŁŃÓŚŹŻ]{3,5})"
        r"(?:\s*/\s*[A-Z]{3,5})?"
        r"\s+(\d{4})",
        value,
    )

    if match:

        day = int(
            match.group(1)
        )

        month_name = match.group(2).upper()

        month = MONTHS.get(
            month_name
        )

        year = int(
            match.group(3)
        )

        if month:

            try:

                return date(
                    year,
                    month,
                    day,
                )

            except ValueError:

                return None

    return None


# ============================================================
# DATE EXTRACTION
# ============================================================

def extract_dates(text: str) -> List[str]:

    if not text:
        return []

    text = normalize_ocr_text(
        text
    )

    matches = DATE_RE.findall(
        text
    )

    result = []

    for value in matches:

        value = value.strip()

        if value and value not in result:

            result.append(value)

    return result


def extract_parsed_dates(text: str) -> List[date]:

    values = extract_dates(
        text
    )

    result = []

    for value in values:

        parsed = to_date(
            value
        )

        if parsed and parsed not in result:

            result.append(parsed)

    return result


# ============================================================
# LABELED DATE EXTRACTION
# ============================================================

def extract_labeled_date(
    text: str,
    labels: List[str],
) -> Optional[date]:

    if not text:
        return None

    normalized = normalize_ocr_text(
        text
    )

    # --------------------------------------------------------
    # Search each line first
    # --------------------------------------------------------

    lines = normalized.splitlines()

    for index, line in enumerate(lines):

        lower_line = line.lower()

        found_label = False

        for label in labels:

            if label.lower() in lower_line:

                found_label = True
                break

        if not found_label:
            continue

        # Date on same line
        dates = extract_dates(
            line
        )

        if dates:

            parsed = to_date(
                dates[0]
            )

            if parsed:
                return parsed

        # Date may be on next line because OCR separated columns
        if index + 1 < len(lines):

            next_line = lines[index + 1]

            dates = extract_dates(
                next_line
            )

            if dates:

                parsed = to_date(
                    dates[0]
                )

                if parsed:
                    return parsed

    # --------------------------------------------------------
    # Search complete text around label
    # --------------------------------------------------------

    lower_text = normalized.lower()

    for label in labels:

        start = lower_text.find(
            label.lower()
        )

        if start == -1:
            continue

        window = normalized[
            start:start + 100
        ]

        dates = extract_dates(
            window
        )

        if dates:

            parsed = to_date(
                dates[0]
            )

            if parsed:
                return parsed

    return None


# ============================================================
# DOCUMENT CLASSIFICATION
# ============================================================

def classify_document(text: str) -> str:

    text = normalize_ocr_text(
        text
    )

    lower = text.lower()

    upper = text.upper()

    scores = {}

    for doc_type, keywords in KEYWORDS.items():

        scores[doc_type] = sum(
            1
            for keyword in keywords
            if keyword.lower() in lower
        )

    # --------------------------------------------------------
    # Aadhaar
    # --------------------------------------------------------

    if (
        AADHAAR_NUMBER_RE.search(text)
        and scores.get("Aadhaar", 0) > 0
    ):

        return "Aadhaar"

    # --------------------------------------------------------
    # PAN
    # --------------------------------------------------------

    if PAN_NUMBER_RE.search(text):

        if (
            scores.get("PAN Card", 0) > 0
            or "income tax" in lower
        ):

            return "PAN Card"

    # --------------------------------------------------------
    # VISA
    #
    # Visa is checked BEFORE passport because visa stickers
    # can contain words such as nationality/date/sex and may
    # otherwise be incorrectly classified.
    # --------------------------------------------------------

    visa_score = scores.get(
        "Visa",
        0,
    )

    visa_indicators = [
        "visa",
        "schengen",
        "etats schengen",
        "états schengen",
        "valid from",
        "valid until",
        "valid till",
        "valid through",
        "entries",
        "duration of stay",
        "visa no",
        "visa number",
        "vfs",
    ]

    visa_indicator_count = sum(
        1
        for item in visa_indicators
        if item.lower() in lower
    )

    if (
        visa_score >= 2
        or visa_indicator_count >= 2
    ):

        return "Visa"

    # --------------------------------------------------------
    # Passport MRZ
    # --------------------------------------------------------

    if (
        "P<" in upper
        or re.search(
            r"\bP<[A-Z]{3}",
            upper,
        )
    ):

        return "Passport"

    # --------------------------------------------------------
    # Passport keywords
    # --------------------------------------------------------

    if (
        PASSPORT_NUMBER_RE.search(text)
        and scores.get("Passport", 0) >= 1
    ):

        return "Passport"

    if scores.get(
        "Passport",
        0,
    ) >= 2:

        return "Passport"

    # --------------------------------------------------------
    # Driving Licence
    # --------------------------------------------------------

    if (
        DL_NUMBER_RE.search(text)
        and scores.get(
            "Driving Licence",
            0,
        ) > 0
    ):

        return "Driving Licence"

    # --------------------------------------------------------
    # National ID
    # --------------------------------------------------------

    if scores.get(
        "National ID",
        0,
    ) >= 1:

        return "National ID"

    # --------------------------------------------------------
    # Permit
    # --------------------------------------------------------

    if scores.get(
        "Permit",
        0,
    ) >= 2:

        return "Permit"

    # --------------------------------------------------------
    # Best keyword score
    # --------------------------------------------------------

    best = max(
        scores,
        key=scores.get,
    )

    if scores[best] > 0:

        return best

    return "Unknown"


# ============================================================
# DOCUMENT NUMBER EXTRACTION
# ============================================================

def extract_document_number(
    text: str,
    doc_type: str,
) -> Optional[str]:

    text = normalize_ocr_text(
        text
    )

    if not text:
        return None

    # ========================================================
    # AADHAAR
    # ========================================================

    if doc_type == "Aadhaar":

        match = AADHAAR_NUMBER_RE.search(
            text
        )

        if match:

            return match.group(0).replace(
                " ",
                "",
            )

    # ========================================================
    # PAN
    # ========================================================

    elif doc_type == "PAN Card":

        match = PAN_NUMBER_RE.search(
            text
        )

        if match:

            return match.group(0).upper()

    # ========================================================
    # PASSPORT
    # ========================================================

    elif doc_type == "Passport":

        # First normal search
        match = PASSPORT_NUMBER_RE.search(
            text
        )

        if match:

            return re.sub(
                r"\s+",
                "",
                match.group(0).upper(),
            )

        # OCR may insert spaces between letter and number
        normalized = re.sub(
            r"(?<=[A-Z])\s+(?=\d)",
            "",
            text.upper(),
        )

        match = PASSPORT_NUMBER_RE.search(
            normalized
        )

        if match:

            return re.sub(
                r"\s+",
                "",
                match.group(0).upper(),
            )

    # ========================================================
    # DRIVING LICENCE
    # ========================================================

    elif doc_type == "Driving Licence":

        match = DL_NUMBER_RE.search(
            text
        )

        if match:

            return re.sub(
                r"\s+",
                "",
                match.group(0).upper(),
            )

    # ========================================================
    # VISA
    # ========================================================

    elif doc_type == "Visa":

        upper_text = text.upper()

        # ----------------------------------------------------
        # Look around explicit visa-number labels
        # ----------------------------------------------------

        label_patterns = [
            r"VISA\s*(?:NO|NUMBER|NR|N[O0])?\s*[:#]?\s*"
            r"([A-Z0-9]{5,15})",

            r"VISA\s*NUMBER\s*[:#]?\s*"
            r"([A-Z0-9]{5,15})",

            r"VISA\s*NO\s*[:#]?\s*"
            r"([A-Z0-9]{5,15})",
        ]

        for pattern in label_patterns:

            match = re.search(
                pattern,
                upper_text,
                re.IGNORECASE,
            )

            if match:

                value = re.sub(
                    r"[^A-Z0-9]",
                    "",
                    match.group(1).upper(),
                )

                if 5 <= len(value) <= 15:

                    return value

        # ----------------------------------------------------
        # Look line-by-line around visa labels
        # ----------------------------------------------------

        lines = upper_text.splitlines()

        for index, line in enumerate(lines):

            if (
                "VISA" in line
                or "NUMBER" in line
                or "NO." in line
                or "NO:" in line
            ):

                candidates = re.findall(
                    r"\b[A-Z0-9]{5,15}\b",
                    line,
                )

                for candidate in candidates:

                    candidate = candidate.upper()

                    if candidate in (
                        "VISA",
                        "NUMBER",
                        "SCHENGEN",
                        "ETATS",
                    ):

                        continue

                    # Must contain either a letter or be long numeric
                    if (
                        re.search(
                            r"[A-Z]",
                            candidate,
                        )
                        or len(candidate) >= 7
                    ):

                        return candidate

                # Next OCR line
                if index + 1 < len(lines):

                    next_line = lines[
                        index + 1
                    ]

                    candidates = re.findall(
                        r"\b[A-Z0-9]{5,15}\b",
                        next_line,
                    )

                    for candidate in candidates:

                        candidate = candidate.upper()

                        if (
                            candidate not in (
                                "VISA",
                                "NUMBER",
                                "SCHENGEN",
                            )
                        ):

                            return candidate

        # ----------------------------------------------------
        # Generic fallback
        # ----------------------------------------------------

        generic_candidates = []

        for pattern in VISA_NUMBER_PATTERNS:

            for match in pattern.finditer(
                upper_text
            ):

                value = match.group(0).upper()

                # Avoid dates
                if re.fullmatch(
                    r"\d{2,4}[/-]\d{2}[/-]\d{2,4}",
                    value,
                ):
                    continue

                # Avoid obvious years
                if re.fullmatch(
                    r"19\d{2}|20\d{2}",
                    value,
                ):
                    continue

                # Avoid common irrelevant OCR words
                if value in {
                    "SCHENGEN",
                    "ETATS",
                    "VISA",
                    "EUROPE",
                    "EUROPEAN",
                }:
                    continue

                generic_candidates.append(
                    value
                )

        if generic_candidates:

            # Prefer alphanumeric values
            alpha_numeric = [
                value
                for value in generic_candidates
                if re.search(
                    r"[A-Z]",
                    value,
                )
                and re.search(
                    r"\d",
                    value,
                )
            ]

            if alpha_numeric:

                return alpha_numeric[0]

            return generic_candidates[0]

    return None


# ============================================================
# PASSPORT MRZ EXTRACTION
# ============================================================

def extract_mrz(text: str) -> dict:

    result = {

        "detected": False,

        "line1": None,

        "line2": None,

        "document_number": None,

        "nationality": None,

        "date_of_birth": None,

        "sex": None,

        "expiry_date": None,

        "name": None,

        "valid": False,
    }

    if not text:
        return result

    raw_lines = text.splitlines()

    lines = []

    for raw_line in raw_lines:

        line = raw_line.upper().strip()

        if not line:
            continue

        # OCR may insert spaces
        line = line.replace(
            " ",
            "<",
        )

        # Keep MRZ characters
        line = re.sub(
            r"[^A-Z0-9<]",
            "",
            line,
        )

        if len(line) >= 30:

            lines.append(
                line
            )

    # --------------------------------------------------------
    # Standard passport TD3 MRZ
    # --------------------------------------------------------

    for i in range(
        len(lines) - 1
    ):

        line1 = lines[i]
        line2 = lines[i + 1]

        # First line of passport MRZ
        if not line1.startswith(
            "P<"
        ):
            continue

        if (
            len(line1) < 35
            or len(line2) < 35
        ):
            continue

        line1 = line1[:44]
        line2 = line2[:44]

        result["detected"] = True

        result["line1"] = line1
        result["line2"] = line2

        # ----------------------------------------------------
        # Name
        # ----------------------------------------------------

        if len(line1) >= 5:

            name_part = line1[5:]

            parts = name_part.split(
                "<<",
                1,
            )

            if len(parts) == 2:

                surname = (
                    parts[0]
                    .replace(
                        "<",
                        " ",
                    )
                    .strip()
                )

                given = (
                    parts[1]
                    .replace(
                        "<",
                        " ",
                    )
                    .strip()
                )

                if given or surname:

                    result["name"] = (
                        f"{given} {surname}"
                    ).strip()

        # ----------------------------------------------------
        # TD3 line 2
        # ----------------------------------------------------

        if len(line2) >= 27:

            passport_number = (
                line2[0:9]
                .replace(
                    "<",
                    "",
                )
                .strip()
            )

            if passport_number:

                result[
                    "document_number"
                ] = passport_number

            nationality = (
                line2[10:13]
                .replace(
                    "<",
                    "",
                )
                .strip()
            )

            if nationality:

                result[
                    "nationality"
                ] = nationality

            # ------------------------------------------------
            # DOB
            # ------------------------------------------------

            dob_raw = line2[13:19]

            if re.fullmatch(
                r"\d{6}",
                dob_raw,
            ):

                yy = int(
                    dob_raw[0:2]
                )

                mm = int(
                    dob_raw[2:4]
                )

                dd = int(
                    dob_raw[4:6]
                )

                year = (
                    2000 + yy
                    if yy <= 30
                    else 1900 + yy
                )

                try:

                    result[
                        "date_of_birth"
                    ] = date(
                        year,
                        mm,
                        dd,
                    ).isoformat()

                except ValueError:
                    pass

            # ------------------------------------------------
            # SEX
            # ------------------------------------------------

            sex = line2[20:21]

            if sex in (
                "M",
                "F",
            ):

                result["sex"] = sex

            # ------------------------------------------------
            # EXPIRY
            # ------------------------------------------------

            expiry_raw = line2[21:27]

            if re.fullmatch(
                r"\d{6}",
                expiry_raw,
            ):

                yy = int(
                    expiry_raw[0:2]
                )

                mm = int(
                    expiry_raw[2:4]
                )

                dd = int(
                    expiry_raw[4:6]
                )

                year = 2000 + yy

                try:

                    result[
                        "expiry_date"
                    ] = date(
                        year,
                        mm,
                        dd,
                    ).isoformat()

                except ValueError:
                    pass

        result["valid"] = bool(
            result["document_number"]
            and result["expiry_date"]
        )

        return result

    return result


# ============================================================
# VISA MRZ / MACHINE READABLE EXTRACTION
# ============================================================

def extract_visa_machine_data(
    text: str,
) -> dict:

    result = {

        "detected": False,

        "line1": None,

        "line2": None,

        "raw_lines": [],

        "document_number": None,

    }

    if not text:
        return result

    raw_lines = text.splitlines()

    candidate_lines = []

    for raw_line in raw_lines:

        line = raw_line.upper().strip()

        if not line:
            continue

        compact = re.sub(
            r"[^A-Z0-9<]",
            "",
            line,
        )

        # Visa MRZ-like lines can be shorter than passport TD3
        if (
            len(compact) >= 25
            and (
                "<" in compact
                or re.search(
                    r"[A-Z0-9]{10,}",
                    compact,
                )
            )
        ):

            candidate_lines.append(
                compact
            )

    # --------------------------------------------------------
    # Look for lines containing many < separators
    # --------------------------------------------------------

    mrz_candidates = [

        line
        for line in candidate_lines

        if line.count("<") >= 3
    ]

    if mrz_candidates:

        result["detected"] = True

        result[
            "raw_lines"
        ] = mrz_candidates[:4]

        result[
            "line1"
        ] = mrz_candidates[0]

        if len(
            mrz_candidates
        ) > 1:

            result[
                "line2"
            ] = mrz_candidates[1]

        # ----------------------------------------------------
        # Try to find alphanumeric document number
        # ----------------------------------------------------

        for line in mrz_candidates:

            parts = line.split("<")

            for part in parts:

                part = part.strip()

                if (
                    5 <= len(part) <= 15
                    and re.search(
                        r"[A-Z]",
                        part,
                    )
                    and re.search(
                        r"\d",
                        part,
                    )
                ):

                    # Don't mistake country codes
                    if len(part) == 3:
                        continue

                    result[
                        "document_number"
                    ] = part

                    return result

    return result


# ============================================================
# VISA FIELD EXTRACTION
# ============================================================

def extract_visa_fields(
    text: str,
) -> dict:

    result = {

        "visa_number": None,

        "valid_from": None,

        "valid_until": None,

        "entries": None,

        "duration_of_stay": None,

        "visa_type": None,

        "issuing_country": None,
    }

    if not text:
        return result

    normalized = normalize_ocr_text(
        text
    )

    # --------------------------------------------------------
    # VISA NUMBER
    # --------------------------------------------------------

    result[
        "visa_number"
    ] = extract_document_number(
        normalized,
        "Visa",
    )

    # --------------------------------------------------------
    # VALID FROM
    # --------------------------------------------------------

    valid_from = extract_labeled_date(
        normalized,
        [
            "valid from",
            "valid from:",
            "from",
            "start date",
            "date of issue",
            "issued",
            "date of validity",
        ],
    )

    if valid_from:

        result[
            "valid_from"
        ] = valid_from.isoformat()

    # --------------------------------------------------------
    # VALID UNTIL
    # --------------------------------------------------------

    valid_until = extract_labeled_date(
        normalized,
        [
            "valid until",
            "valid until:",
            "valid till",
            "valid till:",
            "valid through",
            "valid through:",
            "until",
            "expiry",
            "expiration",
            "date of expiry",
            "date of expiration",
        ],
    )

    if valid_until:

        result[
            "valid_until"
        ] = valid_until.isoformat()

    # --------------------------------------------------------
    # ENTRIES
    # --------------------------------------------------------

    entry_match = re.search(
        r"(?:entries|entry)"
        r"\s*[:\-]?\s*"
        r"(1|2|3|MULT|MULTIPLE|01|02|03)",
        normalized,
        re.IGNORECASE,
    )

    if entry_match:

        result[
            "entries"
        ] = entry_match.group(1).upper()

    # --------------------------------------------------------
    # DURATION OF STAY
    # --------------------------------------------------------

    duration_match = re.search(
        r"(?:duration\s+of\s+stay|stay)"
        r"\s*[:\-]?\s*"
        r"(\d{1,3})\s*(?:DAYS?|JOURS?)?",
        normalized,
        re.IGNORECASE,
    )

    if duration_match:

        result[
            "duration_of_stay"
        ] = duration_match.group(1)

    # --------------------------------------------------------
    # VISA TYPE
    # --------------------------------------------------------

    type_match = re.search(
        r"(?:visa\s+type|type\s+of\s+visa|type)"
        r"\s*[:\-]?\s*"
        r"([A-Z]{1,5})\b",
        normalized,
        re.IGNORECASE,
    )

    if type_match:

        result[
            "visa_type"
        ] = type_match.group(1).upper()

    return result


# ============================================================
# VISA EXPIRY FALLBACK
# ============================================================

def find_best_visa_expiry(
    text: str,
    visa_fields: dict,
) -> Optional[date]:

    # First choice:
    # explicitly labeled valid until / expiry

    if visa_fields.get(
        "valid_until"
    ):

        try:

            return date.fromisoformat(
                visa_fields[
                    "valid_until"
                ]
            )

        except (
            ValueError,
            TypeError,
        ):
            pass

    # --------------------------------------------------------
    # Fallback: search labels again
    # --------------------------------------------------------

    labeled = extract_labeled_date(
        text,
        [
            "valid until",
            "valid till",
            "valid through",
            "expiry",
            "expiration",
            "date of expiry",
        ],
    )

    if labeled:

        return labeled

    # --------------------------------------------------------
    # Last fallback:
    # use the latest future date.
    #
    # This is intentionally NOT the first date.
    # It prevents DOB / issue dates from being used as
    # expiry when OCR does not preserve labels.
    # --------------------------------------------------------

    parsed_dates = extract_parsed_dates(
        text
    )

    if not parsed_dates:
        return None

    today = date.today()

    future_dates = [
        item
        for item in parsed_dates
        if item >= today
    ]

    if future_dates:

        return max(
            future_dates
        )

    return max(
        parsed_dates
    )


# ============================================================
# VISUAL ANALYSIS
# ============================================================

def analyze_document_image(
    file_bytes: bytes,
) -> dict:

    result = {

        "is_image": False,

        "width": None,

        "height": None,

        "aspect_ratio": None,

        "blur_score": None,

        "brightness": None,

        "contrast": None,

        "quality_score": 100,

        "visual_issues": [],

        "visual_checks": [],
    }

    image_array = np.frombuffer(
        file_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:

        result[
            "visual_checks"
        ].append(
            "Visual analysis skipped for non-image document"
        )

        result[
            "quality_score"
        ] = 80

        return result

    result[
        "is_image"
    ] = True

    height, width = image.shape[:2]

    result["width"] = width
    result["height"] = height

    result[
        "aspect_ratio"
    ] = round(
        width / height,
        3,
    )

    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    if width < 500 or height < 500:

        result[
            "visual_issues"
        ].append(
            "Low image resolution"
        )

    else:

        result[
            "visual_checks"
        ].append(
            "Image resolution is sufficient"
        )

    # --------------------------------------------------------
    # Grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # --------------------------------------------------------
    # Blur
    # --------------------------------------------------------

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    ).var()

    result[
        "blur_score"
    ] = round(
        float(blur_score),
        2,
    )

    if blur_score < 50:

        result[
            "visual_issues"
        ].append(
            "Image appears blurry"
        )

    else:

        result[
            "visual_checks"
        ].append(
            "Image sharpness is acceptable"
        )

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    brightness = float(
        np.mean(gray)
    )

    result[
        "brightness"
    ] = round(
        brightness,
        2,
    )

    if brightness < 40:

        result[
            "visual_issues"
        ].append(
            "Image is too dark"
        )

    elif brightness > 230:

        result[
            "visual_issues"
        ].append(
            "Image is excessively bright"
        )

    else:

        result[
            "visual_checks"
        ].append(
            "Image brightness is acceptable"
        )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    contrast = float(
        np.std(gray)
    )

    result[
        "contrast"
    ] = round(
        contrast,
        2,
    )

    if contrast < 20:

        result[
            "visual_issues"
        ].append(
            "Image has very low contrast"
        )

    else:

        result[
            "visual_checks"
        ].append(
            "Image contrast is acceptable"
        )

    # --------------------------------------------------------
    # Quality score
    # --------------------------------------------------------

    quality_score = 100

    quality_score -= (
        len(
            result[
                "visual_issues"
            ]
        )
        * 15
    )

    result[
        "quality_score"
    ] = max(
        0,
        quality_score,
    )

    return result


# ============================================================
# OCR CONFIDENCE
# ============================================================

def estimate_ocr_confidence(
    text: str,
) -> float:

    if not text:
        return 0.0

    length = len(
        text.strip()
    )

    if length < 20:
        return 0.45

    if length < 50:
        return 0.60

    if length < 100:
        return 0.72

    return 0.85


# ============================================================
# VALIDATION
# ============================================================

def run_validation(
    doc_type: str,
    text: str,
    ocr_confidence: float,
    mrz: Optional[dict] = None,
):

    issues = []

    verified = []

    score = 0

    mrz = mrz or {}

    normalized_text = normalize_ocr_text(
        text
    )

    # ========================================================
    # NAME
    # ========================================================

    name = guess_name(
        normalized_text
    )

    if (
        doc_type == "Passport"
        and mrz.get("name")
    ):

        name = mrz[
            "name"
        ]

        verified.append(
            "Passport name detected from MRZ"
        )

    elif name:

        verified.append(
            "Name information was detected"
        )

    else:

        issues.append({

            "title":
                "Name not detected",

            "severity":
                "MEDIUM",

            "description":
                "The OCR could not confidently locate the document name.",
        })

        score += WEIGHTS[
            "MISSING_REQUIRED_FIELD"
        ]

    # ========================================================
    # DOCUMENT NUMBER
    # ========================================================

    doc_number = extract_document_number(
        normalized_text,
        doc_type,
    )

    # Passport MRZ fallback
    if (
        not doc_number
        and doc_type == "Passport"
        and mrz.get(
            "document_number"
        )
    ):

        doc_number = mrz[
            "document_number"
        ]

        verified.append(
            "Passport number recovered from MRZ"
        )

    # Visa machine-readable fallback
    if (
        not doc_number
        and doc_type == "Visa"
    ):

        visa_machine = (
            extract_visa_machine_data(
                normalized_text
            )
        )

        if visa_machine.get(
            "document_number"
        ):

            doc_number = (
                visa_machine[
                    "document_number"
                ]
            )

            verified.append(
                "Visa number recovered from machine-readable data"
            )

    if doc_number:

        verified.append(
            f"{doc_type} number format looks valid"
        )

    elif doc_type in (
        "Aadhaar",
        "PAN Card",
        "Passport",
        "Driving Licence",
        "Visa",
    ):

        issues.append({

            "title":
                "Document number not detected",

            "severity":
                severity_for_weight(
                    WEIGHTS[
                        "INVALID_NUMBER_FORMAT"
                    ]
                ),

            "description":
                f"No valid {doc_type} number could be extracted from the document.",
        })

        score += WEIGHTS[
            "INVALID_NUMBER_FORMAT"
        ]

    # ========================================================
    # DATES
    # ========================================================

    dates_found = extract_dates(
        normalized_text
    )

    parsed_dates = []

    for value in dates_found:

        parsed = to_date(
            value
        )

        if parsed:

            parsed_dates.append(
                parsed
            )

    # Passport MRZ dates
    if doc_type == "Passport":

        for mrz_key in (
            "date_of_birth",
            "expiry_date",
        ):

            if mrz.get(mrz_key):

                try:

                    parsed_dates.append(
                        date.fromisoformat(
                            mrz[
                                mrz_key
                            ]
                        )
                    )

                except (
                    ValueError,
                    TypeError,
                ):

                    pass

    parsed_dates = list(
        dict.fromkeys(
            parsed_dates
        )
    )

    if parsed_dates:

        verified.append(
            f"{len(parsed_dates)} valid date(s) extracted"
        )

    elif doc_type in (
        "Passport",
        "Visa",
        "Driving Licence",
        "Permit",
    ):

        issues.append({

            "title":
                "Date could not be verified",

            "severity":
                "MEDIUM",

            "description":
                "No supported date format could be confidently converted to a calendar date.",
        })

        score += WEIGHTS[
            "MISSING_REQUIRED_FIELD"
        ]

    # ========================================================
    # VISA VALIDATION
    # ========================================================

    visa_fields = {}

    if doc_type == "Visa":

        visa_fields = extract_visa_fields(
            normalized_text
        )

        # ----------------------------------------------------
        # If labeled valid-from date found
        # ----------------------------------------------------

        if visa_fields.get(
            "valid_from"
        ):

            verified.append(
                f"Visa valid-from date detected: "
                f"{visa_fields['valid_from']}"
            )

        # ----------------------------------------------------
        # Expiry
        # ----------------------------------------------------

        visa_expiry = (
            find_best_visa_expiry(
                normalized_text,
                visa_fields,
            )
        )

        if visa_expiry:

            visa_fields[
                "valid_until"
            ] = visa_expiry.isoformat()

            if visa_expiry < date.today():

                issues.append({

                    "title":
                        "Visa appears expired",

                    "severity":
                        "HIGH",

                    "description":
                        f"Detected visa expiry date "
                        f"{visa_expiry.isoformat()} "
                        f"is earlier than today's date.",
                })

                score += WEIGHTS[
                    "EXPIRED_DOCUMENT"
                ]

            else:

                verified.append(
                    f"Visa expiry date "
                    f"{visa_expiry.isoformat()} "
                    f"is valid"
                )

        else:

            issues.append({

                "title":
                    "Visa expiry date not detected",

                "severity":
                    "MEDIUM",

                "description":
                    "The visa was detected, but a valid-until or expiry date could not be confidently extracted.",
            })

            score += WEIGHTS[
                "MISSING_REQUIRED_FIELD"
            ]

    # ========================================================
    # PASSPORT EXPIRY
    # ========================================================

    if doc_type == "Passport":

        expiry_candidate = None

        if mrz.get(
            "expiry_date"
        ):

            try:

                expiry_candidate = date.fromisoformat(
                    mrz[
                        "expiry_date"
                    ]
                )

            except (
                ValueError,
                TypeError,
            ):

                pass

        # Prefer labeled expiry date
        if expiry_candidate is None:

            expiry_candidate = extract_labeled_date(
                normalized_text,
                [
                    "date of expiry",
                    "expiry date",
                    "expiration date",
                    "date of expiration",
                ],
            )

        # Last fallback
        if (
            expiry_candidate is None
            and parsed_dates
        ):

            future_dates = [
                item
                for item in parsed_dates
                if item >= date.today()
            ]

            if future_dates:

                expiry_candidate = max(
                    future_dates
                )

        if expiry_candidate:

            if expiry_candidate < date.today():

                issues.append({

                    "title":
                        "Passport appears expired",

                    "severity":
                        "HIGH",

                    "description":
                        f"Detected expiry date "
                        f"{expiry_candidate.isoformat()} "
                        f"is earlier than today's date.",
                })

                score += WEIGHTS[
                    "EXPIRED_DOCUMENT"
                ]

            else:

                verified.append(
                    f"Passport expiry date "
                    f"{expiry_candidate.isoformat()} "
                    f"is valid"
                )

    # ========================================================
    # OTHER DOCUMENT EXPIRY
    # ========================================================

    elif doc_type in (
        "Driving Licence",
        "Permit",
    ):

        expiry_candidate = extract_labeled_date(
            normalized_text,
            [
                "valid until",
                "valid till",
                "expiry",
                "expiry date",
                "date of expiry",
                "expiration",
            ],
        )

        if expiry_candidate is None:

            future_dates = [
                item
                for item in parsed_dates
                if item >= date.today()
            ]

            if future_dates:

                expiry_candidate = max(
                    future_dates
                )

        if expiry_candidate:

            if expiry_candidate < date.today():

                issues.append({

                    "title":
                        "Document may be expired",

                    "severity":
                        "MEDIUM",

                    "description":
                        f"Detected expiry date "
                        f"{expiry_candidate.isoformat()} "
                        f"is earlier than today's date.",
                })

                score += WEIGHTS[
                    "EXPIRED_DOCUMENT"
                ]

            else:

                verified.append(
                    f"Document expiry date "
                    f"{expiry_candidate.isoformat()} "
                    f"is valid"
                )

    # ========================================================
    # MRZ VALIDATION
    # ========================================================

    if doc_type == "Passport":

        if mrz.get(
            "detected"
        ):

            verified.append(
                "Passport MRZ detected"
            )

            mrz_number = mrz.get(
                "document_number"
            )

            if (
                doc_number
                and mrz_number
            ):

                visible_normalized = re.sub(
                    r"[^A-Z0-9]",
                    "",
                    doc_number.upper(),
                )

                mrz_normalized = re.sub(
                    r"[^A-Z0-9]",
                    "",
                    mrz_number.upper(),
                )

                if (
                    visible_normalized
                    == mrz_normalized
                ):

                    verified.append(
                        "Passport number matches MRZ"
                    )

                else:

                    issues.append({

                        "title":
                            "MRZ passport number mismatch",

                        "severity":
                            "HIGH",

                        "description":
                            (
                                f"Visible passport number "
                                f"'{visible_normalized}' "
                                f"does not match MRZ "
                                f"'{mrz_normalized}'."
                            ),
                    })

                    score += WEIGHTS[
                        "CRITICAL_MISMATCH"
                    ]

            mrz_expiry = mrz.get(
                "expiry_date"
            )

            if mrz_expiry:

                try:

                    mrz_expiry_date = date.fromisoformat(
                        mrz_expiry
                    )

                    if (
                        mrz_expiry_date
                        < date.today()
                    ):

                        issues.append({

                            "title":
                                "MRZ indicates expired passport",

                            "severity":
                                "HIGH",

                            "description":
                                f"MRZ expiry date is "
                                f"{mrz_expiry}.",
                        })

                        score += WEIGHTS[
                            "EXPIRED_DOCUMENT"
                        ]

                    else:

                        verified.append(
                            "MRZ expiry date is valid"
                        )

                except ValueError:

                    issues.append({

                        "title":
                            "Invalid MRZ expiry date",

                        "severity":
                            "MEDIUM",

                        "description":
                            "The passport MRZ was detected but its expiry date could not be parsed.",
                    })

        else:

            verified.append(
                "MRZ not detected; visible passport fields used"
            )

    # ========================================================
    # OCR CONFIDENCE
    # ========================================================

    if ocr_confidence < 0.60:

        issues.append({

            "title":
                "Low OCR confidence",

            "severity":
                "MEDIUM",

            "description":
                "OCR quality appears low. Manual verification is recommended.",
        })

        score += WEIGHTS[
            "LOW_OCR_CONFIDENCE"
        ]

    else:

        verified.append(
            "OCR extraction completed successfully"
        )

    # ========================================================
    # UNKNOWN DOCUMENT
    # ========================================================

    if doc_type == "Unknown":

        issues.append({

            "title":
                "Document type could not be identified",

            "severity":
                "HIGH",

            "description":
                "No supported document pattern was confidently detected.",
        })

        score += WEIGHTS[
            "CRITICAL_MISMATCH"
        ]

    return (
        issues,
        verified,
        score,
        name,
        doc_number,
        visa_fields,
    )


# ============================================================
# OCR.SPACE
# ============================================================

def perform_ocr(
    file_bytes: bytes,
    filename: str,
) -> str:

    if not OCR_SPACE_API_KEY:

        raise HTTPException(

            status_code=500,

            detail=(
                "OCR_SPACE_API_KEY is missing from .env"
            ),
        )

    url = (
        "https://api.ocr.space/parse/image"
    )

    files = {

        "file": (
            filename,
            file_bytes,
        )
    }

    data = {

        "apikey":
            OCR_SPACE_API_KEY,

        "language":
            "eng",

        "isOverlayRequired":
            "false",

        "OCREngine":
            "2",

        "detectOrientation":
            "true",

        "scale":
            "true",

        "isTable":
            "false",

        "isSearchablePdfHideTextLayer":
            "true",
    }

    try:

        response = requests.post(

            url,

            files=files,

            data=data,

            timeout=60,
        )

        response.raise_for_status()

        result = response.json()

    except requests.RequestException as error:

        raise HTTPException(

            status_code=502,

            detail=(
                f"OCR service error: {str(error)}"
            ),
        )

    if result.get(
        "IsErroredOnProcessing"
    ):

        error_message = (

            result.get(
                "ErrorMessage"
            )

            or result.get(
                "ErrorDetails"
            )

            or "OCR processing failed"
        )

        raise HTTPException(

            status_code=502,

            detail=str(
                error_message
            ),
        )

    parsed_results = result.get(
        "ParsedResults",
        [],
    )

    if not parsed_results:

        return ""

    extracted = []

    for item in parsed_results:

        parsed_text = item.get(
            "ParsedText",
            "",
        )

        if parsed_text:

            extracted.append(
                parsed_text
            )

    return normalize_ocr_text(
        "\n".join(
            extracted
        )
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "message":
            "BorderShield AI Backend is Running!",

        "status":
            "ONLINE",

        "version":
            "2.0.0",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {

        "success":
            True,

        "status":
            "ONLINE",

        "service":
            "BorderShield AI",

        "version":
            "2.0.0",
    }


# ============================================================
# SCREEN DOCUMENT
# ============================================================

@app.post("/api/screen")
async def screen_document(

    file: UploadFile = File(...),

    expected_document_type: str = Form(...),

):

    # ========================================================
    # FILE TYPE
    # ========================================================

    if file.content_type not in ALLOWED_CONTENT_TYPES:

        raise HTTPException(

            status_code=400,

            detail=(
                f"Unsupported file type: "
                f"{file.content_type}"
            ),
        )

    # ========================================================
    # READ FILE
    # ========================================================

    file_bytes = await file.read()

    if not file_bytes:

        raise HTTPException(

            status_code=400,

            detail="Uploaded file is empty.",
        )

    # ========================================================
    # FILE SIZE
    # ========================================================

    size_mb = (
        len(file_bytes)
        / (1024 * 1024)
    )

    if size_mb > MAX_FILE_SIZE_MB:

        raise HTTPException(

            status_code=400,

            detail=(
                f"File too large "
                f"({size_mb:.1f} MB). "
                f"Maximum is "
                f"{MAX_FILE_SIZE_MB} MB."
            ),
        )

    # ========================================================
    # VISUAL ANALYSIS
    # ========================================================

    visual_analysis = (
        analyze_document_image(
            file_bytes
        )
    )

    # ========================================================
    # OCR
    # ========================================================

    text = perform_ocr(

        file_bytes,

        file.filename
        or "document",
    )

    text = normalize_ocr_text(
        text
    )

    print(
        "\n================================================"
    )

    print(
        "OCR TEXT:"
    )

    print(
        text
    )

    print(
        "================================================\n"
    )

    # ========================================================
    # NO TEXT
    # ========================================================

    if not text.strip():

        score = 80

        status = status_for_score(
            score
        )

        issues = [

            {

                "title":
                    "No text detected",

                "severity":
                    "HIGH",

                "description":
                    "No readable text could be extracted.",
            }

        ]

        screening_id = None

        try:

            screening_id = save_screening(

                identity_id=None,

                full_name="Unknown",

                document_number="Not detected",

                document_type="Unknown",

                risk_score=score,

                status=status,

                issues=issues,

                checkpoint=(
                    "Main Border Checkpoint"
                ),

                officer_name=(
                    "BorderShield AI"
                ),
            )

            print(
                "SCREENING SAVED SUCCESSFULLY!",
                f"ID = {screening_id}",
            )

        except Exception as error:

            print(
                "DATABASE SAVE ERROR:",
                error,
            )

        return {

            "success":
                True,

            "screening_id":
                screening_id,

            "document_type":
                "Unknown",

            "risk_score":
                score,

            "status":
                status,

            "issues":
                issues,

            "verified_checks":
                [],

            "extracted_fields": {

                "name_guess":
                    None,

                "document_number":
                    None,

                "dates_found":
                    [],

                "mrz":
                    {},

                "visa":
                    {},
            },

            "visual_analysis":
                visual_analysis,
        }

    # ========================================================
    # OCR CONFIDENCE
    # ========================================================

    ocr_confidence = (
        estimate_ocr_confidence(
            text
        )
    )

    # ========================================================
    # DOCUMENT TYPE
    # ========================================================

    doc_type = classify_document(
        text
    )

    print(
        "DETECTED DOCUMENT TYPE:",
        doc_type,
    )

    # ========================================================
    # PASSPORT MRZ
    # ========================================================

    mrz = extract_mrz(
        text
    )

    # ========================================================
    # VISA MACHINE DATA
    # ========================================================

    visa_machine = (
        extract_visa_machine_data(
            text
        )
    )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    aliases = {

        "driving license":
            "driving licence",

        "driving licence":
            "driving licence",

        "passport":
            "passport",

        "visa":
            "visa",

        "national id":
            "national id",

        "permit":
            "permit",

        "aadhaar":
            "aadhaar",

        "pan card":
            "pan card",
    }

    expected = (
        expected_document_type
        .strip()
        .lower()
    )

    detected = (
        doc_type
        .strip()
        .lower()
    )

    expected_normalized = aliases.get(
        expected,
        expected,
    )

    detected_normalized = aliases.get(
        detected,
        detected,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    (
        issues,
        verified,
        score,
        name,
        doc_number,
        visa_fields,
    ) = run_validation(

        doc_type,

        text,

        ocr_confidence,

        mrz,
    )

    # ========================================================
    # VISA NAME FALLBACK
    # ========================================================

    if (
        doc_type == "Visa"
        and not name
    ):

        # Try simple name patterns from OCR
        name_patterns = [

            r"(?:surname|family\s+name)"
            r"\s*[:\-]?\s*"
            r"([A-Z][A-Z\s]{2,40})",

            r"(?:given\s+name|given\s+names)"
            r"\s*[:\-]?\s*"
            r"([A-Z][A-Z\s]{2,40})",
        ]

        for pattern in name_patterns:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE,
            )

            if match:

                candidate = (
                    match.group(1)
                    .strip()
                )

                candidate = re.sub(
                    r"\s+",
                    " ",
                    candidate,
                )

                if len(candidate) >= 3:

                    name = candidate

                    verified.append(
                        "Name recovered from labeled visa field"
                    )

                    break

    # ========================================================
    # TYPE MISMATCH
    # ========================================================

    if (
        expected_normalized
        != detected_normalized
    ):

        issues.append({

            "title":
                "Document Type Mismatch",

            "severity":
                "HIGH",

            "description":
                (
                    f"Expected "
                    f"'{expected_document_type}', "
                    f"but detected "
                    f"'{doc_type}'."
                ),
        })

        score += WEIGHTS[
            "CRITICAL_MISMATCH"
        ]

    else:

        verified.append(
            "Correct document type confirmed: "
            f"{expected_document_type}"
        )

    # ========================================================
    # VISUAL ISSUES
    # ========================================================

    for visual_issue in (
        visual_analysis[
            "visual_issues"
        ]
    ):

        issues.append({

            "title":
                "Visual Document Issue",

            "severity":
                "MEDIUM",

            "description":
                visual_issue,
        })

        score += WEIGHTS[
            "VISUAL_ISSUE"
        ]

    # ========================================================
    # VISUAL CHECKS
    # ========================================================

    for visual_check in (
        visual_analysis[
            "visual_checks"
        ]
    ):

        verified.append(
            f"Visual check: "
            f"{visual_check}"
        )

    # ========================================================
    # LIMIT SCORE
    # ========================================================

    score = max(
        0,
        min(
            100,
            score,
        ),
    )

    status = status_for_score(
        score
    )

    # ========================================================
    # CREATE IDENTITY
    # ========================================================

    identity_id = None

    try:

        if doc_number:

            identity_id = (
                create_or_get_identity(

                    full_name=(
                        name
                        or "Unknown"
                    ),

                    document_number=(
                        doc_number
                    ),
                )
            )

    except Exception as error:

        print(
            "DATABASE IDENTITY ERROR:",
            error,
        )

    # ========================================================
    # SAVE SCREENING
    # ========================================================

    screening_id = None

    try:

        screening_id = save_screening(

            identity_id=identity_id,

            full_name=(
                name
                or "Unknown"
            ),

            document_number=(
                doc_number
                or "Not detected"
            ),

            document_type=doc_type,

            risk_score=score,

            status=status,

            issues=issues,

            checkpoint=(
                "Main Border Checkpoint"
            ),

            officer_name=(
                "BorderShield AI"
            ),
        )

        print(
            "SCREENING SAVED SUCCESSFULLY!",
            f"ID = {screening_id}",
        )

    except Exception as error:

        print(
            "DATABASE SAVE ERROR:",
            error,
        )

    # ========================================================
    # AUDIT LOG
    # ========================================================

    try:

        add_audit_log(

            action=(
                "SCREENING_COMPLETED"
            ),

            details=(

                f"Screening ID: "
                f"{screening_id}, "

                f"Document: "
                f"{doc_type}, "

                f"Risk Score: "
                f"{score}, "

                f"Status: "
                f"{status}"
            ),
        )

    except Exception as error:

        print(
            "AUDIT LOG ERROR:",
            error,
        )

    # ========================================================
    # IDENTITY INTELLIGENCE
    # ========================================================

    identity_intelligence = None

    if identity_id:

        try:

            identity_intelligence = (
                get_identity_intelligence(
                    identity_id
                )
            )

        except Exception as error:

            print(
                "IDENTITY INTELLIGENCE ERROR:",
                error,
            )

    # ========================================================
    # ALL DATES
    # ========================================================

    dates_found = extract_dates(
        text
    )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "success":
            True,

        "screening_id":
            screening_id,

        "identity_id":
            identity_id,

        "document_type":
            doc_type,

        "expected_document_type":
            expected_document_type,

        "risk_score":
            score,

        "status":
            status,

        "message":
            "Document received and analyzed successfully.",

        "issues":
            issues,

        "verified_checks":
            verified,

        "extracted_fields": {

            "name_guess":
                name,

            "document_number":
                doc_number,

            "dates_found":
                dates_found,

            "mrz":
                mrz,

            "visa":
                visa_fields,

            "visa_machine_data":
                visa_machine,

            "ocr_confidence":
                ocr_confidence,

            "raw_ocr_text":
                text,
        },

        "visual_analysis":
            visual_analysis,

        "identity_intelligence":
            identity_intelligence,
    }


# ============================================================
# DASHBOARD API
# ============================================================

@app.get("/api/dashboard")
def get_dashboard():

    conn = sqlite3.connect(
        "bordershield.db"
    )

    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    try:

        # ====================================================
        # TOTAL SCREENINGS
        # ====================================================

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM screenings
            """
        )

        total_screenings = (
            cursor.fetchone()["total"]
        )

        # ====================================================
        # AVERAGE RISK SCORE
        # ====================================================

        cursor.execute(
            """
            SELECT AVG(risk_score) AS average
            FROM screenings
            """
        )

        result = cursor.fetchone()

        average_risk_score = (

            round(
                result["average"],
                1,
            )

            if result["average"]
            is not None

            else 0
        )

        # ====================================================
        # CLEARED
        # ====================================================

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM screenings
            WHERE status = 'LOW RISK'
            """
        )

        cleared_documents = (
            cursor.fetchone()["total"]
        )

        # ====================================================
        # FLAGGED
        # ====================================================

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM screenings
            WHERE status != 'LOW RISK'
            """
        )

        flagged_documents = (
            cursor.fetchone()["total"]
        )

        # ====================================================
        # LOW RISK
        # ====================================================

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM screenings
            WHERE status = 'LOW RISK'
            """
        )

        low_risk = (
            cursor.fetchone()["total"]
        )

        # ====================================================
        # REVIEW REQUIRED
        # ====================================================

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM screenings
            WHERE status = 'REVIEW REQUIRED'
            """
        )

        review_required = (
            cursor.fetchone()["total"]
        )

        # ====================================================
        # SUSPICIOUS
        # ====================================================

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM screenings
            WHERE status = 'SUSPICIOUS'
            """
        )

        suspicious = (
            cursor.fetchone()["total"]
        )

        # ====================================================
        # HIGH RISK
        # ====================================================

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM screenings
            WHERE status = 'HIGH RISK'
            """
        )

        high_risk = (
            cursor.fetchone()["total"]
        )

        # ====================================================
        # RECENT SCREENINGS
        # ====================================================

        cursor.execute(
            """
            SELECT
                id,
                identity_id,
                full_name,
                document_number,
                document_type,
                risk_score,
                status,
                issues,
                checkpoint,
                officer_name,
                created_at
            FROM screenings
            ORDER BY id DESC
            LIMIT 10
            """
        )

        recent_screenings = [

            dict(row)

            for row in cursor.fetchall()
        ]

        # ====================================================
        # CHECKPOINT LOAD
        # ====================================================

        cursor.execute(
            """
            SELECT
                checkpoint,
                COUNT(*) AS total_screenings
            FROM screenings
            GROUP BY checkpoint
            ORDER BY total_screenings DESC
            """
        )

        checkpoint_load = [

            dict(row)

            for row in cursor.fetchall()
        ]

        return {

            "success":
                True,

            "statistics": {

                "screenings_today":
                    total_screenings,

                "average_risk_score":
                    average_risk_score,

                "cleared_documents":
                    cleared_documents,

                "flagged_for_review":
                    flagged_documents,
            },

            "recent_screenings":
                recent_screenings,

            "risk_distribution": {

                "low_risk":
                    low_risk,

                "review_required":
                    review_required,

                "suspicious":
                    suspicious,

                "high_risk":
                    high_risk,
            },

            "checkpoint_load":
                checkpoint_load,
        }

    finally:

        conn.close()


# ============================================================
# IDENTITY INTELLIGENCE API
# ============================================================

@app.get(
    "/api/identity/{identity_id}"
)
def identity_intelligence(
    identity_id: int,
):

    try:

        intelligence = (
            get_identity_intelligence(
                identity_id
            )
        )

        if not intelligence:

            raise HTTPException(

                status_code=404,

                detail="Identity not found",
            )

        return {

            "success":
                True,

            "identity_intelligence":
                intelligence,
        }

    except HTTPException:

        raise

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error),
        )


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

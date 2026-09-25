"""
============================================================
TeamTrace / BorderShield AI - Backend
OCR.space + OpenCV + PostgreSQL Identity Intelligence
============================================================

Supported documents:
- Passport
- Visa
- Driving Licence
- National ID
- Permit
- Aadhaar
- PAN Card

Extraction:
- Passport MRZ
- Document numbers
- Names
- Dates
- Expiry dates
- Nationality
- Sex

Database:
- PostgreSQL / Neon
============================================================
"""

import os
import re
from datetime import date, datetime
from typing import Optional

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
    get_connection,
)


# ============================================================
# SETUP
# ============================================================

load_dotenv()

OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY")

if not OCR_SPACE_API_KEY:
    print("WARNING: OCR_SPACE_API_KEY is not set")


app = FastAPI(
    title="BorderShield AI Backend",
    version="2.0.0",
)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

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
# DOCUMENT NUMBER PATTERNS
# ============================================================

AADHAAR_NUMBER_RE = re.compile(
    r"\b\d{4}\s?\d{4}\s?\d{4}\b"
)

PAN_NUMBER_RE = re.compile(
    r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    re.IGNORECASE,
)

PASSPORT_NUMBER_RE = re.compile(
    r"\b[A-Z][0-9]{7}\b",
    re.IGNORECASE,
)

DL_NUMBER_RE = re.compile(
    r"\b[A-Z]{2}[-\s]?\d{2}[-\s]?\d{4,11}\b",
    re.IGNORECASE,
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
    ],

    "PAN Card": [
        "income tax department",
        "permanent account number",
        "pan card",
        "income tax",
    ],

    "Passport": [
        "passport",
        "paszport",
        "republic",
        "nationality",
        "surname",
        "given name",
        "place of birth",
        "date of issue",
        "date of expiry",
        "type",
    ],

    "Visa": [
        "visa",
        "visa number",
        "entry",
        "valid from",
        "valid until",
        "validity",
        "entries",
        "visa type",
        "date of issue",
    ],

    "National ID": [
        "national identity",
        "identity card",
        "national id",
        "identification card",
        "identity number",
    ],

    "Driving Licence": [
        "driving licence",
        "driving license",
        "transport department",
        "licence no",
        "license no",
        "dl no",
        "driving",
    ],

    "Permit": [
        "permit",
        "validity",
        "issued by",
        "permit number",
        "permit no",
        "permission",
    ],
}


# ============================================================
# DATE PARSER
# ============================================================

def to_date(value: Optional[str]) -> Optional[date]:

    if not value:
        return None

    value = value.strip().upper()

    value = value.replace(".", "")

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    # Polish / bilingual passport month cleanup
    month_map = {

        "STY/JAN": "JAN",
        "LUT/FEB": "FEB",
        "MAR/MAR": "MAR",
        "KWI/APR": "APR",
        "MAJ/MAY": "MAY",
        "CZE/JUN": "JUN",
        "LIP/JUL": "JUL",
        "SIE/AUG": "AUG",
        "WRZ/SEP": "SEP",
        "PAZ/OCT": "OCT",
        "LIS/NOV": "NOV",
        "GRU/DEC": "DEC",

    }

    for old, new in month_map.items():
        value = value.replace(old, new)

    formats = [

        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d %b %Y",
        "%d %B %Y",

    ]

    for fmt in formats:

        try:

            return datetime.strptime(
                value,
                fmt,
            ).date()

        except ValueError:
            continue

    return None


# ============================================================
# PASSPORT MRZ EXTRACTION
# ============================================================

def extract_passport_mrz(text: str) -> dict:

    result = {

        "detected": False,

        "document_number": None,

        "nationality": None,

        "date_of_birth": None,

        "sex": None,

        "expiry_date": None,

        "name": None,

        "line1": None,

        "line2": None,

    }

    raw_lines = text.splitlines()

    lines = []

    for raw_line in raw_lines:

        line = raw_line.upper().strip()

        # Remove spaces only for MRZ processing
        line = re.sub(
            r"\s+",
            "",
            line,
        )

        if len(line) >= 30:
            lines.append(line)


    for i in range(len(lines) - 1):

        line1 = lines[i]
        line2 = lines[i + 1]

        # Passport MRZ starts with P<
        if not line1.startswith("P<"):
            continue

        # TD3 passport second line
        if len(line2) < 35:
            continue

        # OCR corrections commonly seen in MRZ
        line1 = (
            line1
            .replace("«", "<")
            .replace("‹", "<")
            .replace(" ", "")
        )

        line2 = (
            line2
            .replace("«", "<")
            .replace("‹", "<")
            .replace(" ", "")
        )

        result["detected"] = True
        result["line1"] = line1
        result["line2"] = line2


        # ----------------------------------------------------
        # DOCUMENT NUMBER
        # ----------------------------------------------------

        passport_number = line2[:9]

        passport_number = (
            passport_number
            .replace("<", "")
            .strip()
        )

        # Sometimes OCR changes O -> 0
        if passport_number:

            result["document_number"] = (
                passport_number.upper()
            )


        # ----------------------------------------------------
        # NATIONALITY
        # ----------------------------------------------------

        if len(line2) >= 13:

            nationality = line2[10:13]

            nationality = (
                nationality
                .replace("<", "")
                .strip()
            )

            if nationality:
                result["nationality"] = nationality


        # ----------------------------------------------------
        # DATE OF BIRTH
        # ----------------------------------------------------

        if len(line2) >= 19:

            dob_raw = line2[13:19]

            if re.fullmatch(
                r"\d{6}",
                dob_raw,
            ):

                yy = int(dob_raw[0:2])
                mm = int(dob_raw[2:4])
                dd = int(dob_raw[4:6])

                # Practical MRZ century handling
                year = (
                    1900 + yy
                    if yy >= 30
                    else 2000 + yy
                )

                try:

                    parsed = date(
                        year,
                        mm,
                        dd,
                    )

                    result["date_of_birth"] = (
                        parsed.isoformat()
                    )

                except ValueError:
                    pass


        # ----------------------------------------------------
        # SEX
        # ----------------------------------------------------

        if len(line2) >= 21:

            sex = line2[20]

            if sex in (
                "M",
                "F",
                "<",
            ):

                result["sex"] = sex


        # ----------------------------------------------------
        # EXPIRY DATE
        # ----------------------------------------------------

        if len(line2) >= 27:

            expiry_raw = line2[21:27]

            if re.fullmatch(
                r"\d{6}",
                expiry_raw,
            ):

                yy = int(expiry_raw[0:2])
                mm = int(expiry_raw[2:4])
                dd = int(expiry_raw[4:6])

                year = 2000 + yy

                try:

                    parsed = date(
                        year,
                        mm,
                        dd,
                    )

                    result["expiry_date"] = (
                        parsed.isoformat()
                    )

                except ValueError:
                    pass


        # ----------------------------------------------------
        # NAME
        # ----------------------------------------------------

        if line1.startswith("P<"):

            # P< + country code = first 5 characters
            name_part = line1[5:]

            name_part = name_part.replace(
                "<",
                " ",
            )

            name_part = re.sub(
                r"\s+",
                " ",
                name_part,
            ).strip()

            if name_part:

                result["name"] = name_part


        return result


    return result


# ============================================================
# DOCUMENT CLASSIFICATION
# ============================================================

def classify_document(text: str) -> str:

    lower = text.lower()

    scores = {}

    for doc_type, keywords in KEYWORDS.items():

        score = 0

        for keyword in keywords:

            if keyword in lower:
                score += 1

        scores[doc_type] = score


    # --------------------------------------------------------
    # PASSPORT MRZ HAS HIGHEST PRIORITY
    # --------------------------------------------------------

    mrz = extract_passport_mrz(text)

    if mrz.get("detected"):

        return "Passport"


    # --------------------------------------------------------
    # AADHAAR
    # --------------------------------------------------------

    if (
        AADHAAR_NUMBER_RE.search(text)
        and scores.get(
            "Aadhaar",
            0,
        ) > 0
    ):

        return "Aadhaar"


    # --------------------------------------------------------
    # PAN
    # --------------------------------------------------------

    if PAN_NUMBER_RE.search(text):

        if (
            scores.get(
                "PAN Card",
                0,
            ) > 0
            or "income tax" in lower
        ):

            return "PAN Card"


    # --------------------------------------------------------
    # PASSPORT
    # --------------------------------------------------------

    if scores.get(
        "Passport",
        0,
    ) >= 2:

        return "Passport"


    # --------------------------------------------------------
    # VISA
    # --------------------------------------------------------

    if scores.get(
        "Visa",
        0,
    ) >= 2:

        return "Visa"


    # --------------------------------------------------------
    # DRIVING LICENCE
    # --------------------------------------------------------

    if (
        DL_NUMBER_RE.search(text)
        and scores.get(
            "Driving Licence",
            0,
        ) > 0
    ):

        return "Driving Licence"


    if scores.get(
        "Driving Licence",
        0,
    ) >= 2:

        return "Driving Licence"


    # --------------------------------------------------------
    # NATIONAL ID
    # --------------------------------------------------------

    if scores.get(
        "National ID",
        0,
    ) >= 1:

        return "National ID"


    # --------------------------------------------------------
    # PERMIT
    # --------------------------------------------------------

    if scores.get(
        "Permit",
        0,
    ) >= 2:

        return "Permit"


    # --------------------------------------------------------
    # BEST MATCH
    # --------------------------------------------------------

    best = max(
        scores,
        key=scores.get,
    )

    if scores[best] > 0:

        return best

    return "Unknown"


# ============================================================
# NAME VALIDATION
# ============================================================

def is_valid_person_name(
    value: str,
) -> bool:

    if not value:
        return False

    value = value.strip()

    if not (
        3 <= len(value) <= 70
    ):
        return False

    if not re.search(
        r"[A-Za-z]",
        value,
    ):
        return False

    upper = value.upper()

    bad_exact = {

        "PASSPORT",
        "PASZPORT",
        "VISA",
        "PERMIT",
        "LICENCE",
        "LICENSE",
        "NATIONALITY",
        "GOVERNMENT",
        "REPUBLIC",
        "IDENTIFICATION",
        "AUTHORITY",
        "TRANSPORT",
        "DEPARTMENT",
        "AADHAAR",
        "PAN CARD",
        "DATE OF BIRTH",
        "DATE OF EXPIRY",
        "EXPIRY DATE",
        "NAME",

    }

    if upper in bad_exact:
        return False

    # Reject lines containing obvious document labels
    bad_words = {

        "PASSPORT",
        "PASZPORT",
        "VISA",
        "PERMIT",
        "AADHAAR",
        "LICENCE",
        "LICENSE",
        "NATIONALITY",
        "GOVERNMENT",
        "IDENTIFICATION",
        "AUTHORITY",
        "EXPIRY",

    }

    words = set(
        upper.split()
    )

    if words & bad_words:
        return False

    # Person name should contain letters, spaces,
    # apostrophes, dots or hyphens.
    if not re.fullmatch(
        r"[A-Za-zÀ-ÖØ-öø-ÿ .'\-]+",
        value,
    ):

        return False

    return True


# ============================================================
# GUESS NAME
# ============================================================

def guess_name(
    text: str,
    doc_type: Optional[str] = None,
) -> Optional[str]:

    # --------------------------------------------------------
    # PASSPORT → MRZ FIRST
    # --------------------------------------------------------

    if doc_type == "Passport":

        mrz = extract_passport_mrz(text)

        if mrz.get("name"):

            return mrz["name"]


    lines = [

        line.strip()

        for line in text.splitlines()

        if line.strip()

    ]


    # --------------------------------------------------------
    # NAME LABELS
    # --------------------------------------------------------

    name_labels = [

        "FULL NAME",
        "FULLNAME",
        "GIVEN NAME",
        "GIVEN NAMES",
        "APPLICANT NAME",
        "HOLDER NAME",
        "PERSON NAME",
        "LICENSE HOLDER",
        "LICENCE HOLDER",
        "SURNAME",

    ]


    # --------------------------------------------------------
    # LABEL + VALUE
    # --------------------------------------------------------

    for i, line in enumerate(lines):

        upper = line.upper().strip()


        # Same line:
        # NAME: JOHN DOE

        for label in name_labels:

            pattern = (
                rf"{re.escape(label)}"
                r"\s*[:\-]\s*(.+)$"
            )

            match = re.search(
                pattern,
                upper,
                re.IGNORECASE,
            )

            if match:

                candidate = (
                    match.group(1)
                    .strip()
                )

                if is_valid_person_name(
                    candidate
                ):

                    return candidate


        # Next line:
        # NAME
        # JOHN DOE

        if any(
            label in upper
            for label in name_labels
        ):

            if i + 1 < len(lines):

                candidate = (
                    lines[i + 1]
                    .strip()
                )

                if is_valid_person_name(
                    candidate
                ):

                    return candidate


    # --------------------------------------------------------
    # GENERAL OCR FALLBACK
    # --------------------------------------------------------

    skip_words = {

        "GOVERNMENT",
        "INDIA",
        "REPUBLIC",
        "POLAND",
        "POLSKA",
        "INCOME",
        "TAX",
        "DEPARTMENT",
        "UNIQUE",
        "IDENTIFICATION",
        "AUTHORITY",
        "TRANSPORT",
        "LICENCE",
        "LICENSE",
        "AADHAAR",
        "PASSPORT",
        "PASZPORT",
        "NATIONALITY",
        "VISA",
        "PERMIT",
        "VALID",
        "EXPIRY",
        "SURNAME",
        "GIVEN",
        "NAME",
        "DATE",
        "BIRTH",
        "SEX",
        "TYPE",
        "NUMBER",
        "NO",
        "ISSUE",
        "ISSUED",

    }


    for line in lines:

        clean = line.strip()

        if not is_valid_person_name(
            clean
        ):
            continue

        words = {
            word.upper()
            for word in clean.split()
        }

        if words & skip_words:
            continue

        if 1 <= len(words) <= 6:

            return clean


    return None


# ============================================================
# DOCUMENT NUMBER EXTRACTION
# ============================================================

def extract_document_number(
    text: str,
    doc_type: str,
) -> Optional[str]:

    upper = text.upper()


    # --------------------------------------------------------
    # PASSPORT
    # --------------------------------------------------------

    if doc_type == "Passport":

        mrz = extract_passport_mrz(text)

        if mrz.get(
            "document_number"
        ):

            return mrz[
                "document_number"
            ]


        match = PASSPORT_NUMBER_RE.search(
            upper
        )

        if match:

            return (
                match.group(0)
                .replace(" ", "")
                .upper()
            )


        patterns = [

            r"PASSPORT\s*(?:NO|NUMBER|N[Oº°]?)"
            r"\s*[:\-]?\s*([A-Z0-9]{6,10})",

            r"(?:NO|NUMBER)"
            r"\s*[:\-]\s*([A-Z][A-Z0-9]{6,9})",

        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                upper,
            )

            if match:
                return match.group(1)

        return None


    # --------------------------------------------------------
    # VISA
    # --------------------------------------------------------

    if doc_type == "Visa":

        patterns = [

            r"VISA\s*(?:NO|NUMBER|#)"
            r"\s*[:\-]?\s*([A-Z0-9\-]{5,25})",

            r"VISA\s*NUMBER"
            r"\s*[:\-]?\s*([A-Z0-9\-]{5,25})",

            r"VISA\s*ID"
            r"\s*[:\-]?\s*([A-Z0-9\-]{5,25})",

            r"DOCUMENT\s*(?:NO|NUMBER)"
            r"\s*[:\-]?\s*([A-Z0-9\-]{5,25})",

            r"CONTROL\s*(?:NO|NUMBER)"
            r"\s*[:\-]?\s*([A-Z0-9\-]{5,25})",

        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                upper,
            )

            if match:

                candidate = (
                    match.group(1)
                    .strip()
                )

                if candidate:
                    return candidate


        # Visa fallback
        for line in upper.splitlines():

            if "VISA" not in line:
                continue

            match = re.search(
                r"\b[A-Z]{1,4}[0-9]{5,15}\b",
                line,
            )

            if match:
                return match.group(0)


        return None


    # --------------------------------------------------------
    # AADHAAR / NATIONAL ID
    # --------------------------------------------------------

    if doc_type in (
        "Aadhaar",
        "National ID",
    ):

        match = AADHAAR_NUMBER_RE.search(
            text
        )

        if match:

            return (
                match.group(0)
                .replace(" ", "")
            )


        patterns = [

            r"(?:ID\s*(?:NO|NUMBER))"
            r"\s*[:\-]?\s*([A-Z0-9\-]{6,25})",

            r"(?:IDENTITY\s*(?:NO|NUMBER))"
            r"\s*[:\-]?\s*([A-Z0-9\-]{6,25})",

            r"(?:NATIONAL\s*ID)"
            r"\s*[:\-]?\s*([A-Z0-9\-]{6,25})",

            r"(?:IDENTIFICATION\s*(?:NO|NUMBER))"
            r"\s*[:\-]?\s*([A-Z0-9\-]{6,25})",

        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                upper,
            )

            if match:

                return match.group(1).upper()


        return None


    # --------------------------------------------------------
    # PAN
    # --------------------------------------------------------

    if doc_type == "PAN Card":

        match = PAN_NUMBER_RE.search(
            upper
        )

        if match:
            return match.group(0).upper()

        return None


    # --------------------------------------------------------
    # DRIVING LICENCE
    # --------------------------------------------------------

    if doc_type == "Driving Licence":

        patterns = [

            r"DL\s*(?:NO|NUMBER)"
            r"\s*[:\-]?\s*([A-Z0-9\-]{8,25})",

            r"LICEN[CS]E\s*(?:NO|NUMBER)"
            r"\s*[:\-]?\s*([A-Z0-9\-]{8,25})",

            r"DRIVING\s*LICEN[CS]E"
            r"\s*(?:NO|NUMBER)"
            r"\s*[:\-]?\s*([A-Z0-9\-]{8,25})",

            r"DRIVING\s*LICEN[CS]E"
            r"\s*[:\-]\s*([A-Z0-9\-]{8,25})",

        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                upper,
            )

            if match:

                return match.group(1).upper()


        match = DL_NUMBER_RE.search(
            upper
        )

        if match:
            return match.group(0).upper()

        return None


    # --------------------------------------------------------
    # PERMIT
    # --------------------------------------------------------

    if doc_type == "Permit":

        patterns = [

            r"PERMIT\s*(?:NO|NUMBER)"
            r"\s*[:\-]?\s*([A-Z0-9\-]{5,25})",

            r"PERMIT\s*ID"
            r"\s*[:\-]?\s*([A-Z0-9\-]{5,25})",

            r"REFERENCE\s*(?:NO|NUMBER)"
            r"\s*[:\-]?\s*([A-Z0-9\-]{5,25})",

            r"APPLICATION\s*(?:NO|NUMBER)"
            r"\s*[:\-]?\s*([A-Z0-9\-]{5,25})",

        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                upper,
            )

            if match:

                return match.group(1).upper()

        return None


    return None


# ============================================================
# DATE EXTRACTION
# ============================================================

def extract_dates(
    text: str,
    doc_type: Optional[str] = None,
) -> list:

    dates = []


    # --------------------------------------------------------
    # NORMAL NUMERIC DATES
    # --------------------------------------------------------

    numeric_dates = re.findall(
        r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
        text,
    )

    dates.extend(
        numeric_dates
    )


    # --------------------------------------------------------
    # ISO DATES
    # --------------------------------------------------------

    iso_dates = re.findall(
        r"\b\d{4}[/-]\d{2}[/-]\d{2}\b",
        text,
    )

    dates.extend(
        iso_dates
    )


    # --------------------------------------------------------
    # MONTH NAME DATES
    #
    # Examples:
    # 23 APR 1984
    # 25 JAN 2033
    # 25 STY/JAN 2033
    # --------------------------------------------------------

    month_dates = re.findall(
        r"\b\d{1,2}\s+"
        r"(?:"
        r"JAN|FEB|MAR|APR|MAY|JUN|"
        r"JUL|AUG|SEP|OCT|NOV|DEC"
        r"|"
        r"STY/JAN|LUT/FEB|KWI/APR|MAJ/MAY|"
        r"CZE/JUN|LIP/JUL|SIE/AUG|WRZ/SEP|"
        r"PAZ/OCT|LIS/NOV|GRU/DEC"
        r")"
        r"\s+\d{4}\b",
        text.upper(),
    )

    dates.extend(
        month_dates
    )


    # --------------------------------------------------------
    # PASSPORT MRZ DATES
    # --------------------------------------------------------

    if doc_type == "Passport":

        mrz = extract_passport_mrz(
            text
        )

        if mrz.get(
            "date_of_birth"
        ):

            dates.append(
                mrz["date_of_birth"]
            )

        if mrz.get(
            "expiry_date"
        ):

            dates.append(
                mrz["expiry_date"]
            )


    # --------------------------------------------------------
    # UNIQUE DATES
    # --------------------------------------------------------

    unique_dates = []

    for value in dates:

        if value not in unique_dates:

            unique_dates.append(
                value
            )

    return unique_dates


# ============================================================
# EXTRACT EXPIRY DATE
# ============================================================

def extract_expiry_date(
    text: str,
    doc_type: str,
) -> Optional[str]:

    # Passport MRZ first
    if doc_type == "Passport":

        mrz = extract_passport_mrz(
            text
        )

        if mrz.get(
            "expiry_date"
        ):

            return mrz[
                "expiry_date"
            ]


    upper = text.upper()


    expiry_labels = [

        "DATE OF EXPIRY",
        "EXPIRY DATE",
        "EXPIRY",
        "VALID UNTIL",
        "VALID TILL",
        "VALID UP TO",
        "VALID THROUGH",
        "VALIDITY",
        "DATE OF VALIDITY",
        "VALID TO",
        "VISA EXPIRY",

    ]


    # Look near expiry labels
    for label in expiry_labels:

        pattern = (
            rf"{re.escape(label)}"
            r".{0,50}?"
            r"("
            r"\d{2}[/-]\d{2}[/-]\d{4}"
            r"|"
            r"\d{4}[/-]\d{2}[/-]\d{2}"
            r"|"
            r"\d{1,2}\s+"
            r"(?:JAN|FEB|MAR|APR|MAY|JUN|"
            r"JUL|AUG|SEP|OCT|NOV|DEC)"
            r"\s+\d{4}"
            r")"
        )

        match = re.search(
            pattern,
            upper,
            re.DOTALL,
        )

        if match:

            return match.group(1)


    return None


# ============================================================
# VISUAL ANALYSIS
# ============================================================

def analyze_document_image(
    file_bytes: bytes,
):

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


    result["is_image"] = True


    height, width = image.shape[:2]

    result["width"] = width
    result["height"] = height

    result["aspect_ratio"] = round(
        width / height,
        3,
    )


    # --------------------------------------------------------
    # RESOLUTION
    # --------------------------------------------------------

    if (
        width < 500
        or height < 500
    ):

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


    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )


    # --------------------------------------------------------
    # BLUR
    # --------------------------------------------------------

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    ).var()

    result["blur_score"] = round(
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
    # BRIGHTNESS
    # --------------------------------------------------------

    brightness = float(
        np.mean(gray)
    )

    result["brightness"] = round(
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
    # CONTRAST
    # --------------------------------------------------------

    contrast = float(
        np.std(gray)
    )

    result["contrast"] = round(
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
# VALIDATION
# ============================================================

def run_validation(
    doc_type: str,
    text: str,
    ocr_confidence: float,
):

    issues = []
    verified = []
    score = 0


    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    name = guess_name(
        text,
        doc_type,
    )


    if name:

        verified.append(
            "A likely name was detected"
        )

    else:

        issues.append({

            "title":
                "Required field missing",

            "severity":
                severity_for_weight(
                    WEIGHTS[
                        "MISSING_REQUIRED_FIELD"
                    ]
                ),

            "description":
                "Could not locate a likely name on the document.",

        })

        score += WEIGHTS[
            "MISSING_REQUIRED_FIELD"
        ]


    # --------------------------------------------------------
    # DOCUMENT NUMBER
    # --------------------------------------------------------

    doc_number = (
        extract_document_number(
            text,
            doc_type,
        )
    )


    if doc_number:

        verified.append(
            f"{doc_type} number was detected"
        )

    else:

        # Some document types can have
        # different number formats.
        if doc_type in (

            "Aadhaar",
            "PAN Card",
            "Passport",
            "Visa",
            "Driving Licence",
            "National ID",
            "Permit",

        ):

            issues.append({

                "title":
                    "Invalid or missing document number",

                "severity":
                    severity_for_weight(
                        WEIGHTS[
                            "INVALID_NUMBER_FORMAT"
                        ]
                    ),

                "description":
                    f"No valid {doc_type} number could be extracted.",

            })

            score += WEIGHTS[
                "INVALID_NUMBER_FORMAT"
            ]


    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------

    dates_found = extract_dates(
        text,
        doc_type,
    )


    if not dates_found:

        issues.append({

            "title":
                "Required field missing",

            "severity":
                severity_for_weight(
                    WEIGHTS[
                        "MISSING_REQUIRED_FIELD"
                    ]
                ),

            "description":
                "No date could be extracted from the document.",

        })

        score += WEIGHTS[
            "MISSING_REQUIRED_FIELD"
        ]

    else:

        verified.append(
            "Date information was extracted"
        )


        # ----------------------------------------------------
        # EXPIRY CHECK
        # ----------------------------------------------------

        expiry_value = (
            extract_expiry_date(
                text,
                doc_type,
            )
        )


        parsed_expiry = None


        if expiry_value:

            parsed_expiry = to_date(
                expiry_value
            )

        else:

            # Fallback:
            # use latest parsed date
            parsed_dates = []

            for value in dates_found:

                parsed = to_date(
                    value
                )

                if parsed:
                    parsed_dates.append(
                        parsed
                    )

            if parsed_dates:

                parsed_expiry = max(
                    parsed_dates
                )


        if parsed_expiry:

            if (
                parsed_expiry
                < date.today()
            ):

                issues.append({

                    "title":
                        "Possible expired document",

                    "severity":
                        "MEDIUM",

                    "description":
                        f"Document appears to have expired on {parsed_expiry.isoformat()}.",

                })

                score += WEIGHTS[
                    "EXPIRED_DOCUMENT"
                ]

            else:

                verified.append(
                    f"Document expiry date is valid: {parsed_expiry.isoformat()}"
                )


    # --------------------------------------------------------
    # OCR CONFIDENCE
    # --------------------------------------------------------

    if ocr_confidence < 0.60:

        issues.append({

            "title":
                "Low OCR confidence",

            "severity":
                "MEDIUM",

            "description":
                "OCR quality appears low.",

        })

        score += WEIGHTS[
            "LOW_OCR_CONFIDENCE"
        ]

    else:

        verified.append(
            "OCR extraction completed successfully"
        )


    # --------------------------------------------------------
    # UNKNOWN DOCUMENT
    # --------------------------------------------------------

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
        dates_found,
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
                "OCR_SPACE_API_KEY is missing "
                "from environment variables."
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

        "scale":
            "true",

        "detectOrientation":
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


    extracted_text = "\n".join(

        item.get(
            "ParsedText",
            ""
        )

        for item in parsed_results

    )


    return extracted_text


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

    }


# ============================================================
# DATABASE HEALTH
# ============================================================

@app.get("/api/database-health")
def database_health():

    conn = None

    cursor = None

    try:

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute(
            "SELECT 1 AS ok"
        )

        result = cursor.fetchone()

        if result and result["ok"] == 1:

            return {

                "success":
                    True,

                "database":
                    "PostgreSQL",

                "connected":
                    True,

            }


        return {

            "success":
                False,

            "database":
                "PostgreSQL",

            "connected":
                False,

            "error":
                "Database test query failed",

        }


    except Exception as error:

        return {

            "success":
                False,

            "database":
                "PostgreSQL",

            "connected":
                False,

            "error":
                str(error),

        }


    finally:

        if cursor:
            cursor.close()

        if conn:
            conn.close()


# ============================================================
# SCREEN DOCUMENT
# ============================================================

@app.post("/api/screen")
async def screen_document(

    file: UploadFile = File(...),

    expected_document_type: str = Form(
        "Unknown"
    ),

):

    # --------------------------------------------------------
    # FILE TYPE
    # --------------------------------------------------------

    if file.content_type not in (
        ALLOWED_CONTENT_TYPES
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                "Unsupported file type. "
                "Use JPG, PNG or PDF."
            ),

        )


    # --------------------------------------------------------
    # READ FILE
    # --------------------------------------------------------

    file_bytes = await file.read()


    if not file_bytes:

        raise HTTPException(

            status_code=400,

            detail="Uploaded file is empty.",

        )


    file_size_mb = (
        len(file_bytes)
        / (
            1024
            * 1024
        )
    )


    if (
        file_size_mb
        > MAX_FILE_SIZE_MB
    ):

        raise HTTPException(

            status_code=400,

            detail=(
                f"File exceeds "
                f"{MAX_FILE_SIZE_MB} MB limit."
            ),

        )


    # --------------------------------------------------------
    # VISUAL ANALYSIS
    # --------------------------------------------------------

    visual_analysis = (
        analyze_document_image(
            file_bytes
        )
    )


    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    extracted_text = perform_ocr(
        file_bytes,
        file.filename or "document",
    )


    # OCR confidence used by existing scoring
    ocr_confidence = 0.85

    if not extracted_text.strip():

        ocr_confidence = 0.0


    # --------------------------------------------------------
    # NO OCR TEXT
    # --------------------------------------------------------

    if not extracted_text.strip():

        issues = [

            {

                "title":
                    "No text detected",

                "severity":
                    "HIGH",

                "description":
                    "OCR could not extract readable text from the document.",

            }

        ]


        score = 80

        status = status_for_score(
            score
        )


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


            add_audit_log(

                action="DOCUMENT_SCREENED",

                details=(
                    f"No OCR text detected. "
                    f"File: {file.filename}"
                ),

            )


        except Exception as error:

            print(
                "DATABASE SAVE ERROR:",
                error,
            )

            raise HTTPException(

                status_code=500,

                detail=(
                    f"Database error: {str(error)}"
                ),

            )


        return {

            "success":
                True,

            "message":
                "Document received but no readable text was detected.",

            "document_type":
                "Unknown",

            "extracted_information": {

                "name":
                    "Not detected",

                "document_number":
                    "Not detected",

                "dates_found":
                    [],

            },

            "risk_score":
                score,

            "status":
                status,

            "issues":
                issues,

            "verified_checks":
                [],

            "visual_analysis":
                visual_analysis,

            "ocr_confidence":
                ocr_confidence,

            "screening_id":
                screening_id,

        }


    # --------------------------------------------------------
    # CLASSIFY DOCUMENT
    # --------------------------------------------------------

    detected_document_type = (
        classify_document(
            extracted_text
        )
    )


    # --------------------------------------------------------
    # NORMALIZE EXPECTED TYPE
    # --------------------------------------------------------

    expected = (
        expected_document_type
        or "Unknown"
    ).strip()


    aliases = {

        "passport":
            "Passport",

        "visa":
            "Visa",

        "driving licence":
            "Driving Licence",

        "driving license":
            "Driving Licence",

        "dl":
            "Driving Licence",

        "national id":
            "National ID",

        "national identity":
            "National ID",

        "aadhaar":
            "Aadhaar",

        "aadhar":
            "Aadhaar",

        "pan":
            "PAN Card",

        "pan card":
            "PAN Card",

        "permit":
            "Permit",

    }


    expected_normalized = aliases.get(
        expected.lower(),
        expected,
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    (
        issues,
        verified,
        score,
        name,
        document_number,
        dates_found,
    ) = run_validation(

        detected_document_type,

        extracted_text,

        ocr_confidence,

    )


    # --------------------------------------------------------
    # EXPECTED / DETECTED MISMATCH
    # --------------------------------------------------------

    if (

        expected_normalized != "Unknown"

        and detected_document_type
        != "Unknown"

        and expected_normalized
        != detected_document_type

    ):

        issues.append({

            "title":
                "Document type mismatch",

            "severity":
                severity_for_weight(
                    WEIGHTS[
                        "CRITICAL_MISMATCH"
                    ]
                ),

            "description":
                (
                    f"Expected {expected_normalized}, "
                    f"but detected {detected_document_type}."
                ),

        })

        score += WEIGHTS[
            "CRITICAL_MISMATCH"
        ]


    # --------------------------------------------------------
    # VISUAL ISSUES
    # --------------------------------------------------------

    for visual_issue in (
        visual_analysis[
            "visual_issues"
        ]
    ):

        issues.append({

            "title":
                "Visual quality issue",

            "severity":
                severity_for_weight(
                    WEIGHTS[
                        "VISUAL_ISSUE"
                    ]
                ),

            "description":
                visual_issue,

        })

        score += WEIGHTS[
            "VISUAL_ISSUE"
        ]


    # --------------------------------------------------------
    # LIMIT SCORE
    # --------------------------------------------------------

    score = min(
        score,
        100,
    )


    status = status_for_score(
        score
    )


    # --------------------------------------------------------
    # IDENTITY
    # --------------------------------------------------------

    identity_id = None


    if document_number:

        try:

            identity_id = (
                create_or_get_identity(

                    full_name=(
                        name
                        or "Unknown"
                    ),

                    document_number=(
                        document_number
                    ),

                    document_type=(
                        detected_document_type
                    ),

                )
            )

        except Exception as error:

            print(
                "IDENTITY DATABASE ERROR:",
                error,
            )

            raise HTTPException(

                status_code=500,

                detail=(
                    f"Identity database error: {str(error)}"
                ),

            )


    # --------------------------------------------------------
    # SAVE SCREENING
    # --------------------------------------------------------

    try:

        screening_id = save_screening(

            identity_id=identity_id,

            full_name=(
                name
                or "Unknown"
            ),

            document_number=(
                document_number
                or "Not detected"
            ),

            document_type=(
                detected_document_type
            ),

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

    except Exception as error:

        print(
            "SCREENING DATABASE ERROR:",
            error,
        )

        raise HTTPException(

            status_code=500,

            detail=(
                f"Screening database error: {str(error)}"
            ),

        )


    # --------------------------------------------------------
    # AUDIT LOG
    # --------------------------------------------------------

    try:

        add_audit_log(

            action="DOCUMENT_SCREENED",

            details=(
                f"File={file.filename}; "
                f"DocumentType={detected_document_type}; "
                f"DocumentNumber={document_number or 'Not detected'}; "
                f"RiskScore={score}; "
                f"Status={status}"
            ),

        )

    except Exception as error:

        print(
            "AUDIT LOG ERROR:",
            error,
        )


    # --------------------------------------------------------
    # IDENTITY INTELLIGENCE
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PASSPORT EXTRA DATA
    # --------------------------------------------------------

    passport_data = {}


    if detected_document_type == "Passport":

        passport_data = (
            extract_passport_mrz(
                extracted_text
            )
        )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "success":
            True,

        "message":
            "Document received and analyzed successfully.",

        "filename":
            file.filename,

        "document_type":
            detected_document_type,

        "expected_document_type":
            expected_normalized,

        "extracted_information": {

            "name":
                name
                or "Not detected",

            "document_number":
                document_number
                or "Not detected",

            "dates_found":
                dates_found,

            "nationality":
                passport_data.get(
                    "nationality"
                ),

            "date_of_birth":
                passport_data.get(
                    "date_of_birth"
                ),

            "sex":
                passport_data.get(
                    "sex"
                ),

            "expiry_date":
                passport_data.get(
                    "expiry_date"
                ),

        },

        "passport_mrz":
            passport_data
            if detected_document_type
            == "Passport"
            else None,

        "ocr_text":
            extracted_text,

        "ocr_confidence":
            ocr_confidence,

        "risk_score":
            score,

        "status":
            status,

        "issues":
            issues,

        "verified_checks":
            verified,

        "visual_analysis":
            visual_analysis,

        "screening_id":
            screening_id,

        "identity_id":
            identity_id,

        "identity_intelligence":
            identity_intelligence,

    }


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/api/dashboard")
def dashboard():

    conn = None

    try:

        conn = get_connection()

        cursor = conn.cursor()


        # ----------------------------------------------------
        # TOTAL
        # ----------------------------------------------------

        cursor.execute(
            "SELECT COUNT(*) AS total FROM screenings"
        )

        total_screenings = int(
            cursor.fetchone()["total"]
            or 0
        )


        # ----------------------------------------------------
        # AVERAGE RISK
        # ----------------------------------------------------

        cursor.execute(
            "SELECT AVG(risk_score) AS average FROM screenings"
        )

        average = (
            cursor.fetchone()["average"]
        )

        average_risk_score = (
            round(
                float(average),
                1,
            )
            if average is not None
            else 0.0
        )


        # ----------------------------------------------------
        # CLEARED
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM screenings
            WHERE status = %s
            """,
            ("LOW RISK",),
        )

        cleared_documents = int(
            cursor.fetchone()["total"]
            or 0
        )


        # ----------------------------------------------------
        # FLAGGED
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM screenings
            WHERE status != %s
            """,
            ("LOW RISK",),
        )

        flagged_documents = int(
            cursor.fetchone()["total"]
            or 0
        )


        # ----------------------------------------------------
        # RISK DISTRIBUTION
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM screenings
            WHERE status = %s
            """,
            ("LOW RISK",),
        )

        low_risk = int(
            cursor.fetchone()["total"]
            or 0
        )


        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM screenings
            WHERE status = %s
            """,
            ("REVIEW REQUIRED",),
        )

        review_required = int(
            cursor.fetchone()["total"]
            or 0
        )


        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM screenings
            WHERE status = %s
            """,
            ("SUSPICIOUS",),
        )

        suspicious = int(
            cursor.fetchone()["total"]
            or 0
        )


        cursor.execute(
            """
            SELECT COUNT(*) AS total
            FROM screenings
            WHERE status = %s
            """,
            ("HIGH RISK",),
        )

        high_risk = int(
            cursor.fetchone()["total"]
            or 0
        )


        # ----------------------------------------------------
        # RECENT SCREENINGS
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # CHECKPOINT LOAD
        # ----------------------------------------------------

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


    except Exception as error:

        print(
            "DASHBOARD DATABASE ERROR:",
            error,
        )

        raise HTTPException(

            status_code=500,

            detail=(
                f"Dashboard database error: {str(error)}"
            ),

        )


    finally:

        if conn:
            conn.close()


# ============================================================
# AUDIT LOG API
# ============================================================

@app.get("/api/audit")
def get_audit_logs(
    limit: int = 100,
):

    limit = max(
        1,
        min(
            limit,
            500,
        ),
    )

    conn = None

    try:

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                officer_name,
                action,
                details,
                created_at
            FROM audit_logs
            ORDER BY id DESC
            LIMIT %s
            """,
            (limit,),
        )

        logs = [

            dict(row)

            for row in cursor.fetchall()

        ]


        return {

            "success":
                True,

            "count":
                len(logs),

            "audit_logs":
                logs,

        }


    except Exception as error:

        print(
            "AUDIT DATABASE ERROR:",
            error,
        )

        raise HTTPException(

            status_code=500,

            detail=(
                f"Audit database error: {str(error)}"
            ),

        )


    finally:

        if conn:
            conn.close()


# ============================================================
# AUDIT VERIFY
# ============================================================

@app.get("/api/audit/verify")
def verify_audit_chain():

    conn = None

    try:

        conn = get_connection()

        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                id,
                officer_name,
                action,
                details,
                created_at
            FROM audit_logs
            ORDER BY id ASC
            """
        )

        logs = [

            dict(row)

            for row in cursor.fetchall()

        ]


        # Current schema stores normal audit records.
        # It does NOT implement cryptographic hashing.

        return {

            "success":
                True,

            "verified":
                True,

            "count":
                len(logs),

            "cryptographic_hash_chain":
                False,

            "message":
                (
                    "Audit records are readable and "
                    "ordered correctly. Cryptographic "
                    "hash-chain verification is not "
                    "implemented in the current schema."
                ),

        }


    except Exception as error:

        print(
            "AUDIT VERIFY ERROR:",
            error,
        )

        raise HTTPException(

            status_code=500,

            detail=(
                f"Audit verification error: {str(error)}"
            ),

        )


    finally:

        if conn:
            conn.close()


# ============================================================
# IDENTITY INTELLIGENCE
# ============================================================

@app.get("/api/identity/{identity_id}")
def get_identity(
    identity_id: int,
):

    try:

        result = get_identity_intelligence(
            identity_id
        )


        if not result:

            raise HTTPException(

                status_code=404,

                detail="Identity not found",

            )


        return {

            "success":
                True,

            **result,

        }


    except HTTPException:
        raise

    except Exception as error:

        print(
            "IDENTITY ERROR:",
            error,
        )

        raise HTTPException(

            status_code=500,

            detail=(
                f"Identity lookup error: {str(error)}"
            ),

        )


# ============================================================
# STARTUP
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000",
            )
        ),
        reload=True,
    )

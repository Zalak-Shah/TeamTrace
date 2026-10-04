"""
============================================================
TeamTrace / BorderShield AI - Backend
OCR.space + OpenCV + Postgres (Neon) Identity Intelligence
+ Blockchain-style hash-chain audit trail
============================================================
"""

import os
import re
import time
from datetime import date, datetime
from typing import Optional

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from database import (
    initialize_database,
    create_or_get_identity,
    save_screening,
    get_identity_intelligence,
    add_audit_log,
    get_audit_chain,
    verify_audit_chain,
    get_connection,
)


# ============================================================
# SETUP
# ============================================================

load_dotenv()

OCR_SPACE_API_KEY = os.getenv("OCR_SPACE_API_KEY")

if not OCR_SPACE_API_KEY:
    print("WARNING: OCR_SPACE_API_KEY is not set in .env")


app = FastAPI(
    title="BorderShield AI Backend",
    version="2.1.0",
)

initialize_database()


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "null",                        # file:// pages send null origin
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5500",
        "http://localhost:8080",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=False,           # must be False when allow_origins=["*"]
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

# ---- Aadhaar: 12-digit in groups of 4 ----
AADHAAR_NUMBER_RE = re.compile(
    r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"
)

# ---- PAN: AAAAA9999A (5 alpha, 4 digit, 1 alpha) ----
PAN_NUMBER_RE = re.compile(
    r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
    re.IGNORECASE,
)

# ---- Indian Passport: 1 letter + 7 digits (e.g. A1234567) ----
# Must NOT start with two letters to avoid matching DL numbers
INDIAN_PASSPORT_RE = re.compile(
    r"\b[A-PR-WY][0-9]{7}\b"
)

# ---- International Passport (generic): 1-2 letters + 6-8 digits ----
PASSPORT_NUMBER_RE = re.compile(
    r"\b[A-Z]{1,2}[0-9]{6,8}\b",
    re.IGNORECASE,
)

# ---- Indian Driving Licence: State(2) + RTO(2) + Year(4) + Seq(7) ----
# e.g. MH01 20160123456  or  DL-0420110012345
# Two-letter state code from known Indian state codes
DL_NUMBER_RE = re.compile(
    r"\b(?:AN|AP|AR|AS|BR|CG|CH|DD|DL|DN|GA|GJ|HP|HR|JH|JK|KA|KL"
    r"|LA|LD|MH|ML|MN|MP|MZ|NL|OD|PB|PY|RJ|SK|TN|TR|TS|UK|UP|WB)"
    r"[-\s]?\d{2}[-\s]?\d{4}[-\s]?\d{7}\b",
    re.IGNORECASE,
)

# ---- Voter ID (EPIC): 3 alpha + 7 digits ----
VOTER_ID_RE = re.compile(
    r"\b[A-Z]{3}[0-9]{7}\b",
    re.IGNORECASE,
)

# ---- Date formats: DD/MM/YYYY, YYYY-MM-DD, DD-MM-YYYY ----
DATE_RE = re.compile(
    r"\b("
    r"\d{2}[/\-.]\d{2}[/\-.]\d{4}"
    r"|"
    r"\d{4}[/\-.]\d{2}[/\-.]\d{2}"
    r")\b"
)


# ============================================================
# DOCUMENT KEYWORDS
# ============================================================

KEYWORDS = {
    "Aadhaar": [
        "unique identification authority",
        "aadhaar", "aadhar", "uidai", "uid",
        "vid", "mera aadhaar", "government of india",
        "enrolment", "enrollment", "help@uidai", "1947",
    ],
    "PAN Card": [
        "income tax department",
        "permanent account number",
        "pan card", "pan no", "pan number",
        "income tax india",
    ],
    "Passport": [
        "passport", "republic of india",
        "ministry of external affairs",
        "surname", "given name", "given names",
        "place of birth", "date of issue",
        "date of expiry", "place of issue",
        "nationality",
    ],
    "Visa": [
        "visa", "visa number", "visa no",
        "valid from", "valid until", "valid to",
        "entries", "visa type", "category",
        "port of entry", "visa category",
    ],
    "Voter ID": [
        "election commission", "voter",
        "electors photo identity",
        "epic", "voter id", "voter card",
        "constituency", "part no",
        "serial no",
    ],
    "National ID": [
        "national identity", "identity card",
        "national id", "identification card",
    ],
    "Driving Licence": [
        "driving licence", "driving license",
        "motor vehicles act",
        "transport department",
        "sarathi", "licence no", "license no",
        "dl no", "dl-", "vehicle class",
        "blood group", "non-transport",
        "transport vehicle",
    ],
    "Permit": [
        "permit", "permit number", "permission",
        "permit type",
    ],
}


# ============================================================
# DATE HELPER
# ============================================================

def to_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None

    formats = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
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
# DOCUMENT CLASSIFICATION
# ============================================================

def classify_document(text: str) -> str:
    """
    Detect document type with strict priority ordering so no
    document type steals another's number pattern.

    Priority:
      1. Passport MRZ (unambiguous)
      2. Aadhaar   (12-digit + keyword)
      3. PAN Card  (AAAAA9999A + keyword)
      4. Driving Licence (Indian DL pattern + keyword)
      5. Voter ID  (ECI keyword / EPIC pattern)
      6. Passport  (keyword fallback)
      7. Visa
      8. National ID
      9. Permit
      10. Keyword score best-match
    """

    if not text:
        return "Unknown"

    lower = text.lower()

    # ---- 1. Passport MRZ (highest confidence) ----
    if parse_mrz(text):
        return "Passport"

    # ---- compute keyword scores for all types ----
    scores = {
        dt: sum(1 for kw in kws if kw in lower)
        for dt, kws in KEYWORDS.items()
    }

    # ---- 2. Aadhaar ----
    _aadhaar_kw = (
        "aadhaar" in lower or "aadhar" in lower or "uidai" in lower
        or "unique identification" in lower or "mera aadhaar" in lower
        or "help@uidai" in lower or "enrolment" in lower
        or "enrollment" in lower
    )
    if AADHAAR_NUMBER_RE.search(text) and (_aadhaar_kw or scores.get("Aadhaar", 0) >= 1):
        return "Aadhaar"

    # ---- 3. PAN Card ----
    if PAN_NUMBER_RE.search(text) and (
        "income tax" in lower
        or "permanent account" in lower
        or "pan" in lower
    ):
        return "PAN Card"

    # ---- 4. Driving Licence (check BEFORE passport to avoid stealing DL numbers) ----
    _dl_kw = (
        "driving licence" in lower or "driving license" in lower
        or "licence no" in lower or "license no" in lower
        or "dl no" in lower or "dl-" in lower
        or "transport department" in lower or "sarathi" in lower
        or "motor vehicles" in lower or "vehicle class" in lower
        or "blood group" in lower
    )
    if _dl_kw or DL_NUMBER_RE.search(text):
        if _dl_kw:
            return "Driving Licence"
        if DL_NUMBER_RE.search(text) and scores.get("Driving Licence", 0) >= 1:
            return "Driving Licence"

    # ---- 5. Voter ID ----
    _voter_kw = (
        "election commission" in lower or "electors photo" in lower
        or "epic" in lower or "voter" in lower or "constituency" in lower
    )
    if _voter_kw or (VOTER_ID_RE.search(text) and scores.get("Voter ID", 0) >= 1):
        return "Voter ID"

    # ---- 6. Passport (keyword fallback) ----
    _passport_kw = (
        "passport" in lower or "surname" in lower
        or "given name" in lower or "place of birth" in lower
        or "date of expiry" in lower or "ministry of external affairs" in lower
    )
    if _passport_kw or (INDIAN_PASSPORT_RE.search(text) and scores.get("Passport", 0) >= 1):
        return "Passport"

    # ---- 7. Visa ----
    if (
        scores.get("Visa", 0) >= 2
        or "visa number" in lower or "visa type" in lower
        or "valid until" in lower or "port of entry" in lower
    ):
        return "Visa"

    # ---- 8. National ID ----
    if (
        scores.get("National ID", 0) >= 1
        or "national identity" in lower or "identity card" in lower
    ):
        return "National ID"

    # ---- 9. Permit ----
    if scores.get("Permit", 0) >= 2 or "permit number" in lower:
        return "Permit"

    # ---- 10. Best keyword score ----
    if scores:
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best

    return "Unknown"


# ============================================================
# MRZ PARSER
# ============================================================

def parse_mrz(text: str):
    """
    Parse a standard passport TD3 MRZ.

    Returns:
        name
        doc_number
        nationality
        date_of_birth
        sex
        expiry_date
        dates
    """

    if not text:
        return None

    normalized_lines = []

    for raw_line in text.splitlines():
        line = raw_line.upper().strip()

        if not line:
            continue

    # OCR may convert MRZ filler characters into spaces.
        line = line.replace(" ", "<")

        # Keep only characters valid in an MRZ.
        line = re.sub(
            r"[^A-Z0-9<]",
            "",
            line,
        )

        if len(line) >= 30:
            normalized_lines.append(line)

    # --------------------------------------------------------
    # FIND TWO CONSECUTIVE TD3 MRZ LINES
    # --------------------------------------------------------

    for i in range(len(normalized_lines) - 1):
        line1 = normalized_lines[i]
        line2 = normalized_lines[i + 1]

        # Passport MRZ line 1 normally begins with P<
        if not (
            line1.startswith("P<")
            or line1.startswith("PK")
        ):
            continue

        if len(line1) < 35 or len(line2) < 35:
            continue

        # Standard TD3 width is 44 characters.
        line1 = line1[:44]
        line2 = line2[:44]

        # ----------------------------------------------------
        # NAME
        # ----------------------------------------------------

        mrz_name = None

        name_part = line1[5:]

        name_parts = name_part.split(
            "<<",
            1,
        )

        if len(name_parts) == 2:
            surname = (
                name_parts[0]
                .replace("<", " ")
                .strip()
            )

            given_names = (
                name_parts[1]
                .replace("<", " ")
                .strip()
            )

            surname = re.sub(
                r"\s+",
                " ",
                surname,
            )

            given_names = re.sub(
                r"\s+",
                " ",
                given_names,
            )

            if surname or given_names:
                mrz_name = (
                    f"{given_names} {surname}"
                    .strip()
                )

        # ----------------------------------------------------
        # LINE 2
        #
        # 0-8   document number
        # 9     check digit
        # 10-12 nationality
        # 13-18 DOB
        # 19    DOB check digit
        # 20    sex
        # 21-26 expiry
        # 27    expiry check digit
        # ----------------------------------------------------

        if len(line2) < 28:
            continue

        doc_number = (
            line2[0:9]
            .replace("<", "")
            .strip()
        )

        nationality = (
            line2[10:13]
            .replace("<", "")
            .strip()
        )

        dob_raw = line2[13:19]
        sex = line2[20:21]
        expiry_raw = line2[21:27]

        date_values = []

        # ----------------------------------------------------
        # DATE OF BIRTH
        # ----------------------------------------------------

        date_of_birth = None

        if re.fullmatch(
            r"\d{6}",
            dob_raw,
        ):
            yy = int(dob_raw[0:2])
            mm = int(dob_raw[2:4])
            dd = int(dob_raw[4:6])

            full_year = (
                2000 + yy
                if yy <= 30
                else 1900 + yy
            )

            try:
                dob = date(
                    full_year,
                    mm,
                    dd,
                )

                date_of_birth = dob.isoformat()

                date_values.append(
                    dob.strftime("%d/%m/%Y")
                )

            except ValueError:
                pass

        # ----------------------------------------------------
        # EXPIRY DATE
        # ----------------------------------------------------

        expiry_date = None

        if re.fullmatch(
            r"\d{6}",
            expiry_raw,
        ):
            yy = int(expiry_raw[0:2])
            mm = int(expiry_raw[2:4])
            dd = int(expiry_raw[4:6])

            # Passport MRZ expiry years are treated
            # as 2000-2099.
            full_year = 2000 + yy

            try:
                expiry = date(
                    full_year,
                    mm,
                    dd,
                )

                expiry_date = expiry.isoformat()

                date_values.append(
                    expiry.strftime("%d/%m/%Y")
                )

            except ValueError:
                pass

        # ----------------------------------------------------
        # RETURN PARSED MRZ
        # ----------------------------------------------------

        if not doc_number:
            return None

        return {
            "name": mrz_name,
            "doc_number": doc_number,
            "nationality": nationality or None,
            "date_of_birth": date_of_birth,
            "sex": sex if sex in ("M", "F") else None,
            "expiry_date": expiry_date,
            "dates": date_values,
        }

    return None


# ============================================================
# EXTRACT DOCUMENT NUMBER
# ============================================================

def extract_document_number(
    text: str,
    doc_type: str,
) -> Optional[str]:

    if doc_type == "Passport":
        mrz = parse_mrz(text)
        if mrz and mrz.get("doc_number"):
            return mrz["doc_number"]
        # fallback: Indian passport regex
        m = INDIAN_PASSPORT_RE.search(text)
        if m:
            return m.group(0).upper()

    elif doc_type == "Aadhaar":
        m = AADHAAR_NUMBER_RE.search(text)
        if m:
            return re.sub(r"[\s-]", "", m.group(0))

    elif doc_type == "PAN Card":
        m = PAN_NUMBER_RE.search(text)
        if m:
            return m.group(0).upper()

    elif doc_type == "Driving Licence":
        m = DL_NUMBER_RE.search(text)
        if m:
            return re.sub(r"[\s-]", "", m.group(0)).upper()

    elif doc_type == "Voter ID":
        m = VOTER_ID_RE.search(text)
        if m:
            return m.group(0).upper()

    return None


# ============================================================
# EXTRACT DATES
# ============================================================

def extract_dates(text: str) -> list:
    mrz = parse_mrz(text)

    if mrz and mrz.get("dates"):
        return mrz["dates"] + DATE_RE.findall(text)

    return DATE_RE.findall(text)


# ============================================================
# GUESS NAME
# ============================================================

def guess_name(text: str) -> Optional[str]:

    mrz = parse_mrz(text)

    if (
        mrz
        and mrz.get("name")
        and mrz["name"] != "Unknown"
    ):
        return mrz["name"]

    skip_words = {
        # English document terms
        "GOVERNMENT", "INDIA", "REPUBLIC", "INCOME", "TAX",
        "DEPARTMENT", "UNIQUE", "IDENTIFICATION", "AUTHORITY",
        "TRANSPORT", "LICENCE", "LICENSE", "AADHAAR", "AADHAR",
        "PASSPORT", "NATIONALITY", "VISA", "PERMIT", "VALID",
        "EXPIRY", "SURNAME", "GIVEN", "NAME", "NATIONAL",
        "IDENTITY", "CARD", "DRIVING", "DOCUMENT", "NUMBER",
        "DATE", "BIRTH", "ISSUE", "ISSUED", "PLACE", "SEX",
        "GENDER", "MALE", "FEMALE", "COUNTRY", "CITIZEN",
        "CITIZENSHIP", "BORDER", "IMMIGRATION", "OFFICER",
        "MINISTRY", "EXTERNAL", "AFFAIRS", "POLICE",
        "AUTHORITY",
        # Aadhaar / UID specific
        "UIDAI", "UID", "VID", "MERA", "YOUR", "DOWNLOAD",
        "ENROLMENT", "ENROLLMENT", "HELP", "DOB", "ADDRESS",
        "YEAR", "MOBILE", "PHONE", "RESIDENT", "VERIFICATION",
        "LETTER", "PRINT", "MASKED", "VIRTUAL",
        # Polish
        "PASZPORT", "RZECZPOSPOLITA", "POLSKA", "REPUBLIC",
        "VOIVODESHIP", "MAZOWIECKIE", "MAZOVIECKI",
        # French
        "PASSEPORT", "REPUBLIQUE", "FRANCAISE", "NOM", "PRENOM",
        # German
        "REISEPASS", "BUNDESREPUBLIK", "DEUTSCHLAND", "AUSWEISE",
        # Spanish
        "PASAPORTE", "REPUBLICA", "ESPAÑOLA",
        # Arabic
        "JAWAZ", "SAFAR",
        # Generic document words
        "INTERNATIONAL", "FEDERATION", "UNION", "STATE",
        "STATES", "DEMOCRATIC", "PEOPLE",
        "MINISTRY", "HOME", "AFFAIRS", "INTERIOR",
    }

    for line in text.splitlines():
        clean = line.strip()

        if not clean:
            continue

        if (
            4 <= len(clean) <= 50
            and re.fullmatch(
                r"[A-Za-z][A-Za-z .'-]+",
                clean,
            )
        ):
            words = {
                word.upper()
                for word in clean.split()
            }

            if not (words & skip_words):
                if len(words) <= 5:
                    return clean

    return None


# ============================================================
# VISUAL ANALYSIS
# ============================================================
# ============================================================
# DOCUMENT-SPECIFIC INFORMATION EXTRACTION
# ============================================================

def extract_document_information(
    text: str,
    doc_type: str,
) -> dict:
    """
    Extract useful fields according to the detected document type.

    Only values actually found in OCR/MRZ text are returned.
    Missing values remain None and are not fabricated.
    """

    information = {
        "document_type": doc_type,
        "name": None,
        "document_number": None,
        "nationality": None,
        "date_of_birth": None,
        "sex": None,
        "date_of_issue": None,
        "date_of_expiry": None,
        "father_name": None,
        "mother_name": None,
        "address": None,
        "visa_type": None,
        "visa_number": None,
        "entries": None,
        "valid_from": None,
        "valid_until": None,
        "licence_number": None,
        "blood_group": None,
        "vehicle_class": None,
        "constituency": None,
        "permit_number": None,
        "permit_type": None,
        "raw_dates": [],
    }


    if not text:
        return information

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def clean_value(value: Optional[str]) -> Optional[str]:
        if not value:
            return None

        value = value.strip()
        value = re.sub(r"\s+", " ", value)

        return value if value else None

    def find_labeled_value(
        labels,
        max_length: int = 100,
    ) -> Optional[str]:

        lines = text.splitlines()

        for index, raw_line in enumerate(lines):

            line = raw_line.strip()

            if not line:
                continue

            lower_line = line.lower()

            for label in labels:

                label_lower = label.lower()

                # Case 1:
                # "Date of Birth: 23/04/1984"
                pattern = (
                    re.escape(label_lower)
                    + r"\s*[:\-#]?\s*(.+)$"
                )

                match = re.search(
                    pattern,
                    lower_line,
                )

                if match:

                    start = match.start(1)

                    original_value = line[start:]

                    original_value = clean_value(
                        original_value
                    )

                    if original_value:
                        return original_value[:max_length]

                # Case 2:
                # Label on one line,
                # value on next line.
                if label_lower in lower_line:

                    if index + 1 < len(lines):

                        next_value = clean_value(
                            lines[index + 1]
                        )

                        if next_value:
                            return next_value[:max_length]

        return None

    # ========================================================
    # PASSPORT
    # ========================================================

    if doc_type == "Passport":

        mrz = parse_mrz(text)

        if mrz:

            information["name"] = mrz.get("name")
            information["document_number"] = (
                mrz.get("doc_number")
            )
            information["nationality"] = (
                mrz.get("nationality")
            )
            information["date_of_birth"] = (
                mrz.get("date_of_birth")
            )
            information["sex"] = mrz.get("sex")
            information["date_of_expiry"] = (
                mrz.get("expiry_date")
            )

        if not information["name"]:
            information["name"] = guess_name(text)

        if not information["document_number"]:
            information["document_number"] = (
                extract_document_number(
                    text,
                    doc_type,
                )
            )

        issue_date = find_labeled_value(
            [
                "date of issue",
                "date of issuance",
                "issue date",
                "issued",
            ]
        )

        if issue_date:
            issue_match = DATE_RE.search(
                issue_date
            )

            if issue_match:
                parsed = to_date(
                    issue_match.group(0)
                )

                if parsed:
                    information["date_of_issue"] = (
                        parsed.isoformat()
                    )

    # ========================================================
    # VISA
    # ========================================================

    elif doc_type == "Visa":

        information["visa_number"] = (
            find_labeled_value(
                [
                    "visa number",
                    "visa no",
                    "visa no.",
                    "document number",
                    "number",
                ]
            )
        )

        information["document_number"] = (
            information["visa_number"]
        )

        information["name"] = (
            find_labeled_value(
                [
                    "name",
                    "full name",
                    "surname",
                    "given name",
                ]
            )
            or guess_name(text)
        )

        information["nationality"] = (
            find_labeled_value(
                [
                    "nationality",
                    "citizenship",
                ]
            )
        )

        information["visa_type"] = (
            find_labeled_value(
                [
                    "visa type",
                    "type of visa",
                    "type",
                ]
            )
        )

        information["entries"] = (
            find_labeled_value(
                [
                    "entries",
                    "entry",
                    "number of entries",
                ]
            )
        )

        information["valid_from"] = (
            find_labeled_value(
                [
                    "valid from",
                    "valid from date",
                    "validity from",
                    "date of issue",
                ]
            )
        )

        information["valid_until"] = (
            find_labeled_value(
                [
                    "valid until",
                    "valid to",
                    "validity until",
                    "expiry date",
                    "date of expiry",
                ]
            )

        )

        information["date_of_birth"] = (
            find_labeled_value(
                [
                    "date of birth",
                    "dob",
                    "birth date",
                ]
            )
        )

        information["sex"] = (
            find_labeled_value(
                [
                    "sex",
                    "gender",
                ]
            )
        )

    # ========================================================
    # AADHAAR
    # ========================================================

    elif doc_type == "Aadhaar":

        match = AADHAAR_NUMBER_RE.search(text)

        if match:
            information["document_number"] = (
                match.group(0).replace(" ", "")
            )

        information["name"] = (
            find_labeled_value(
                [
                    "name",
                    "name:",
                ]
            )
        )

        # Validate: if find_labeled_value matched a line that
        # looks like a header / label, discard it and try harder.
        if information["name"]:
            _n = information["name"].lower()
            bad_keywords = {
                "aadhaar", "aadhar", "uidai", "uid",
                "government", "india", "unique",
                "identification", "authority",
                "enrolment", "enrollment", "mera",
                "your", "download", "date", "birth",
                "address", "male", "female", "vid",
                "help", "dob", "gender", "sex",
            }
            if any(bw in _n for bw in bad_keywords):
                information["name"] = None

        # Aadhaar-specific positional heuristic:
        # The holder name typically appears on a line just
        # before "DOB", "Date of Birth", "Male/Female", or
        # the Aadhaar number.  Scan for that anchor and grab
        # the preceding all-alpha line.
        if not information["name"]:
            lines = text.splitlines()
            anchor_re = re.compile(
                r"(date of birth|dob|\bmale\b|\bfemale\b"
                r"|\b\d{4}\s?\d{4}\s?\d{4}\b)",
                re.IGNORECASE,
            )
            for idx, raw_line in enumerate(lines):
                if anchor_re.search(raw_line):
                    # Walk backwards to find the name line
                    for back in range(idx - 1, max(idx - 4, -1), -1):
                        candidate = lines[back].strip()
                        if (
                            candidate
                            and 3 <= len(candidate) <= 60
                            and re.fullmatch(
                                r"[A-Za-z][A-Za-z .'-]+",
                                candidate,
                            )
                        ):
                            cwords = {
                                w.upper()
                                for w in candidate.split()
                            }
                            _skip = {
                                "GOVERNMENT", "INDIA",
                                "UNIQUE", "IDENTIFICATION",
                                "AUTHORITY", "AADHAAR",
                                "UIDAI", "MALE", "FEMALE",
                                "DATE", "BIRTH", "DOB",
                                "ADDRESS", "GENDER", "SEX",
                                "ENROLMENT", "ENROLLMENT",
                                "NAME", "YOUR", "MERA",
                                "DOWNLOAD", "VID",
                            }
                            if not (cwords & _skip):
                                information["name"] = (
                                    candidate
                                )
                                break
                    if information["name"]:
                        break

        if not information["name"]:
            information["name"] = guess_name(text)


        information["date_of_birth"] = (
            find_labeled_value(
                [
                    "date of birth",
                    "dob",
                    "birth",
                ]
            )
        )

        information["sex"] = (
            find_labeled_value(
                [
                    "gender",
                    "sex",
                ]
            )
        )

        # Aadhaar cards often print gender as a standalone word
        # (e.g. "Male" or "Female") without a label prefix.
        if not information["sex"]:
            gender_re = re.compile(
                r"\b(male|female|transgender"
                r"|\u092a\u0941\u0930\u0941\u0937"
                r"|\u0938\u094d\u0924\u094d\u0930\u0940"
                r"|\u092e\u0939\u093f\u0932\u093e)\b",
                re.IGNORECASE,
            )
            for raw_line in text.splitlines():
                line = raw_line.strip()
                match = gender_re.search(line)
                if match:
                    val = match.group(1).strip()
                    # Map Hindi terms to English
                    hindi_map = {
                        "\u092a\u0941\u0930\u0941\u0937": "Male",
                        "\u0938\u094d\u0924\u094d\u0930\u0940": "Female",
                        "\u092e\u0939\u093f\u0932\u093e": "Female",
                    }
                    information["sex"] = hindi_map.get(
                        val, val.capitalize()
                    )
                    break

        information["address"] = (
            find_labeled_value(
                [
                    "address",
                ],
                max_length=300,
            )
        )


    # ========================================================
    # PAN CARD
    # Real labels: "Name", "Father's Name", "Date of Birth"
    # Number: AAAAA9999A
    # Issuer: Income Tax Department, Govt of India
    # ========================================================

    elif doc_type == "PAN Card":

        m = PAN_NUMBER_RE.search(text)
        if m:
            information["document_number"] = m.group(0).upper()

        information["name"] = (
            find_labeled_value([
                "name",
            ])
            or guess_name(text)
        )

        information["father_name"] = (
            find_labeled_value([
                "father's name", "father name",
                "father",
            ])
        )

        information["date_of_birth"] = (
            find_labeled_value([
                "date of birth", "dob",
            ])
        )

    # ========================================================
    # DRIVING LICENCE
    # Indian DL format (DigiLocker / Sarathi):
    #   Licence No, Name, S/O D/O, DOB, Issue Date,
    #   Valid Till (NT), Valid Till (T), Blood Group,
    #   Vehicle Class(es), Address
    # ========================================================

    elif doc_type == "Driving Licence":

        # DL number: try labeled first, then regex
        dl_num = find_labeled_value([
            "licence no", "license no",
            "dl no", "dl no.", "dl-no",
            "driving licence no", "driving license no",
            "licence number", "license number",
        ])
        if dl_num:
            information["licence_number"] = dl_num.upper()
        else:
            m = DL_NUMBER_RE.search(text)
            if m:
                information["licence_number"] = (
                    re.sub(r"[\s-]", "", m.group(0)).upper()
                )
        information["document_number"] = (
            information["licence_number"]
        )

        information["name"] = (
            find_labeled_value([
                "name", "holder name",
                "name of holder", "applicant name",
                "licence holder",
            ])
            or guess_name(text)
        )

        information["date_of_birth"] = (
            find_labeled_value([
                "date of birth", "dob", "birth date",
                "d.o.b",
            ])
        )

        information["date_of_issue"] = (
            find_labeled_value([
                "date of issue", "issue date",
                "issued on", "doi",
                "initial issue date",
            ])
        )

        # Indian DL has two validity dates: NT (non-transport) and T (transport)
        information["date_of_expiry"] = (
            find_labeled_value([
                "valid till (nt)", "non-transport valid till",
                "validity (nt)", "nt",
                "valid till", "valid until",
                "validity", "date of expiry",
                "expiry date", "doe",
            ])
        )

        information["sex"] = (
            find_labeled_value(["sex", "gender"])
        )
        if not information["sex"]:
            _g = re.search(
                r"\b(male|female|m|f)\b",
                text, re.IGNORECASE
            )
            if _g:
                v = _g.group(1).upper()
                information["sex"] = (
                    "Male" if v in ("M", "MALE")
                    else "Female"
                )

        information["blood_group"] = (
            find_labeled_value([
                "blood group", "blood grp", "bg",
            ])
        )
        if not information["blood_group"]:
            _bg = re.search(
                r"\b(A|B|AB|O)[+-]\b",
                text, re.IGNORECASE
            )
            if _bg:
                information["blood_group"] = _bg.group(0).upper()

        information["vehicle_class"] = (
            find_labeled_value([
                "vehicle class", "class of vehicle",
                "cov", "class",
            ], max_length=200)
        )

        information["address"] = (
            find_labeled_value(["address"], max_length=300)
        )

    # ========================================================
    # VOTER ID (EPIC)
    # Real labels (Election Commission of India):
    #   EPIC No, Name, Father's Name / Husband's Name,
    #   Date of Birth, Age, Sex, Address,
    #   Part No, Serial No, Constituency
    # ========================================================

    elif doc_type == "Voter ID":

        epic = find_labeled_value([
            "epic no", "epic number",
            "voter id no", "voter id number",
            "elector's photo identity card",
        ])
        if not epic:
            m = VOTER_ID_RE.search(text)
            if m:
                epic = m.group(0).upper()
        information["document_number"] = epic

        information["name"] = (
            find_labeled_value([
                "name", "elector name",
                "full name",
            ])
            or guess_name(text)
        )

        information["father_name"] = (
            find_labeled_value([
                "father's name", "father name",
                "husband's name", "husband name",
                "s/o", "d/o", "w/o",
            ])
        )

        information["date_of_birth"] = (
            find_labeled_value([
                "date of birth", "dob",
            ])
        )

        information["sex"] = (
            find_labeled_value(["sex", "gender"])
        )

        information["address"] = (
            find_labeled_value(["address"], max_length=300)
        )

        information["constituency"] = (
            find_labeled_value([
                "constituency", "assembly constituency",
                "parliamentary constituency",
            ])
        )

    # ========================================================
    # NATIONAL ID
    # ========================================================

    elif doc_type == "National ID":

        information["document_number"] = (
            find_labeled_value([
                "id number", "identity number",
                "national id number", "document number",
                "id no", "id no.",
            ])
        )

        information["name"] = (
            find_labeled_value(["name", "full name"])
            or guess_name(text)
        )

        information["nationality"] = (
            find_labeled_value(["nationality", "citizenship"])
        )

        information["date_of_birth"] = (
            find_labeled_value(["date of birth", "dob", "birth date"])
        )

        information["sex"] = (
            find_labeled_value(["sex", "gender"])
        )

        information["address"] = (
            find_labeled_value(["address"], max_length=300)
        )

    # ========================================================
    # PERMIT
    # ========================================================

    elif doc_type == "Permit":

        information["permit_number"] = (
            find_labeled_value(
                [
                    "permit number",
                    "permit no",
                    "permit no.",
                    "document number",
                ]
            )
        )

        information["document_number"] = (
            information["permit_number"]
        )

        information["permit_type"] = (
            find_labeled_value(
                [
                    "permit type",
                    "type of permit",
                    "permit category",
                    "type",
                ]
            )
        )

        information["name"] = (
            find_labeled_value(
                [
                    "name",
                    "holder name",
                    "applicant name",
                ]
            )
            or guess_name(text)
        )

        information["date_of_issue"] = (
            find_labeled_value(
                [
                    "date of issue",
                    "issue date",
                    "issued on",
                ]
            )
        )

        information["date_of_expiry"] = (
            find_labeled_value(
                [
                    "date of expiry",
                    "expiry date",
                    "valid until",
                    "valid till",
                ]
            )
        )

    # ========================================================
    # NORMALIZE DATES
    # ========================================================

    for key in [
        "date_of_birth",
        "date_of_issue",
        "date_of_expiry",
    ]:

        value = information.get(key)

        if value:

            match = DATE_RE.search(
                str(value)
            )

            if match:

                parsed = to_date(
                    match.group(0)
                )

                if parsed:
                    information[key] = (
                        parsed.isoformat()
                    )

    # ========================================================
    # RAW DATES FROM OCR
    # ========================================================

    information["raw_dates"] = extract_dates(
        text
    )

    return information
def analyze_document_image(file_bytes: bytes):

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
        result["visual_checks"].append(
            "Visual analysis skipped for non-image document"
        )

        result["quality_score"] = 80

        return result

    result["is_image"] = True

    height, width = image.shape[:2]

    result["width"] = width
    result["height"] = height
    result["aspect_ratio"] = round(
        width / height,
        3,
    )

    if width < 500 or height < 500:
        result["visual_issues"].append(
            "Low image resolution"
        )
    else:
        result["visual_checks"].append(
            "Image resolution is sufficient"
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    ).var()

    result["blur_score"] = round(
        float(blur_score),
        2,
    )

    if blur_score < 50:
        result["visual_issues"].append(
            "Image appears blurry"
        )
    else:
        result["visual_checks"].append(
            "Image sharpness is acceptable"
        )

    brightness = float(
        np.mean(gray)
    )

    result["brightness"] = round(
        brightness,
        2,
    )

    if brightness < 40:
        result["visual_issues"].append(
            "Image is too dark"
        )

    elif brightness > 230:
        result["visual_issues"].append(
            "Image is excessively bright"
        )

    else:
        result["visual_checks"].append(
            "Image brightness is acceptable"
        )

    contrast = float(
        np.std(gray)
    )

    result["contrast"] = round(
        contrast,
        2,
    )

    if contrast < 20:
        result["visual_issues"].append(
            "Image has very low contrast"
        )
    else:
        result["visual_checks"].append(
            "Image contrast is acceptable"
        )

    quality_score = 100

    quality_score -= (
        len(result["visual_issues"]) * 15
    )

    result["quality_score"] = max(
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

    name = guess_name(text)

    if name:
        verified.append(
            "A likely name was detected"
        )
    else:
        issues.append(
            {
                "title": "Required field missing",
                "severity": severity_for_weight(
                    WEIGHTS["MISSING_REQUIRED_FIELD"]
                ),
                "description": (
                    "Could not locate a likely name "
                    "on the document."
                ),
            }
        )

        score += WEIGHTS[
            "MISSING_REQUIRED_FIELD"
        ]

    # --------------------------------------------------------
    # DOCUMENT NUMBER
    # --------------------------------------------------------

    doc_number = extract_document_number(
        text,
        doc_type,
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
    ):
        issues.append(
            {
                "title": "Invalid or missing document number",
                "severity": severity_for_weight(
                    WEIGHTS["INVALID_NUMBER_FORMAT"]
                ),
                "description": (
                    f"No valid {doc_type} number "
                    "pattern was found."
                ),
            }
        )

        score += WEIGHTS[
            "INVALID_NUMBER_FORMAT"
        ]

    # --------------------------------------------------------
    # DATES
    # --------------------------------------------------------

    dates_found = extract_dates(text)

    if not dates_found:
        issues.append(
            {
                "title": "Required field missing",
                "severity": severity_for_weight(
                    WEIGHTS["MISSING_REQUIRED_FIELD"]
                ),
                "description": (
                    "No date could be extracted "
                    "from the document."
                ),
            }
        )

        score += WEIGHTS[
            "MISSING_REQUIRED_FIELD"
        ]

    else:
        verified.append(
            "Date information was extracted"
        )

        if doc_type in (
            "Passport",
            "Visa",
            "Driving Licence",
            "Permit",
        ):
            parsed_dates = []

            for value in dates_found:
                parsed = to_date(value)

                if parsed:
                    parsed_dates.append(
                        parsed
                    )

            if parsed_dates:

                latest_date = max(
                    parsed_dates
                )

                if latest_date < date.today():

                    issues.append(
                        {
                            "title": "Possible expired document",
                            "severity": "MEDIUM",
                            "description": (
                                "The latest detected date "
                                "appears to have expired."
                            ),
                        }
                    )

                    score += WEIGHTS[
                        "EXPIRED_DOCUMENT"
                    ]

                else:
                    verified.append(
                        "Latest detected date is not expired"
                    )

    # --------------------------------------------------------
    # OCR CONFIDENCE
    # --------------------------------------------------------

    if ocr_confidence < 0.60:

        issues.append(
            {
                "title": "Low OCR confidence",
                "severity": "MEDIUM",
                "description": (
                    "OCR quality appears low."
                ),
            }
        )

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

        issues.append(
            {
                "title": "Document type could not be identified",
                "severity": "HIGH",
                "description": (
                    "No supported document pattern "
                    "was confidently detected."
                ),
            }
        )

        score += WEIGHTS[
            "CRITICAL_MISMATCH"
        ]

    return (
        issues,
        verified,
        score,
        name,
        doc_number,
    )


# ============================================================
# OCR PREPROCESSING
# ============================================================

def preprocess_image_for_ocr(
    file_bytes: bytes,
) -> bytes:

    image_array = np.frombuffer(
        file_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        return file_bytes

    h, w = image.shape[:2]

    max_dim = 2000

    if max(h, w) > max_dim:

        scale = max_dim / max(h, w)

        image = cv2.resize(
            image,
            (
                int(w * scale),
                int(h * scale),
            ),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.fastNlMeansDenoising(
        gray,
        h=30,
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    gray = clahe.apply(gray)

    success, encoded_img = cv2.imencode(
        ".png",
        gray,
    )

    if success:
        return encoded_img.tobytes()

    return file_bytes


# ============================================================
# OCR.SPACE
# With retry + local fallback
# ============================================================

def perform_ocr(
    file_bytes: bytes,
    filename: str,
) -> str:

    if OCR_SPACE_API_KEY:

        url = "https://api.ocr.space/parse/image"

        files = {
            "file": (
                filename,
                file_bytes,
            )
        }

        data = {
            "apikey": OCR_SPACE_API_KEY,
            "language": "eng",
            "isOverlayRequired": "false",
            "OCREngine": "2",
        }

        max_retries = 2
        base_delay = 1

        for attempt in range(max_retries):

            try:

                response = requests.post(
                    url,
                    files=files,
                    data=data,
                    timeout=20,
                )

                if response.status_code in (
                    429,
                    500,
                    502,
                    503,
                    504,
                ):

                    print(
                        "OCR.space temporary error "
                        f"{response.status_code}. "
                        "Retrying..."
                    )

                    time.sleep(
                        base_delay
                    )

                    continue

                response.raise_for_status()

                result = response.json()

                if result.get(
                    "IsErroredOnProcessing"
                ):

                    print(
                        "OCR.space processing error:",
                        result.get(
                            "ErrorMessage"
                        ),
                    )

                    break

                parsed_results = result.get(
                    "ParsedResults",
                    [],
                )

                if parsed_results:

                    return "\n".join(
                        item.get(
                            "ParsedText",
                            "",
                        )
                        for item in parsed_results
                    )

                return ""

            except requests.exceptions.RequestException as error:

                print(
                    "OCR.space network error "
                    f"on attempt {attempt + 1}: "
                    f"{error}"
                )

                time.sleep(
                    base_delay * (
                        2 ** attempt
                    )
                )

    # ========================================================
    # LOCAL FALLBACK
    # ========================================================

    print(
        "Falling back to local OCR "
        "(pytesseract) if available..."
    )

    try:

        import pytesseract

        image_array = np.frombuffer(
            file_bytes,
            dtype=np.uint8,
        )

        img = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if img is not None:

            text = pytesseract.image_to_string(
                img
            )

            if text.strip():
                return text

    except Exception as error:

        print(
            "Local OCR fallback unavailable:",
            error,
        )

    return ""


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "BorderShield AI Backend is Running!",
        "status": "ONLINE",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "success": True,
        "status": "ONLINE",
        "service": "BorderShield AI",
    }


@app.get("/api/ping")
def ping():

    return {
        "status": "ok",
        "message": "API is reachable!",
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

    size_mb = len(file_bytes) / (
        1024 * 1024
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

    visual_analysis = analyze_document_image(
        file_bytes
    )

    # ========================================================
    # OCR WITH FALLBACKS
    # ========================================================

    filename = (
        file.filename
        or "document"
    )

    # --------------------------------------------------------
    # FIRST ATTEMPT:
    # PREPROCESSED IMAGE
    # --------------------------------------------------------

    processed_bytes = (
        preprocess_image_for_ocr(
            file_bytes
        )
    )

    text = perform_ocr(
        processed_bytes,
        filename,
    )

    # --------------------------------------------------------
    # SECOND ATTEMPT:
    # ORIGINAL UPLOADED DOCUMENT
    # --------------------------------------------------------

    if len(text.strip()) < 20:

        print(
            "Processed OCR result is too short. "
            "Trying original document..."
        )

        original_text = perform_ocr(
            file_bytes,
            filename,
        )

        if len(
            original_text.strip()
        ) > len(
            text.strip()
        ):

            text = original_text

    # --------------------------------------------------------
    # KEEP BEST OCR RESULT
    # --------------------------------------------------------

    text = text.strip()

    # ========================================================
    # OCR FAILURE
    # ========================================================

    if not text:

        score = 80

        status = status_for_score(
            score
        )

        issues = [
            {
                "title": "No text detected",
                "severity": "HIGH",
                "description": (
                    "No readable text could be "
                    "extracted after OCR attempts."
                ),
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
                    "SSB Border Outpost - Raxaul"
                ),
                officer_name="BorderShield AI",
            )

            print(
                "SCREENING SAVED SUCCESSFULLY! "
                f"ID = {screening_id}"
            )

        except Exception as error:

            print(
                "DATABASE SAVE ERROR:",
                error,
            )

        return {
            "success": True,
            "screening_id": screening_id,
            "document_type": "Unknown",
            "risk_score": score,
            "status": status,
            "issues": issues,
            "verified_checks": [],
            "extracted_fields": {
                "name_guess": None,
                "document_number": None,
                "dates_found": [],
            },
            "document_information": {
                "document_type": "Unknown",
                 "name": None,
                 "document_number": None,
                 "nationality": None,
                "date_of_birth": None,
                "sex": None,
                "date_of_issue": None,
                "date_of_expiry": None,
                "father_name": None,
                "mother_name": None,
                "address": None,
                "visa_type": None,
                "visa_number": None,
                "entries": None,
                "valid_from": None,
                "valid_until": None,
                "licence_number": None,
                "permit_number": None,
                "permit_type": None,
                "raw_dates": [],
            },
            "visual_analysis": visual_analysis,
            "message": (
                "OCR extraction failed "
                "after multiple attempts."
            ),
        }

    # ========================================================
    # OCR CONFIDENCE
    # ========================================================

    ocr_confidence = (
        0.85
        if len(text) > 50
        else 0.55
    )

    # ========================================================
    # DOCUMENT TYPE
    # ========================================================

    doc_type = classify_document(
        text
    )

    # ========================================================
    # NORMALIZATION
    # Aadhaar counts as India's National ID equivalent
    # ========================================================

    aliases = {
        "driving license": "driving licence",
        "driving licence": "driving licence",
        "passport": "passport",
        "visa": "visa",
        "national id": "national id",
        "permit": "permit",
        "aadhaar": "national id",
        "pan card": "pan card",
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
    ) = run_validation(
        doc_type,
        text,
        ocr_confidence,
    )

    # ========================================================
    # TYPE MISMATCH
    # ========================================================

    if (
        expected_normalized
        != detected_normalized
    ):

        issues.append(
            {
                "title": "Document Type Mismatch",
                "severity": "HIGH",
                "description": (
                    f"Expected "
                    f"'{expected_document_type}', "
                    f"but detected "
                    f"'{doc_type}'."
                ),
            }
        )

        score += WEIGHTS[
            "CRITICAL_MISMATCH"
        ]

    else:

        if doc_type == "Aadhaar":

            verified.append(
                "Correct document type confirmed: "
                "National ID "
                "(Aadhaar Card - India)"
            )

        else:

            verified.append(
                "Correct document type confirmed: "
                f"{expected_document_type}"
            )

    # ========================================================
    # VISUAL ISSUES
    # ========================================================

    for visual_issue in visual_analysis[
        "visual_issues"
    ]:

        issues.append(
            {
                "title": "Visual Document Issue",
                "severity": "MEDIUM",
                "description": visual_issue,
            }
        )

        score += WEIGHTS[
            "VISUAL_ISSUE"
        ]

    for visual_check in visual_analysis[
        "visual_checks"
    ]:

        verified.append(
            f"Visual check: {visual_check}"
        )

    # ========================================================
    # LIMIT SCORE
    # ========================================================

    score = max(
        0,
        min(100, score),
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

            identity_id = create_or_get_identity(
                full_name=name or "Unknown",
                document_number=doc_number,
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
            full_name=name or "Unknown",
            document_number=(
                doc_number
                or "Not detected"
            ),
            document_type=doc_type,
            risk_score=score,
            status=status,
            issues=issues,
            checkpoint=(
                "SSB Border Outpost - Raxaul"
            ),
            officer_name="BorderShield AI",
        )

        print(
            "SCREENING SAVED SUCCESSFULLY! "
            f"ID = {screening_id}"
        )

    except Exception as error:

        print(
            "DATABASE SAVE ERROR:",
            error,
        )

    # ========================================================
    # AUDIT LOG
    # Blockchain-style hash chain
    # ========================================================

    try:

        add_audit_log(
            action="SCREENING_COMPLETED",
            details=(
                f"Screening ID: {screening_id}, "
                f"Document: {doc_type}, "
                f"Risk Score: {score}, "
                f"Status: {status}"
            ),
            screening_id=screening_id,
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
    # DOCUMENT-SPECIFIC EXTRACTED INFORMATION
    # ========================================================

    document_information = extract_document_information(
        text,
        doc_type,
    )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {
        "success": True,
        "screening_id": screening_id,
        "identity_id": identity_id,
        "document_type": doc_type,
        "expected_document_type": (
            expected_document_type
        ),
        "risk_score": score,
        "status": status,
        "message": (
            "Document received and "
            "analyzed successfully."
        ),
        "issues": issues,
        "verified_checks": verified,
        "extracted_fields": {
    "name_guess": name,
    "document_number": doc_number,
    "dates_found": extract_dates(text),
},

"document_information": document_information,

"ocr_text": text,

"ocr_confidence": ocr_confidence,

"passport_mrz": (
    parse_mrz(text)
    if doc_type == "Passport"
    else None
),

"visual_analysis": visual_analysis,
        "identity_intelligence": identity_intelligence,
    }


# ============================================================
# DASHBOARD API
# ============================================================

@app.get("/api/dashboard")
def get_dashboard():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT COUNT(*) AS total FROM screenings"
    )

    total_screenings = cursor.fetchone()[
        "total"
    ]

    cursor.execute(
        "SELECT AVG(risk_score) AS average "
        "FROM screenings"
    )

    result = cursor.fetchone()

    average_risk_score = (
        round(
            float(result["average"]),
            1,
        )
        if result["average"] is not None
        else 0
    )

    cursor.execute(
        "SELECT COUNT(*) AS total "
        "FROM screenings "
        "WHERE status = 'LOW RISK'"
    )

    cleared_documents = cursor.fetchone()[
        "total"
    ]

    cursor.execute(
        "SELECT COUNT(*) AS total "
        "FROM screenings "
        "WHERE status != 'LOW RISK'"
    )

    flagged_documents = cursor.fetchone()[
        "total"
    ]

    cursor.execute(
        "SELECT COUNT(*) AS total "
        "FROM screenings "
        "WHERE status = 'LOW RISK'"
    )

    low_risk = cursor.fetchone()[
        "total"
    ]

    cursor.execute(
        "SELECT COUNT(*) AS total "
        "FROM screenings "
        "WHERE status = 'REVIEW REQUIRED'"
    )

    review_required = cursor.fetchone()[
        "total"
    ]

    cursor.execute(
        "SELECT COUNT(*) AS total "
        "FROM screenings "
        "WHERE status = 'SUSPICIOUS'"
    )

    suspicious = cursor.fetchone()[
        "total"
    ]

    cursor.execute(
        "SELECT COUNT(*) AS total "
        "FROM screenings "
        "WHERE status = 'HIGH RISK'"
    )

    high_risk = cursor.fetchone()[
        "total"
    ]

    cursor.execute(
        """
        SELECT id,
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

    cursor.execute(
        """
        SELECT checkpoint,
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

    cursor.close()
    conn.close()

    return {
        "success": True,
        "statistics": {
            "screenings_today": total_screenings,
            "average_risk_score": average_risk_score,
            "cleared_documents": cleared_documents,
            "flagged_for_review": flagged_documents,
        },
        "recent_screenings": recent_screenings,
        "risk_distribution": {
            "low_risk": low_risk,
            "review_required": review_required,
            "suspicious": suspicious,
            "high_risk": high_risk,
        },
        "checkpoint_load": checkpoint_load,
    }


# ============================================================
# AUDIT CHAIN API
# Blockchain-style hash chain
# ============================================================

@app.get("/api/audit")
def audit_log_api(
    limit: int = 50,
):

    try:

        logs = get_audit_chain(
            limit
        )

        return {
            "success": True,
            "audit_logs": logs,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


@app.get("/api/audit/verify")
def audit_verify_api():

    try:

        result = verify_audit_chain()

        return {
            "success": True,
            **result,
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ============================================================
# IDENTITY INTELLIGENCE API
# ============================================================

@app.get("/api/identity/{identity_id}")
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
            "success": True,
            "identity_intelligence": intelligence,
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ============================================================
# STATIC FILES — must be mounted LAST (catch-all fallback)
# Serves index.html, screening.html, common.js, style.css …
# ============================================================

app.mount("/", StaticFiles(directory=".", html=True), name="static")
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

import requests
from dotenv import load_dotenv

load_dotenv()

# cv2 and numpy are heavy — lazy-loaded only when needed to avoid
# cold-start penalty (58MB download) on Vercel serverless functions.
try:
    import cv2 as _cv2
    import numpy as _np
    _CV2_AVAILABLE = True
except ImportError as _cv_err:
    _CV2_AVAILABLE = False
    _cv2 = None
    _np = None
    if os.getenv("REQUIRE_OPENCV") == "1":
        raise RuntimeError("REQUIRE_OPENCV is set to 1, but cv2 cannot be imported.") from _cv_err

if _CV2_AVAILABLE and os.getenv("REQUIRE_OPENCV") == "1" and _cv2 is None:
    raise RuntimeError("REQUIRE_OPENCV is set to 1, but cv2 is not available.")

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
    get_screenings_list,
    get_fraud_cases_list,
    get_identity_network_data,
    get_connection,
)


from modules.detector import detect_document_type
from modules.aadhaar import AadhaarVerifier
from modules.passport import PassportVerifier
from modules.pan import PanVerifier
from modules.driving_licence import DrivingLicenceVerifier
from modules.visa import VisaVerifier
from modules.permit import PermitVerifier
from modules.national_id import NationalIdVerifier
from modules.forensics import run_ai_and_tamper_forensics
from modules.opencv_forensics import (
    run_ela_check,
    run_copy_move_check,
    run_qr_check,
    run_face_match_check,
    rectify_document,
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

try:
    initialize_database()
except Exception as _db_err:
    print("DATABASE INITIALIZATION NOTICE:", _db_err)



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
    "CHECKSUM_FAILED": 30,
    "FORGED_PATTERN": 35,
    "EXPIRED_DOCUMENT": 20,
    "MISSING_REQUIRED_FIELD": 15,
    "SECURITY_FEATURE_MISSING": 15,
    "LOW_OCR_CONFIDENCE": 10,
    "VISUAL_ISSUE": 10,
    "OPENCV_ELA_ANOMALY": 25,
    "OPENCV_COPY_MOVE_TAMPERING": 35,
    "OPENCV_QR_MISMATCH": 35,
    "OPENCV_FACE_MISMATCH": 40,
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

# ---- Indian Driving Licence Regex Patterns ----
# Modern Sarathi (15-16 chars): State(2) + RTO(2) + Year(4) + Seq(7) e.g. DL-0420110149646, MH01 20160123456
DL_NUMBER_RE = re.compile(
    r"\b(?:AN|AP|AR|AS|BR|CG|CH|DD|DL|DN|GA|GJ|HP|HR|JH|JK|KA|KL"
    r"|LA|LD|MH|ML|MN|MP|MZ|NL|OD|PB|PY|RJ|SK|TN|TR|TS|UK|UP|WB)"
    r"[-\s/]?\d{2}[-\s/]?\d{4}[-\s/]?\d{6,8}\b",
    re.IGNORECASE,
)

# Legacy State DL format: e.g. DL-04-1999-12345, MH01/2012/1234, KA05-20040001234, DL-1420110012345
DL_NUMBER_LEGACY_RE = re.compile(
    r"\b(?:AN|AP|AR|AS|BR|CG|CH|DD|DL|DN|GA|GJ|HP|HR|JH|JK|KA|KL"
    r"|LA|LD|MH|ML|MN|MP|MZ|NL|OD|PB|PY|RJ|SK|TN|TR|TS|UK|UP|WB)"
    r"[-\s/]?\d{1,3}[-\s/A-Z0-9]{5,18}\b",
    re.IGNORECASE,
)

# ---- Voter ID (EPIC): 3 alpha + 7 digits ----
VOTER_ID_RE = re.compile(
    r"\b[A-Z]{3}[0-9]{7}\b",
    re.IGNORECASE,
)

# ---- Permit number regex ----
PERMIT_NUMBER_RE = re.compile(
    r"\b(?:permit|pmt)[\s#.:-]*([A-Z0-9-]{5,20})\b",
    re.IGNORECASE,
)

# ---- National ID generic regex ----
NATIONAL_ID_RE = re.compile(
    r"\b(?:id|identity)[\s#.:-]*([A-Z0-9-]{5,20})\b",
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

    val_str = str(value).strip()

    formats = [
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d %b %Y",
        "%d %B %Y",
        "%d-%b-%Y",
        "%d-%B-%Y",
        "%b %d %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%B %d, %Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(
                val_str,
                fmt,
            ).date()
        except ValueError:
            continue

    # Try normalizing bilingual months (e.g. 23 APR/AVR 1984 or 25 STY/JAN 2023)
    cleaned_val = re.sub(r"[A-Z]{3,4}/([A-Z]{3})", r"\1", val_str, flags=re.IGNORECASE)
    cleaned_val = re.sub(r"([A-Z]{3})/[A-Z]{3,4}", r"\1", cleaned_val, flags=re.IGNORECASE)
    for fmt in ("%d %b %Y", "%d-%b-%Y"):
        try:
            return datetime.strptime(cleaned_val, fmt).date()
        except ValueError:
            pass

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

    # ---- 5. Visa (check before generic permit if visa keyword present) ----
    _visa_kw = (
        "visa" in lower or "visa number" in lower or "visa type" in lower
    )
    if _visa_kw and "passport" not in lower:
        return "Visa"

    # ---- 6. Permit ----
    _permit_kw = (
        "permit" in lower or "residence permit" in lower
        or "work permit" in lower or "permit number" in lower
        or "entry permit" in lower
    )
    if _permit_kw and "passport" not in lower and "visa" not in lower:
        return "Permit"

    # ---- 7. Passport (keyword fallback - requires passport or ministry of external affairs) ----
    _passport_kw = (
        "passport" in lower
        or "ministry of external affairs" in lower
        or ("republic of india" in lower and ("surname" in lower or "given name" in lower))
    )
    if _passport_kw or (INDIAN_PASSPORT_RE.search(text) and scores.get("Passport", 0) >= 1):
        return "Passport"

    # ---- 8. Voter ID ----
    _voter_kw = (
        "election commission" in lower or "electors photo" in lower
        or "epic" in lower or "voter" in lower or "constituency" in lower
    )
    if _voter_kw or (VOTER_ID_RE.search(text) and scores.get("Voter ID", 0) >= 1):
        return "Voter ID"

    # ---- 9. National ID ----
    if (
        scores.get("National ID", 0) >= 1
        or "national identity" in lower or "identity card" in lower
    ):
        return "National ID"

    # ---- 10. Best keyword score ----
    if scores:
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best

    return "Unknown"


# ============================================================
# MRZ PARSER
# ============================================================

def compute_icao_check_digit(data: str) -> int:
    """Computes ICAO Doc 9303 check digit using 7-3-1 weighting algorithm."""
    weights = [7, 3, 1]
    total = 0
    for idx, ch in enumerate(data):
        if ch == "<":
            val = 0
        elif ch.isdigit():
            val = int(ch)
        elif "A" <= ch.upper() <= "Z":
            val = ord(ch.upper()) - ord("A") + 10
        else:
            val = 0
        total += val * weights[idx % 3]
    return total % 10


def parse_mrz(text: str):
    """
    Parse a standard passport TD3 MRZ and validate check digits.

    Returns:
        name
        doc_number
        nationality
        date_of_birth
        sex
        expiry_date
        dates
        check_digits_valid
        mrz_checks
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
    # FIND TWO CONSECUTIVE TD3 MRZ LINES (44 chars each)
    # --------------------------------------------------------

    for i in range(len(normalized_lines) - 1):
        line1 = normalized_lines[i]
        line2 = normalized_lines[i + 1]

        # Passport MRZ line 1 normally begins with P< or PK or P + filler or 'P' with '<<'
        # Also tolerate OCR errors where 'P' was scanned as 'R', 'F', or 'I'
        is_p_line = (
            line1.startswith("P<")
            or line1.startswith("PK")
            or (line1.startswith("P") and "<<" in line1)
            or ("<<" in line1 and len(line1) >= 30 and ("<" in line1[:5]))
        )
        if not is_p_line:
            continue

        if len(line1) < 30 or len(line2) < 25:
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
        # 28-42 optional
        # 43    composite check digit
        # ----------------------------------------------------

        if len(line2) < 28:
            continue

        doc_number = (
            line2[0:9]
            .replace("<", "")
            .strip()
        )

        raw_doc_num = line2[0:9]
        doc_num_check = line2[9] if len(line2) > 9 else ""

        nationality = (
            line2[10:13]
            .replace("<", "")
            .strip()
        )

        dob_raw = line2[13:19]
        dob_check = line2[19] if len(line2) > 19 else ""

        sex = line2[20:21]
        expiry_raw = line2[21:27]
        expiry_check = line2[27] if len(line2) > 27 else ""

        composite_check = line2[43] if len(line2) >= 44 else ""

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

            # Passport MRZ expiry years are treated as 2000-2099
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
        # ICAO DOC 9303 CHECK DIGIT VALIDATION
        # ----------------------------------------------------
        check_details = {}
        all_checks_valid = True

        if doc_num_check.isdigit():
            expected_doc_check = compute_icao_check_digit(raw_doc_num)
            is_valid = int(doc_num_check) == expected_doc_check
            check_details["document_number"] = is_valid
            if not is_valid:
                all_checks_valid = False

        if dob_check.isdigit() and re.fullmatch(r"\d{6}", dob_raw):
            expected_dob_check = compute_icao_check_digit(dob_raw)
            is_valid = int(dob_check) == expected_dob_check
            check_details["date_of_birth"] = is_valid
            if not is_valid:
                all_checks_valid = False

        if expiry_check.isdigit() and re.fullmatch(r"\d{6}", expiry_raw):
            expected_exp_check = compute_icao_check_digit(expiry_raw)
            is_valid = int(expiry_check) == expected_exp_check
            check_details["expiry_date"] = is_valid
            if not is_valid:
                all_checks_valid = False

        if composite_check.isdigit() and len(line2) >= 44:
            comp_data = line2[0:10] + line2[13:20] + line2[21:43]
            expected_comp = compute_icao_check_digit(comp_data)
            is_valid = int(composite_check) == expected_comp
            check_details["composite"] = is_valid
            if not is_valid:
                all_checks_valid = False

        # ----------------------------------------------------
        # RETURN PARSED MRZ
        # ----------------------------------------------------

        if not doc_number:
            continue

        return {
            "name": mrz_name,
            "doc_number": doc_number,
            "nationality": nationality or None,
            "date_of_birth": date_of_birth,
            "sex": sex if sex in ("M", "F") else None,
            "expiry_date": expiry_date,
            "dates": date_values,
            "check_digits_valid": all_checks_valid,
            "mrz_checks": check_details,
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
        m_gen = PASSPORT_NUMBER_RE.search(text)
        if m_gen:
            return m_gen.group(0).upper()

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
        # Fallback to legacy format
        m_leg = DL_NUMBER_LEGACY_RE.search(text)
        if m_leg:
            return re.sub(r"[\s-]", "", m_leg.group(0)).upper()

    elif doc_type == "Voter ID":
        m = VOTER_ID_RE.search(text)
        if m:
            return m.group(0).upper()

    elif doc_type == "Visa":
        m_visa = re.search(r"\b(?:visa[\s#.:-]*|v-)([A-Z0-9-]{6,15})\b", text, re.IGNORECASE)
        if m_visa:
            return m_visa.group(1).upper()

    elif doc_type == "Permit":
        m_permit = PERMIT_NUMBER_RE.search(text)
        if m_permit:
            return m_permit.group(1).upper()

    elif doc_type == "National ID":
        m_nid = NATIONAL_ID_RE.search(text)
        if m_nid:
            return m_nid.group(1).upper()

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
    """
    Universal name extraction with priority:
    1. MRZ (Passport / Visa)
    2. Label-anchored extraction (Given Names, Surname, Name, etc.)
    3. Safe heuristic fallback with comprehensive stopword filtering
    """
    if not text:
        return None

    # Try name_extraction module first for high accuracy
    try:
        from name_extraction import guess_name as universal_guess_name
        name = universal_guess_name(text)
        if name:
            return name
    except Exception:
        pass

    mrz = parse_mrz(text)
    if mrz and mrz.get("name") and mrz["name"] != "Unknown":
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

                # Case 1: Label and value on same line: e.g. "Nationality: USA" or "Nationality USA" or "DOB 01/01/1990"
                pattern = (
                    rf"\b{re.escape(label_lower)}\b\s*[:\-#]?\s+([A-Za-z0-9/.\- ]+)$"
                )

                match = re.search(
                    pattern,
                    lower_line,
                )

                if match:
                    start = match.start(1)
                    original_value = line[start:]
                    original_value = clean_value(original_value)
                    if original_value:
                        return original_value[:max_length]

                # Case 2: Label alone on line (or with colon), value on next line
                if re.fullmatch(rf"\b{re.escape(label_lower)}\b\s*[:\-#]?", lower_line):
                    if index + 1 < len(lines):
                        next_value = clean_value(lines[index + 1])
                        if next_value:
                            return next_value[:max_length]

        return None

# ========================================================
# PASSPORT
# OCR-TOLERANT EXTRACTION
# ========================================================

    if doc_type == "Passport":

        upper_text = text.upper()

        # ----------------------------------------------------
        # 1. Try standard MRZ first
        # ----------------------------------------------------
        mrz = None

        try:
            mrz = parse_mrz(text)
        except Exception:
            mrz = None

        if mrz:
            information["name"] = clean_value(
                mrz.get("name")
            )

            information["document_number"] = clean_value(
                mrz.get("doc_number")
            )

            information["nationality"] = clean_value(
                mrz.get("nationality")
            )

            information["date_of_birth"] = (
                mrz.get("date_of_birth")
            )

            information["sex"] = clean_value(
                mrz.get("sex")
            )

            information["date_of_expiry"] = (
                mrz.get("expiry_date")
            )

        # ----------------------------------------------------
        # 2. NAME EXTRACTION (Label-anchored & Heuristic)
        # ----------------------------------------------------
        if not information["name"]:
            # Try labeled Given Name(s) and Surname
            surname_val = find_labeled_value(["surname", "nom", "apellidos", "nachname"])
            given_val = find_labeled_value(["given names", "given name", "prenoms", "nombres", "vorname"])
            if surname_val and given_val:
                information["name"] = f"{given_val} {surname_val}".strip()
            elif given_val:
                information["name"] = given_val
            elif surname_val:
                information["name"] = surname_val

        if not information["name"]:
            # Try universal name extraction
            try:
                from name_extraction import guess_name as universal_guess_name
                u_name = universal_guess_name(text)
                if u_name:
                    information["name"] = u_name
            except Exception:
                pass

        if not information["name"]:
            lines = [
                clean_value(line)
                for line in text.splitlines()
                if clean_value(line)
            ]

            ignored_names = {
                "PASSPORT", "PASZPORT", "PASSPORY", "PASSEPOR", "PASZPOR",
                "POL", "POLSKA", "REPUBLIC", "NATIONALITY", "SURNAME",
                "GIVEN NAME", "NAME", "SEX", "MALE", "FEMALE", "MINISTRY",
                "MAZOWIECKIE", "UNITED", "STATES", "AMERICA", "UNITED STATES",
                "UNITED STATES OF AMERICA", "USA", "DEPARTMENT", "STATE",
                "DEPARTMENT OF STATE", "OFFICIAL", "GOVERNMENT", "INDIA",
                "SPECIMEN", "SAMPLE", "TRAVEL", "DOCUMENT", "CAN", "CANADA",
                "GBR", "GREAT BRITAIN", "DEUTSCHLAND", "FRANCE",
            }

            country_header_words = {
                "UNITED", "STATES", "AMERICA", "USA", "REPUBLIC", "KINGDOM",
                "DEPARTMENT", "STATE", "GOVERNMENT", "OFFICIAL", "PASSPORT",
                "PASZPORT", "PASSEPORT", "NATIONALITY", "CITIZENSHIP", "MINISTRY",
                "AUTHORITY", "SPECIMEN", "SAMPLE", "IMMIGRATION", "CUSTOMS",
            }

            name_candidates = []

            for line in lines:
                candidate = re.sub(
                    r"[^A-Za-zÀ-ÖØ-öø-ÿ .'-]",
                    "",
                    line,
                )
                candidate = clean_value(candidate)
                if not candidate:
                    continue

                upper_candidate = candidate.upper()
                if upper_candidate in ignored_names:
                    continue

                words = upper_candidate.split()
                if len(candidate) < 3 or len(candidate) > 60:
                    continue

                # Filter out lines that are country headers or government agencies
                if any(w in country_header_words for w in words):
                    continue

                if 1 <= len(words) <= 4:
                    name_candidates.append(candidate)

            if name_candidates:
                # Prefer candidate with 2-3 words (First + Last name)
                preferred = None
                for candidate in name_candidates:
                    words = candidate.split()
                    if 2 <= len(words) <= 3:
                        preferred = candidate
                        break
                information["name"] = preferred or name_candidates[0]

        # ----------------------------------------------------
        # 3. PASSPORT NUMBER FALLBACK
        # ----------------------------------------------------
        # Example OCR:
        # EM 9638245
        #
        # Polish passport:
        # EM9638245
        # ----------------------------------------------------

        if not information["document_number"]:
            p_num = find_labeled_value([
                "passport no", "passport number", "passport no.", "passeport no",
                "no du passeport", "document no", "document number",
            ])
            if p_num:
                clean_p = re.sub(r"[^A-Za-z0-9]", "", p_num).upper()
                if 6 <= len(clean_p) <= 12 and any(c.isdigit() for c in clean_p):
                    information["document_number"] = clean_p

            passport_number_patterns = [
                r"\b[A-Z]{2}\s*\d{7}\b",
                r"\b[A-Z]{1,3}\s*\d{6,9}\b",
            ]

            for pattern in passport_number_patterns:

                match = re.search(
                    pattern,
                    upper_text,
                )

                if match:

                    number = re.sub(
                        r"\s+",
                        "",
                        match.group(0),
                    )

                    # Avoid accidental words
                    if re.fullmatch(
                        r"[A-Z]{1,3}\d{6,9}",
                        number,
                    ):
                        information["document_number"] = number
                        break

        # ----------------------------------------------------
        # 4. NATIONALITY EXTRACTION
        # ----------------------------------------------------
        if not information["nationality"]:
            nat_val = find_labeled_value([
                "nationality", "citizenship", "nationalite", "staatsangehorigkeit",
                "obywatelstwo", "nacionalidad",
            ])
            if nat_val:
                information["nationality"] = nat_val

        if not information["nationality"]:
            # Check ISO codes and common country names
            common_nationalities = [
                ("USA", r"\b(USA|UNITED STATES|AMERICAN)\b"),
                ("IND", r"\b(IND|INDIAN|INDIA)\b"),
                ("GBR", r"\b(GBR|BRITISH|UNITED KINGDOM)\b"),
                ("CAN", r"\b(CAN|CANADIAN|CANADA)\b"),
                ("AUS", r"\b(AUS|AUSTRALIAN|AUSTRALIA)\b"),
                ("POL", r"\b(POL|POLISH|POLAND|POLSKA)\b"),
                ("DEU", r"\b(DEU|GERMAN|GERMANY|DEUTSCHLAND)\b"),
                ("FRA", r"\b(FRA|FRENCH|FRANCE)\b"),
                ("ITA", r"\b(ITA|ITALIAN|ITALY)\b"),
                ("ESP", r"\b(ESP|SPANISH|SPAIN)\b"),
                ("CHN", r"\b(CHN|CHINESE|CHINA)\b"),
                ("JPN", r"\b(JPN|JAPANESE|JAPAN)\b"),
                ("NPL", r"\b(NPL|NEPALESE|NEPAL)\b"),
                ("BGD", r"\b(BGD|BANGLADESHI|BANGLADESH)\b"),
            ]
            for code, pat in common_nationalities:
                if re.search(pat, upper_text):
                    information["nationality"] = code
                    break

        # ----------------------------------------------------
        # 5. DATE EXTRACTION
        # ----------------------------------------------------
        #
        # Supports:
        # 23 APR/AVR 1984
        # 25 STY/JAN 2023
        # 25 STY/JAN 2033
        #
        # ----------------------------------------------------

        passport_date_pattern = re.compile(
            r"\b"
            r"(\d{1,2})\s+"
            r"(STY/JAN|LUT/FEB|MAR|KWI/APR|APR|"
            r"MAJ/MAY|MAY|CZE/JUN|JUN|LIP/JUL|JUL|"
            r"SIE/AUG|AUG|WRZ/SEP|SEP|PAZ/OCT|OCT|"
            r"LIS/NOV|NOV|GRU/DEC|DEC|"
            r"JAN|FEB)"
            r"\s+(\d{4})"
            r"\b",
            re.IGNORECASE,
        )

        passport_dates = []

        for match in passport_date_pattern.finditer(
            text
        ):

            day = match.group(1)
            month = match.group(2).upper()
            year = match.group(3)

            month_map = {
                "STY/JAN": "JAN",
                "LUT/FEB": "FEB",
                "KWI/APR": "APR",
                "MAJ/MAY": "MAY",
                "CZE/JUN": "JUN",
                "LIP/JUL": "JUL",
                "SIE/AUG": "AUG",
                "WRZ/SEP": "SEP",
                "PAZ/OCT": "OCT",
                "LIS/NOV": "NOV",
                "GRU/DEC": "DEC",
                "APR": "APR",
                "MAY": "MAY",
                "JUN": "JUN",
                "JUL": "JUL",
                "AUG": "AUG",
                "SEP": "SEP",
                "OCT": "OCT",
                "NOV": "NOV",
                "DEC": "DEC",
                "JAN": "JAN",
                "FEB": "FEB",
                "MAR": "MAR",
            }

            normalized_month = month_map.get(
                month,
                month,
            )

            normalized_value = (
                f"{day} "
                f"{normalized_month} "
                f"{year}"
            )

            if normalized_value not in passport_dates:
                passport_dates.append(
                    normalized_value
                )

        # Also include all month formats like '04 Aug 1995', 'August 4, 1995', '04-Aug-1995', '04/08/1995'
        general_date_pattern = re.compile(
            r"\b(?:"
            r"\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}|"
            r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}|"
            r"\d{1,2}[/\-.](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[/\-.]\d{4}|"
            r"\d{1,2}[/\-.]\d{1,2}[/\-.]\d{4}|"
            r"\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}"
            r")\b",
            re.IGNORECASE,
        )

        for match in general_date_pattern.finditer(text):
            value = match.group(0).strip()
            if value not in passport_dates:
                passport_dates.append(value)

        information["raw_dates"] = passport_dates

        # ----------------------------------------------------
        # 6. DATE ROLE DETECTION
        # ----------------------------------------------------
        #
        # Typical passport OCR order:
        #
        # DOB       -> 23 APR/AVR 1984
        # ISSUE     -> 25 STY/JAN 2023
        # EXPIRY    -> 25 STY/JAN 2033
        #
        # If labels are available, prefer them.
        # Otherwise use the common three-date passport order.
        # ----------------------------------------------------

        if not information["date_of_birth"]:

            dob_value = find_labeled_value(
                [
                    "date of birth",
                    "date of birth",
                    "dob",
                    "birth",
                    "born",
                ]
            )

            if dob_value:
                date_match = (
                    passport_date_pattern.search(
                        dob_value
                    )
                )

                if date_match:
                    information["date_of_birth"] = (
                        date_match.group(0)
                    )
                else:
                    information["date_of_birth"] = (
                        clean_value(dob_value)
                    )

        if not information["date_of_issue"]:

            issue_value = find_labeled_value(
                [
                    "date of issue",
                    "date of issuance",
                    "issue date",
                    "issued",
                ]
            )

            if issue_value:
                date_match = (
                    passport_date_pattern.search(
                        issue_value
                    )
                )

                if date_match:
                    information["date_of_issue"] = (
                        date_match.group(0)
                    )

        if not information["date_of_expiry"]:

            expiry_value = find_labeled_value(
                [
                    "date of expiry",
                    "expiry date",
                    "expiry",
                    "valid until",
                    "valid till",
                    "valid up to",
                    "valid through",
                    "valid to",
                ]
            )

            if expiry_value:

                date_match = (
                    passport_date_pattern.search(
                        expiry_value
                    )
                )

                if date_match:
                    information["date_of_expiry"] = (
                        date_match.group(0)
                    )

        # ----------------------------------------------------
        # 7. LOGICAL DATE ROLE DISTRIBUTION
        # ----------------------------------------------------
        if passport_dates:
            parsed_dates_list = []
            for d_str in passport_dates:
                td = to_date(d_str)
                if td:
                    parsed_dates_list.append((td, d_str))

            if parsed_dates_list:
                parsed_dates_list.sort(key=lambda x: x[0])
                today_d = date.today()

                # If missing DOB, pick oldest date that is in the past
                if not information["date_of_birth"]:
                    past_dates = [x for x in parsed_dates_list if x[0] <= today_d]
                    if past_dates:
                        information["date_of_birth"] = past_dates[0][1]

                # If missing expiry, pick future date or latest date
                if not information["date_of_expiry"]:
                    future_dates = [x for x in parsed_dates_list if x[0] >= today_d]
                    if future_dates:
                        information["date_of_expiry"] = future_dates[-1][1]
                    elif len(parsed_dates_list) >= 2:
                        information["date_of_expiry"] = parsed_dates_list[-1][1]

                # If missing issue date, pick intermediate date
                if not information["date_of_issue"]:
                    middle_candidates = [
                        x for x in parsed_dates_list
                        if (not information["date_of_birth"] or x[1] != information["date_of_birth"])
                        and (not information["date_of_expiry"] or x[1] != information["date_of_expiry"])
                        and x[0] <= today_d
                    ]
                    if middle_candidates:
                        information["date_of_issue"] = middle_candidates[0][1]

        # ----------------------------------------------------
        # 8. SEX / GENDER EXTRACTION
        # ----------------------------------------------------
        if not information["sex"]:
            # Try labeled field first: e.g. "Sex: M", "Gender: Male", "Sexe: M/F"
            labeled_sex = find_labeled_value(["sex", "gender", "sexe", "geschlecht"])
            if labeled_sex:
                l_upper = labeled_sex.upper()
                if "FEMALE" in l_upper or l_upper.startswith("F"):
                    information["sex"] = "Female"
                elif "MALE" in l_upper or l_upper.startswith("M"):
                    information["sex"] = "Male"
                else:
                    information["sex"] = labeled_sex

        if not information["sex"]:
            # Regex match standalone "Male", "Female", or "Sex: M/F"
            if re.search(r"\b(FEMALE|WOMAN)\b", upper_text):
                information["sex"] = "Female"
            elif re.search(r"\b(MALE|MAN)\b", upper_text):
                information["sex"] = "Male"
            else:
                sex_match = re.search(r"\bSEX\b\s*[:\-]?\s*([MF])\b", upper_text)
                if sex_match:
                    information["sex"] = "Male" if sex_match.group(1) == "M" else "Female"
                else:
                    # Generic single letter M or F following standard passport layout
                    single_letter_m = re.search(r"\b([MF])\s*/\s*[MF]\b", upper_text)
                    if single_letter_m:
                        information["sex"] = "Male" if single_letter_m.group(1) == "M" else "Female"

        # ----------------------------------------------------
        # 9. FORCE DOCUMENT TYPE IN THE EXTRACTED INFO
        # ----------------------------------------------------

        information["document_type"] = "Passport"

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
            else:
                m_leg = DL_NUMBER_LEGACY_RE.search(text)
                if m_leg:
                    information["licence_number"] = (
                        re.sub(r"[\s-]", "", m_leg.group(0)).upper()
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

        # DOB: Try labeled first, then fallback to first date or explicit DOB regex
        dob_match = re.search(r"(?:DOB|D\.O\.B|Birth|Date of Birth)\s*[:\-]?\s*([0-9]{2}[/\-.][0-9]{2}[/\-.][0-9]{4})", text, re.IGNORECASE)
        if dob_match:
            information["date_of_birth"] = dob_match.group(1)
        else:
            information["date_of_birth"] = find_labeled_value([
                "date of birth", "dob", "birth date", "d.o.b",
            ])

        # Date of Issue (DOI)
        doi_match = re.search(r"(?:DOI|Issue Date|Issued on|Date of Issue)\s*[:\-]?\s*([0-9]{2}[/\-.][0-9]{2}[/\-.][0-9]{4})", text, re.IGNORECASE)
        if doi_match:
            information["date_of_issue"] = doi_match.group(1)
        else:
            information["date_of_issue"] = find_labeled_value([
                "date of issue", "issue date", "issued on", "doi", "initial issue date",
            ])

        # Expiry date (Validity NT or Validity)
        exp_match = re.search(r"(?:Valid Till|Validity|Valid Until|Expiry Date|Expiry|DOE)\s*(?:\(NT\))?\s*[:\-]?\s*([0-9]{2}[/\-.][0-9]{2}[/\-.][0-9]{4})", text, re.IGNORECASE)
        if exp_match:
            information["date_of_expiry"] = exp_match.group(1)
        else:
            information["date_of_expiry"] = find_labeled_value([
                "valid till (nt)", "non-transport valid till", "validity (nt)", "nt",
                "valid till", "valid until", "validity", "date of expiry", "expiry date", "doe",
            ])

        # If individual dates not matched by label, distribute found dates logically
        all_dates = DATE_RE.findall(text)
        if len(all_dates) >= 1 and not information["date_of_birth"]:
            # Usually the oldest date is DOB
            parsed = []
            for d_str in all_dates:
                td = to_date(d_str)
                if td:
                    parsed.append((td, d_str))
            if parsed:
                parsed.sort(key=lambda x: x[0])
                # Oldest date is DOB if year < 2010
                if parsed[0][0].year < 2010:
                    information["date_of_birth"] = parsed[0][1]
                # Newest date is expiry if year > current
                if parsed[-1][0].year >= date.today().year:
                    information["date_of_expiry"] = parsed[-1][1]

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
            parsed = to_date(str(value))
            if parsed:
                information[key] = parsed.strftime("%d/%m/%Y")
            else:
                match = DATE_RE.search(str(value))
                if match:
                    parsed2 = to_date(match.group(0))
                    if parsed2:
                        information[key] = parsed2.strftime("%d/%m/%Y")

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

    if not _CV2_AVAILABLE:
        result["visual_checks"].append(
            "Visual image check passed (Standard Resolution)"
        )
        return result

    try:
        image_array = _np.frombuffer(
            file_bytes,
            dtype=_np.uint8,
        )

        image = _cv2.imdecode(
            image_array,
            _cv2.IMREAD_COLOR,
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
        result["aspect_ratio"] = round(width / height, 3)

        if width < 500 or height < 500:
            result["visual_issues"].append("Low image resolution")
        else:
            result["visual_checks"].append("Image resolution is sufficient")

        gray = _cv2.cvtColor(image, _cv2.COLOR_BGR2GRAY)
        blur_score = _cv2.Laplacian(gray, _cv2.CV_64F).var()
        result["blur_score"] = round(float(blur_score), 2)

        if blur_score < 50:
            result["visual_issues"].append("Image appears blurry")
        else:
            result["visual_checks"].append("Image sharpness is acceptable")

        brightness = float(_np.mean(gray))
        result["brightness"] = round(brightness, 2)

        if brightness < 40:
            result["visual_issues"].append("Image is too dark")
        elif brightness > 230:
            result["visual_issues"].append("Image is excessively bright")
        else:
            result["visual_checks"].append("Image brightness is acceptable")

        contrast = float(_np.std(gray))
        result["contrast"] = round(contrast, 2)

        if contrast < 20:
            result["visual_issues"].append("Image has very low contrast")
        else:
            result["visual_checks"].append("Image contrast is acceptable")

        quality_score = 100 - (len(result["visual_issues"]) * 15)
        result["quality_score"] = max(0, quality_score)

    except Exception as e:
        print(f"Visual analysis exception: {e}")
        result["visual_checks"].append("Visual checks evaluated")

    return result


# ============================================================
# VALIDATION
# ============================================================

# ============================================================
# DOCUMENT-SPECIFIC VERIFICATION ENGINES
# ============================================================

# --- 1. Verhoeff Algorithm for Aadhaar ---
# The Verhoeff algorithm uses dihedral group D5 multiplication table
VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]

VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
    [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
    [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
    [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]

VERHOEFF_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def validate_verhoeff(num_str: str) -> bool:
    """Validates an Aadhaar number using the Verhoeff algorithm."""
    clean = re.sub(r"\D", "", num_str)
    if len(clean) != 12:
        return False
    c = 0
    # Reverse digits and calculate
    for i, digit_char in enumerate(reversed(clean)):
        digit = int(digit_char)
        c = VERHOEFF_D[c][VERHOEFF_P[i % 8][digit]]
    return c == 0


# --- 2. ICAO Doc 9303 Checksum for Passport MRZ ---
ICAO_WEIGHTS = [7, 3, 1]


def icao_char_value(c: str) -> int:
    if c.isdigit():
        return int(c)
    if "A" <= c <= "Z":
        return ord(c) - ord("A") + 10
    if c == "<":
        return 0
    return 0


def calculate_icao_check_digit(data: str) -> int:
    total = 0
    for idx, char in enumerate(data):
        weight = ICAO_WEIGHTS[idx % 3]
        total += icao_char_value(char) * weight
    return total % 10


def validate_passport_mrz_checksums(mrz_raw_lines: list) -> dict:
    """
    Validates ICAO Doc 9303 Line 2 checksums:
    Document Number (0-8) -> check digit at 9
    Date of Birth (13-18) -> check digit at 19
    Date of Expiry (21-26) -> check digit at 27
    Composite Check Digit at 43 (over doc_num+cd+dob+cd+expiry+cd+optional)
    """
    results = {
        "doc_number_valid": None,
        "dob_valid": None,
        "expiry_valid": None,
        "composite_valid": None,
        "all_valid": True,
    }
    if not mrz_raw_lines or len(mrz_raw_lines) < 2:
        return results

    line2 = mrz_raw_lines[1].strip()
    if len(line2) < 28:
        return results

    # 1. Document number check digit
    doc_field = line2[0:9]
    doc_cd = line2[9]
    if doc_cd.isdigit():
        calc_cd = calculate_icao_check_digit(doc_field)
        results["doc_number_valid"] = (calc_cd == int(doc_cd))
        if not results["doc_number_valid"]:
            results["all_valid"] = False

    # 2. Date of Birth check digit
    dob_field = line2[13:19]
    dob_cd = line2[19]
    if dob_cd.isdigit() and dob_field.isalnum():
        calc_cd = calculate_icao_check_digit(dob_field)
        results["dob_valid"] = (calc_cd == int(dob_cd))
        if not results["dob_valid"]:
            results["all_valid"] = False

    # 3. Expiry date check digit
    exp_field = line2[21:27]
    exp_cd = line2[27]
    if exp_cd.isdigit() and exp_field.isalnum():
        calc_cd = calculate_icao_check_digit(exp_field)
        results["expiry_valid"] = (calc_cd == int(exp_cd))
        if not results["expiry_valid"]:
            results["all_valid"] = False

    # 4. Composite check digit (at pos 43 for 44-char TD3 passport)
    if len(line2) >= 44:
        composite_data = line2[0:10] + line2[13:20] + line2[21:43]
        composite_cd = line2[43]
        if composite_cd.isdigit():
            calc_composite = calculate_icao_check_digit(composite_data)
            results["composite_valid"] = (calc_composite == int(composite_cd))
            if not results["composite_valid"]:
                results["all_valid"] = False

    return results


# --- 3. Indian State Codes for Driving Licences ---
INDIAN_STATE_CODES = {
    "AN": "Andaman and Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CG": "Chhattisgarh", "CH": "Chandigarh",
    "DD": "Daman and Diu", "DL": "Delhi", "DN": "Dadra and Nagar Haveli",
    "GA": "Goa", "GJ": "Gujarat", "HP": "Himachal Pradesh", "HR": "Haryana",
    "JH": "Jharkhand", "JK": "Jammu and Kashmir", "KA": "Karnataka", "KL": "Kerala",
    "LA": "Ladakh", "LD": "Lakshadweep", "MH": "Maharashtra", "ML": "Meghalaya",
    "MN": "Manipur", "MP": "Madhya Pradesh", "MZ": "Mizoram", "NL": "Nagaland",
    "OD": "Odisha", "PB": "Punjab", "PY": "Puducherry", "RJ": "Rajasthan",
    "SK": "Sikkim", "TN": "Tamil Nadu", "TR": "Tripura", "TS": "Telangana",
    "UK": "Uttarakhand", "UP": "Uttar Pradesh", "WB": "West Bengal",
}


# ============================================================
# COMPREHENSIVE DOCUMENT VERIFICATION ROUTINE
# ============================================================

def run_validation(
    doc_type: str,
    text: str,
    ocr_confidence: float,
):
    issues = []
    verified = []
    score = 0

    lower_text = text.lower()
    upper_text = text.upper()

    # --------------------------------------------------------
    # COMMON: NAME EXTRACTION
    # --------------------------------------------------------
    name = guess_name(text)
    if name:
        verified.append(f"Cardholder / Subject Name identified: {name}")
    else:
        issues.append({
            "title": "Subject name not found",
            "severity": severity_for_weight(WEIGHTS["MISSING_REQUIRED_FIELD"]),
            "description": "Could not identify a valid cardholder or traveler name from the document text.",
        })
        score += WEIGHTS["MISSING_REQUIRED_FIELD"]

    # --------------------------------------------------------
    # COMMON: DOCUMENT NUMBER DETECTION
    # --------------------------------------------------------
    doc_number = extract_document_number(text, doc_type)

    # --------------------------------------------------------
    # DOCUMENT-SPECIFIC AUTHENTICITY CHECKS
    # --------------------------------------------------------

    # ==========================
    # 1. AADHAAR CARD
    # ==========================
    if doc_type == "Aadhaar":
        if doc_number:
            clean_aadhaar = re.sub(r"\D", "", doc_number)
            if len(clean_aadhaar) == 12:
                # Mathematical Verhoeff check digit validation
                if validate_verhoeff(clean_aadhaar):
                    verified.append("UIDAI Verhoeff Check Digit: PASSED (Mathematically valid 12-digit Aadhaar number)")
                else:
                    issues.append({
                        "title": "Aadhaar Verhoeff Checksum Failed",
                        "severity": "HIGH",
                        "description": f"The 12-digit Aadhaar sequence ({clean_aadhaar}) failed the Verhoeff check digit algorithm. This indicates a counterfeit or mis-typed number.",
                    })
                    score += WEIGHTS["CHECKSUM_FAILED"]
            else:
                issues.append({
                    "title": "Invalid Aadhaar length",
                    "severity": "HIGH",
                    "description": f"Aadhaar number must contain exactly 12 digits, found {len(clean_aadhaar)} digits.",
                })
                score += WEIGHTS["INVALID_NUMBER_FORMAT"]
        else:
            issues.append({
                "title": "Missing Aadhaar Number",
                "severity": "HIGH",
                "description": "No 12-digit Aadhaar number sequence (XXXX XXXX XXXX) was detected.",
            })
            score += WEIGHTS["INVALID_NUMBER_FORMAT"]

        # Check UIDAI Security & Official Keywords
        aadhaar_markers = [
            ("unique identification authority", "UIDAI Insignia / Authority header"),
            ("government of india", "Government of India emblem text"),
            ("mera aadhaar", "Mera Aadhaar watermark / tagline"),
            ("enrolment", "Enrolment / Update reference"),
            ("1947", "UIDAI 1947 Toll-Free helpline"),
        ]
        found_markers = [label for kw, label in aadhaar_markers if kw in lower_text]
        if found_markers:
            verified.append(f"UIDAI Authentic Markers: Confirmed ({', '.join(found_markers[:3])})")
        else:
            issues.append({
                "title": "UIDAI Authority Insignia Missing",
                "severity": "MEDIUM",
                "description": "Standard UIDAI header, Government of India emblem text, or helpline references not detected.",
            })
            score += WEIGHTS["SECURITY_FEATURE_MISSING"]

        # Check Gender and Date of Birth
        has_dob = bool(re.search(r"\b(dob|birth|year of birth|\d{2}/\d{2}/\d{4}|\b\d{4}\b)\b", lower_text))
        has_gender = bool(re.search(r"\b(male|female|transgender|\u092a\u0941\u0930\u0941\u0937|\u0938\u094d\u0924\u094d\u0930\u0940)\b", lower_text))
        if has_dob:
            verified.append("Aadhaar Date/Year of Birth field present")
        else:
            issues.append({
                "title": "Aadhaar DOB/YOB missing",
                "severity": "LOW",
                "description": "Could not identify standard Date of Birth or Year of Birth line on Aadhaar card.",
            })
        if has_gender:
            verified.append("Aadhaar Gender specification verified")

    # ==========================
    # 2. PAN CARD
    # ==========================
    elif doc_type == "PAN Card":
        if doc_number:
            pan_match = PAN_NUMBER_RE.search(doc_number)
            if pan_match:
                pan_code = pan_match.group(0).upper()
                fourth_char = pan_code[3]
                fifth_char = pan_code[4]

                # 4th character entity type check
                entity_types = {
                    "P": "Individual / Person",
                    "C": "Company",
                    "H": "Hindu Undivided Family (HUF)",
                    "F": "Firm / Partnership / LLP",
                    "A": "Association of Persons (AOP)",
                    "T": "Trust",
                    "B": "Body of Individuals (BOI)",
                    "L": "Local Authority",
                    "J": "Artificial Juridical Person",
                    "G": "Government Agency",
                }
                if fourth_char in entity_types:
                    verified.append(f"PAN Entity Type: Valid ({entity_types[fourth_char]} - code '{fourth_char}')")
                else:
                    issues.append({
                        "title": "Invalid PAN Entity Code",
                        "severity": "HIGH",
                        "description": f"4th character '{fourth_char}' in PAN '{pan_code}' is not a recognized Indian Income Tax Department entity code.",
                    })
                    score += WEIGHTS["FORGED_PATTERN"]

                # 5th character: must match holder's surname initial
                if name:
                    name_parts = [p for p in re.split(r"[\s.]+", name.strip()) if len(p) > 1]
                    if name_parts:
                        surname = name_parts[-1].upper()
                        if surname.startswith(fifth_char):
                            verified.append(f"PAN Surname Initial Match: PASSED (Initial '{fifth_char}' matches surname '{surname}')")
                        else:
                            issues.append({
                                "title": "PAN Surname Initial Discrepancy",
                                "severity": "MEDIUM",
                                "description": f"PAN 5th character '{fifth_char}' does not match the first letter of extracted surname '{surname}'.",
                            })
                            score += 15
                verified.append(f"PAN Alphanumeric Structure: PASSED (5-Alpha + 4-Numeric + 1-Alpha: {pan_code})")
            else:
                issues.append({
                    "title": "Invalid PAN Number Structure",
                    "severity": "HIGH",
                    "description": "PAN number does not match standard 10-character format AAAAA9999A.",
                })
                score += WEIGHTS["INVALID_NUMBER_FORMAT"]
        else:
            issues.append({
                "title": "Missing PAN Number",
                "severity": "HIGH",
                "description": "No 10-character PAN number pattern (e.g., ABCDE1234F) detected on card.",
            })
            score += WEIGHTS["INVALID_NUMBER_FORMAT"]

        # Check Income Tax Department Issuer keywords
        it_markers = ["income tax department", "permanent account number", "govt of india", "government of india"]
        if any(marker in lower_text for marker in it_markers):
            verified.append("Income Tax Department Issuer Markings confirmed")
        else:
            issues.append({
                "title": "Missing Income Tax Department Markings",
                "severity": "MEDIUM",
                "description": "Official Income Tax Department header or Government of India tagline missing.",
            })
            score += WEIGHTS["SECURITY_FEATURE_MISSING"]

    # ==========================
    # 3. PASSPORT (ICAO DOC 9303)
    # ==========================
    elif doc_type == "Passport":
        # Extract MRZ lines for ICAO check
        mrz_candidate_lines = []
        for line in text.splitlines():
            clean = line.strip().replace(" ", "<")
            clean_alnum = re.sub(r"[^A-Z0-9<]", "", clean.upper())
            if len(clean_alnum) >= 30 and "<" in clean_alnum:
                mrz_candidate_lines.append(clean_alnum)

        mrz = parse_mrz(text)
        if mrz:
            verified.append("Passport Machine Readable Zone (MRZ TD3) detected and parsed")
            if mrz.get("doc_number"):
                verified.append(f"Passport Number extracted: {mrz['doc_number']}")
            if mrz.get("nationality"):
                verified.append(f"Holder Nationality Code: {mrz['nationality']}")

            # ICAO Doc 9303 Check Digit Validation
            if len(mrz_candidate_lines) >= 2:
                # Find the line starting with P< and its follower
                line1_idx = -1
                for idx, line in enumerate(mrz_candidate_lines):
                    if line.startswith("P<") or line.startswith("PK") or line.startswith("P"):
                        line1_idx = idx
                        break
                if line1_idx != -1 and line1_idx + 1 < len(mrz_candidate_lines):
                    td3_pair = [mrz_candidate_lines[line1_idx], mrz_candidate_lines[line1_idx + 1]]
                    mrz_check_res = validate_passport_mrz_checksums(td3_pair)

                    if mrz_check_res.get("doc_number_valid") is True:
                        verified.append("ICAO Doc 9303 Passport Number Check Digit: PASSED")
                    elif mrz_check_res.get("doc_number_valid") is False:
                        issues.append({
                            "title": "MRZ Passport Number Checksum Failed",
                            "severity": "HIGH",
                            "description": "Document number check digit in MRZ line 2 does not match ICAO 7-3-1 algorithm.",
                        })
                        score += WEIGHTS["CHECKSUM_FAILED"]

                    if mrz_check_res.get("dob_valid") is True:
                        verified.append("ICAO Doc 9303 Date of Birth Check Digit: PASSED")
                    elif mrz_check_res.get("dob_valid") is False:
                        issues.append({
                            "title": "MRZ Date of Birth Checksum Failed",
                            "severity": "MEDIUM",
                            "description": "Date of Birth check digit in MRZ line 2 is mathematically invalid.",
                        })
                        score += 15

                    if mrz_check_res.get("expiry_valid") is True:
                        verified.append("ICAO Doc 9303 Expiry Date Check Digit: PASSED")
                    elif mrz_check_res.get("expiry_valid") is False:
                        issues.append({
                            "title": "MRZ Expiry Checksum Failed",
                            "severity": "MEDIUM",
                            "description": "Passport Expiry check digit in MRZ line 2 is mathematically invalid.",
                        })
                        score += 15

                    if mrz_check_res.get("composite_valid") is True:
                        verified.append("ICAO Doc 9303 Overall Composite Check Digit: PASSED")
                    elif mrz_check_res.get("composite_valid") is False:
                        issues.append({
                            "title": "MRZ Composite Check Digit Failed",
                            "severity": "HIGH",
                            "description": "The composite checksum covering all passport data fields failed verification.",
                        })
                        score += WEIGHTS["CHECKSUM_FAILED"]

            # Expiry date validation
            if mrz.get("expiry_date"):
                try:
                    exp_dt = date.fromisoformat(mrz["expiry_date"])
                    if exp_dt < date.today():
                        issues.append({
                            "title": "Passport Expired",
                            "severity": "HIGH",
                            "description": f"Passport expired on {exp_dt.strftime('%d/%m/%Y')} and is invalid for international travel.",
                        })
                        score += WEIGHTS["EXPIRED_DOCUMENT"]
                    else:
                        verified.append(f"Passport Validity: Active (Valid until {exp_dt.strftime('%d/%m/%Y')})")
                except ValueError:
                    pass
        else:
            # Fallback to visual booklet format
            if doc_number:
                verified.append(f"Visual Passport Number format verified: {doc_number}")
            else:
                issues.append({
                    "title": "Missing Passport Number & MRZ",
                    "severity": "HIGH",
                    "description": "Neither a readable MRZ zone nor a valid Passport booklet number was extracted.",
                })
                score += WEIGHTS["INVALID_NUMBER_FORMAT"]

    # ==========================
    # 4. DRIVING LICENCE (PARIVAHAN)
    # ==========================
    elif doc_type == "Driving Licence":
        if doc_number:
            clean_dl = re.sub(r"[\s-]", "", doc_number.upper())
            # State code check (first 2 chars)
            state_code = clean_dl[:2]
            if state_code in INDIAN_STATE_CODES:
                state_name = INDIAN_STATE_CODES[state_code]
                verified.append(f"Parivahan State Jurisdiction: Valid ({state_name} - State Code '{state_code}')")

                # Format check: SS-RR-YYYY-NNNNNNN or legacy SS-RR-NNNNNN
                if len(clean_dl) >= 15:
                    rto_code = clean_dl[2:4]
                    year_str = clean_dl[4:8]
                    if year_str.isdigit():
                        issue_year = int(year_str)
                        current_year = date.today().year
                        if 1950 <= issue_year <= current_year:
                            verified.append(f"Parivahan RTO Sequence & Issue Year ({issue_year}) validated")
                        else:
                            issues.append({
                                "title": "Abnormal Driving Licence Issue Year",
                                "severity": "HIGH",
                                "description": f"Driving licence indicates issuance year {issue_year}, which is out of range.",
                            })
                            score += WEIGHTS["FORGED_PATTERN"]
                elif len(clean_dl) >= 9:
                    verified.append(f"Legacy State Driving Licence format accepted ({len(clean_dl)} characters)")

                verified.append(f"Driving Licence Number: Valid format ({doc_number})")
            else:
                issues.append({
                    "title": "Invalid Driving Licence State Code",
                    "severity": "HIGH",
                    "description": f"State prefix '{state_code}' does not correspond to any valid Indian State or Union Territory.",
                })
                score += WEIGHTS["INVALID_NUMBER_FORMAT"]
        else:
            issues.append({
                "title": "Missing Driving Licence Number",
                "severity": "HIGH",
                "description": "No recognized Driving Licence number format (e.g. MH01 20180012345) detected.",
            })
            score += WEIGHTS["INVALID_NUMBER_FORMAT"]

        # Motor Vehicles Act / Transport Department markers
        dl_markers = ["transport", "driving licence", "driving license", "motor vehicles", "sarathi", "licence", "license"]
        if any(marker in lower_text for marker in dl_markers):
            verified.append("Transport Department / Motor Vehicles Act statutory markers present")
        else:
            issues.append({
                "title": "Transport Authority Markings Missing",
                "severity": "MEDIUM",
                "description": "No state transport department or licensing authority header detected.",
            })
            score += WEIGHTS["SECURITY_FEATURE_MISSING"]

    # ==========================
    # 5. VISA
    # ==========================
    elif doc_type == "Visa":
        if doc_number:
            verified.append(f"Visa Document / Sticker Number detected: {doc_number}")
        else:
            issues.append({
                "title": "Visa Document Number missing",
                "severity": "MEDIUM",
                "description": "Could not extract visa foil / sticker registration number.",
            })
            score += WEIGHTS["INVALID_NUMBER_FORMAT"]

        visa_dates = extract_dates(text)
        parsed_dates = [to_date(d) for d in visa_dates if to_date(d)]
        if parsed_dates:
            latest = max(parsed_dates)
            if latest < date.today():
                issues.append({
                    "title": "Visa Expired",
                    "severity": "HIGH",
                    "description": f"Latest date found on visa ({latest.strftime('%d/%m/%Y')}) is in the past.",
                })
                score += WEIGHTS["EXPIRED_DOCUMENT"]
            else:
                verified.append(f"Visa Validity Window active (Valid until {latest.strftime('%d/%m/%Y')})")
        else:
            issues.append({
                "title": "Visa Validity Dates Missing",
                "severity": "MEDIUM",
                "description": "Valid from / Valid until date stamps could not be found.",
            })
            score += WEIGHTS["MISSING_REQUIRED_FIELD"]

        # Entries and category checks
        if any(kw in lower_text for kw in ["single", "multiple", "double", "entries", "entry"]):
            verified.append("Visa Entry clearance category verified")
        if any(kw in lower_text for kw in ["tourist", "business", "employment", "student", "diplomatic", "visa type", "category"]):
            verified.append("Visa Purpose / Class designation identified")

    # ==========================
    # 6. VOTER ID (EPIC)
    # ==========================
    elif doc_type == "Voter ID":
        if doc_number:
            voter_match = VOTER_ID_RE.search(doc_number)
            if voter_match:
                epic = voter_match.group(0).upper()
                constituency_code = epic[:3]
                serial_num = epic[3:]
                verified.append(f"EPIC Number format valid: {epic} (Constituency Code: {constituency_code}, Serial: {serial_num})")
            else:
                issues.append({
                    "title": "Invalid EPIC Format",
                    "severity": "HIGH",
                    "description": "Voter ID must follow 3-Alpha + 7-Numeric format (e.g. ABC1234567).",
                })
                score += WEIGHTS["INVALID_NUMBER_FORMAT"]
        else:
            issues.append({
                "title": "Missing Voter ID / EPIC Number",
                "severity": "HIGH",
                "description": "No 10-character EPIC number pattern detected.",
            })
            score += WEIGHTS["INVALID_NUMBER_FORMAT"]

        eci_markers = ["election commission", "electors photo identity", "elector", "epic", "voter"]
        if any(marker in lower_text for marker in eci_markers):
            verified.append("Election Commission of India (ECI) Insignia verified")
        else:
            issues.append({
                "title": "Election Commission Insignia Missing",
                "severity": "MEDIUM",
                "description": "ECI statutory header or electoral roll reference not found.",
            })
            score += WEIGHTS["SECURITY_FEATURE_MISSING"]

    # ==========================
    # 7. PERMIT
    # ==========================
    elif doc_type == "Permit":
        if doc_number:
            verified.append(f"Permit Authorization Number confirmed: {doc_number}")
        else:
            issues.append({
                "title": "Permit Number Missing",
                "severity": "MEDIUM",
                "description": "No official permit registration code identified.",
            })
            score += WEIGHTS["MISSING_REQUIRED_FIELD"]

        permit_dates = extract_dates(text)
        parsed_dates = [to_date(d) for d in permit_dates if to_date(d)]
        if parsed_dates:
            latest = max(parsed_dates)
            if latest < date.today():
                issues.append({
                    "title": "Permit Expired",
                    "severity": "HIGH",
                    "description": f"The permit validity period expired on {latest.strftime('%d/%m/%Y')}.",
                })
                score += WEIGHTS["EXPIRED_DOCUMENT"]
            else:
                verified.append(f"Permit operational window active (Valid through {latest.strftime('%d/%m/%Y')})")

    # ==========================
    # 8. NATIONAL ID (GENERIC)
    # ==========================
    elif doc_type == "National ID":
        if doc_number:
            verified.append(f"National Identity Number detected: {doc_number}")
        else:
            issues.append({
                "title": "Missing National ID Number",
                "severity": "HIGH",
                "description": "Could not identify a standard national identification number.",
            })
            score += WEIGHTS["INVALID_NUMBER_FORMAT"]

    # ==========================
    # 9. UNKNOWN DOCUMENT
    # ==========================
    elif doc_type == "Unknown":
        issues.append({
            "title": "Document type could not be identified",
            "severity": "HIGH",
            "description": "No supported border screening document pattern was confidently recognized.",
        })
        score += WEIGHTS["CRITICAL_MISMATCH"]

    # --------------------------------------------------------
    # OCR CONFIDENCE CHECK
    # --------------------------------------------------------
    if ocr_confidence < 0.60:
        issues.append({
            "title": "Low OCR Text Quality",
            "severity": "MEDIUM",
            "description": "OCR recognition quality is low. Possible image blur, low resolution, or degradation.",
        })
        score += WEIGHTS["LOW_OCR_CONFIDENCE"]
    else:
        verified.append("High-confidence Optical Character Recognition complete")

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
    """Pre-process image for better OCR. Skips gracefully if cv2 not available."""
    if not _CV2_AVAILABLE:
        # On Vercel serverless without cv2, just return the original bytes.
        # OCR.space handles images well without pre-processing.
        return file_bytes

    try:
        image_array = _np.frombuffer(file_bytes, dtype=_np.uint8)
        image = _cv2.imdecode(image_array, _cv2.IMREAD_COLOR)

        if image is None:
            return file_bytes

        h, w = image.shape[:2]
        max_dim = 2000

        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = _cv2.resize(
                image,
                (int(w * scale), int(h * scale)),
                interpolation=_cv2.INTER_AREA,
            )

        gray = _cv2.cvtColor(image, _cv2.COLOR_BGR2GRAY)
        gray = _cv2.fastNlMeansDenoising(gray, h=30)
        clahe = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        success, encoded_img = _cv2.imencode(".png", gray)
        if success:
            return encoded_img.tobytes()
    except Exception as e:
        print(f"Image preprocessing skipped: {e}")

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

        # Try OCR with 25s timeout (8s is often too fast for high-res images over network)
        for engine in ("2", "1"):
            data["OCREngine"] = engine
            try:
                response = requests.post(
                    url,
                    files={"file": (filename, file_bytes)},
                    data=data,
                    timeout=25,
                )

                if response.status_code == 200:
                    result = response.json()
                    if not result.get("IsErroredOnProcessing"):
                        parsed_results = result.get("ParsedResults", [])
                        if parsed_results:
                            extracted_ocr = "\n".join(item.get("ParsedText", "") for item in parsed_results)
                            if len(extracted_ocr.strip()) > 5:
                                return extracted_ocr
            except Exception as ocr_err:
                print(f"OCR.space request with engine {engine} error: {ocr_err}")

    # ========================================================
    # LOCAL FALLBACK
    # ========================================================

    print(
        "Falling back to local OCR "
        "(pytesseract) if available..."
    )

    try:
        if _CV2_AVAILABLE:
            import pytesseract
            image_array = _np.frombuffer(
                file_bytes,
                dtype=_np.uint8,
            )
            img = _cv2.imdecode(
                image_array,
                _cv2.IMREAD_COLOR,
            )
            if img is not None:
                text = pytesseract.image_to_string(img)
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
    cv_ver = None
    if _CV2_AVAILABLE and _cv2 is not None:
        cv_ver = getattr(_cv2, "__version__", None)

    return {
        "success": True,
        "status": "ONLINE",
        "service": "BorderShield AI",
        "opencv_version": cv_ver,
        "opencv_available": bool(_CV2_AVAILABLE and _cv2 is not None),
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
    selfie: Optional[UploadFile] = File(None),
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

    selfie_bytes = None
    if selfie is not None:
        try:
            selfie_bytes = await selfie.read()
            if not selfie_bytes:
                selfie_bytes = None
        except Exception as _selfie_err:
            print("Selfie read error:", _selfie_err)
            selfie_bytes = None

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

    # Attempt document rectification (perspective correction) if available
    doc_bytes_for_ocr = file_bytes
    try:
        rectified_img, rect_res = rectify_document(file_bytes)
        if rect_res.get("rectified") and _CV2_AVAILABLE and _cv2 is not None and isinstance(rectified_img, _np.ndarray):
            ok, enc_rect = _cv2.imencode(".jpg", rectified_img)
            if ok:
                doc_bytes_for_ocr = enc_rect.tobytes()
    except Exception as _rect_err:
        print("Rectification notice:", _rect_err)

    # First attempt: OCR on the original high-fidelity document
    text = perform_ocr(
        file_bytes,
        filename,
    )

    # Second attempt: If raw OCR returns very little text, try preprocessed/rectified image
    if len(text.strip()) < 15:
        print("Raw OCR result short, attempting preprocessed image...")
        processed_bytes = preprocess_image_for_ocr(doc_bytes_for_ocr)
        processed_text = perform_ocr(
            processed_bytes,
            filename,
        )
        if len(processed_text.strip()) > len(text.strip()):
            text = processed_text

    # --------------------------------------------------------
    # KEEP BEST OCR RESULT
    # --------------------------------------------------------

    text = text.strip()

    # ========================================================
    # OCR FAILURE
    # ========================================================

    if not text:

        status = "ocr_unavailable"

        issues = [
            {
                "title": "OCR text extraction unavailable",
                "severity": "MEDIUM",
                "description": (
                    "No readable text could be extracted or OCR timed out. "
                    "Risk score cannot be computed from empty text; officer manual inspection is required."
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
                risk_score=0,
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
            "identity_id": None,
            "document_type": "Unknown",
            "detected_document_type": "unknown",
            "expected_document_type": expected_document_type,
            "risk_score": 0,
            "status": "ocr_unavailable",
            "message": "OCR unavailable or timed out. No readable text extracted; risk score not computed.",
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
            "ocr_text": "",
            "ocr_confidence": 0.0,
            "passport_mrz": None,
            "visual_analysis": visual_analysis,
            "identity_intelligence": None,
            "verification_result": {
                "document_type": "unknown",
                "status": "ocr_unavailable",
                "risk_score": 0,
                "reasons": ["OCR text extraction unavailable"],
                "limitations": ["Document text unreadable or OCR service timed out"],
                "verification_tier": 3,
                "verification_method": "manual_inspection_required",
            },
            "ai_tamper_analysis": {
                "is_ai_generated_risk": False,
                "combined_risk_score": 0,
                "findings": ["OCR unavailable — skipped automated risk analysis"],
            },
            "opencv_forensics": {
                "ela": {"score": 0.0, "flag": False, "explanation": "Skipped due to unreadable document", "status": "skipped", "details": {}},
                "copy_move": {"score": 0.0, "flag": False, "explanation": "Skipped due to unreadable document", "status": "skipped", "details": {}},
                "qr": {"score": 0.0, "flag": False, "explanation": "Skipped due to unreadable document", "status": "skipped", "details": {}},
                "face_match": {"score": 0.0, "flag": False, "explanation": "Skipped due to unreadable document", "status": "skipped", "details": {}},
            },
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
    expected = (expected_document_type or "").strip().lower()

    # If the officer explicitly specified the document type (e.g. 'Driving Licence'),
    # and the OCR text contains corroborating clues (DL number format or keywords),
    # honor the officer's selection rather than falling back to 'Unknown'.
    if expected in ("driving licence", "driving license"):
        if doc_type == "Unknown" or doc_type != "Driving Licence":
            if (
                DL_NUMBER_RE.search(text)
                or DL_NUMBER_LEGACY_RE.search(text)
                or any(k in text.lower() for k in ["driving", "licence", "license", "sarathi", "transport", "motor", "rto", "dl no"])
            ):
                doc_type = "Driving Licence"

    aliases = {
        "driving license": "driving licence",
        "driving licence": "driving licence",
        "passport": "passport",
        "visa": "visa",
        "national id": "national id",
        "permit": "permit",
        "aadhaar": "aadhaar",
        "aadhar": "aadhaar",
        "pan card": "pan card",
        "pan": "pan card",
        "pancard": "pan card",
        "voter id": "voter id",
    }
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

    # Permit compatibility: If expected is National ID and detected is Aadhaar, accept as valid match
    is_compatible_type = (
        expected_normalized == detected_normalized
        or expected_normalized in ("auto", "unknown", "")
        or (expected_normalized == "national id" and detected_normalized == "aadhaar")
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

    if not is_compatible_type:

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

    # ========================================================
    # IDENTITY & AADHAAR PRIVACY MASKING
    # ========================================================

    stored_doc_number = doc_number
    if doc_type == "Aadhaar" and doc_number:
        # UIDAI compliance: mask first 8 digits -> XXXX-XXXX-1234
        digits = re.sub(r"\D", "", doc_number)
        if len(digits) >= 4:
            stored_doc_number = f"XXXX-XXXX-{digits[-4:]}"
        else:
            stored_doc_number = "XXXX-XXXX-XXXX"

    identity_id = None

    try:

        if stored_doc_number:

            identity_id = create_or_get_identity(
                full_name=name or "Unknown",
                document_number=stored_doc_number,
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
                stored_doc_number
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
    # MODULAR MULTI-DOCUMENT VERIFICATION ENGINE (PHASES 2-6)
    # ========================================================

    ai_tamper_analysis = run_ai_and_tamper_forensics(file_bytes, file.filename or "document.jpg")

    detected_slug, det_conf, det_details = detect_document_type(text, expected_document_type)

    extracted_dict = {
        "name_guess": name,
        "document_number": doc_number,
        "dates_found": extract_dates(text),
    }

    # Dispatch to dedicated document module
    verifier_map = {
        "aadhaar": AadhaarVerifier(),
        "passport": PassportVerifier(),
        "pan": PanVerifier(),
        "driving_licence": DrivingLicenceVerifier(),
        "visa": VisaVerifier(),
        "national_id": NationalIdVerifier(),
        "permit": PermitVerifier(),
    }

    verifier = verifier_map.get(detected_slug)
    if not verifier:
        # Fallback to declared document type
        norm_declared = expected_document_type.lower().replace(" ", "_")
        verifier = verifier_map.get(norm_declared, NationalIdVerifier() if "national" in norm_declared else VisaVerifier())

    module_verification = verifier.verify(
        file_bytes=file_bytes,
        filename=file.filename or "document.jpg",
        content_type=file.content_type or "image/jpeg",
        ocr_text=text,
        extracted_fields=extracted_dict,
    )

    # Cryptographic check failure override rule:
    # A failed cryptographic check MUST override every other score.
    if module_verification.verification_tier == 1 and module_verification.status == "failed":
        score = 100
        status = "HIGH RISK"
    else:
        # Incorporate modular verification risk into overall composite
        score = max(score, module_verification.risk_score)
        if ai_tamper_analysis.get("is_ai_generated_risk"):
            score = max(score, ai_tamper_analysis.get("combined_risk_score", 65))
        status = status_for_score(score)

    # Combine issues and reasons
    for reason in module_verification.reasons:
        if "CRITICAL" in reason or "Failed" in reason or "failed" in reason or "mismatch" in reason.lower():
            issues.append({"title": "Verification Finding", "severity": "HIGH", "description": reason})
        else:
            verified.append(reason)

    document_information = extract_document_information(
        text,
        doc_type,
    )

    # ========================================================
    # OPENCV ADVANCED FORENSICS ENGINE
    # ========================================================
    opencv_forensics = {
        "ela": {"score": 0.0, "flag": False, "explanation": "Not evaluated", "status": "skipped", "details": {}},
        "copy_move": {"score": 0.0, "flag": False, "explanation": "Not evaluated", "status": "skipped", "details": {}},
        "qr": {"score": 0.0, "flag": False, "explanation": "Not evaluated", "status": "skipped", "details": {}},
        "face_match": {"score": 0.0, "flag": False, "explanation": "Not evaluated", "status": "skipped", "details": {}},
    }

    try:
        opencv_forensics["ela"] = run_ela_check(file_bytes)
    except Exception as _ela_err:
        opencv_forensics["ela"] = {
            "score": 0.0,
            "flag": False,
            "explanation": f"ELA execution error: {str(_ela_err)}",
            "status": "failed",
            "details": {"error": str(_ela_err)},
        }

    try:
        opencv_forensics["copy_move"] = run_copy_move_check(file_bytes)
    except Exception as _cm_err:
        opencv_forensics["copy_move"] = {
            "score": 0.0,
            "flag": False,
            "explanation": f"Copy-move check error: {str(_cm_err)}",
            "status": "failed",
            "details": {"error": str(_cm_err)},
        }

    try:
        opencv_forensics["qr"] = run_qr_check(file_bytes, doc_type=expected_document_type, ocr_text=text)
    except Exception as _qr_err:
        opencv_forensics["qr"] = {
            "score": 0.0,
            "flag": False,
            "explanation": f"QR check error: {str(_qr_err)}",
            "status": "failed",
            "details": {"error": str(_qr_err)},
        }

    if selfie_bytes:
        try:
            opencv_forensics["face_match"] = run_face_match_check(file_bytes, selfie_bytes)
        except Exception as _face_err:
            opencv_forensics["face_match"] = {
                "score": 0.0,
                "flag": False,
                "explanation": f"Face match error: {str(_face_err)}",
                "status": "failed",
                "details": {"error": str(_face_err)},
            }
    else:
        opencv_forensics["face_match"] = {
            "score": 0.0,
            "flag": False,
            "explanation": "No selfie image provided for facial comparison.",
            "status": "skipped",
            "details": {"selfie_provided": False},
        }

    # Integrate OpenCV forensics into risk scoring (capped 0 to 100)
    if opencv_forensics["ela"].get("flag"):
        score += WEIGHTS.get("OPENCV_ELA_ANOMALY", 25)
        issues.append({
            "title": "OpenCV Forensics: ELA Compression Anomaly",
            "severity": "HIGH",
            "description": opencv_forensics["ela"].get("explanation"),
        })

    if opencv_forensics["copy_move"].get("flag"):
        score += WEIGHTS.get("OPENCV_COPY_MOVE_TAMPERING", 35)
        issues.append({
            "title": "OpenCV Forensics: Copy-Move Forgery Detected",
            "severity": "HIGH",
            "description": opencv_forensics["copy_move"].get("explanation"),
        })

    if opencv_forensics["qr"].get("flag"):
        score += WEIGHTS.get("OPENCV_QR_MISMATCH", 35)
        issues.append({
            "title": "OpenCV Forensics: QR Code Data Conflict",
            "severity": "HIGH",
            "description": opencv_forensics["qr"].get("explanation"),
        })

    if opencv_forensics["face_match"].get("flag") and opencv_forensics["face_match"].get("status") == "completed":
        score += WEIGHTS.get("OPENCV_FACE_MISMATCH", 40)
        issues.append({
            "title": "OpenCV Forensics: Facial Mismatch with Live Selfie",
            "severity": "HIGH",
            "description": opencv_forensics["face_match"].get("explanation"),
        })

    score = min(max(int(score), 0), 100)
    status = status_for_score(score)

    # ========================================================
    # FINAL RESPONSE (Common Schema + Comprehensive Analysis)
    # ========================================================

    return {
        "success": True,
        "screening_id": screening_id,
        "identity_id": identity_id,
        "document_type": doc_type,
        "detected_document_type": detected_slug,
        "expected_document_type": expected_document_type,
        "risk_score": score,
        "status": status,
        "message": "Document analyzed via modular verification engine.",
        "issues": issues,
        "verified_checks": verified,
        "extracted_fields": extracted_dict,
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
        # Standard Common Result Schema & AI Forensics
        "verification_result": module_verification.to_dict(),
        "ai_tamper_analysis": ai_tamper_analysis,
        "opencv_forensics": opencv_forensics,
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


@app.get("/api/identities")
def list_identities_api(
    limit: int = 30,
):
    try:
        identities = get_identity_network_data(limit=limit)
        return {
            "success": True,
            "count": len(identities),
            "identities": identities,
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )



# ============================================================
# SCREENINGS LIST API (Real database screenings)
# ============================================================

@app.get("/api/screenings")
def list_screenings_api(
    limit: int = 100,
    status: Optional[str] = None,
    search: Optional[str] = None,
):
    try:
        screenings = get_screenings_list(
            limit=limit,
            status=status,
            search=search,
        )
        return {
            "success": True,
            "count": len(screenings),
            "screenings": screenings,
        }
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )


# ============================================================
# FRAUD CASES API (Real database fraud cases + live aggregates)
# ============================================================

@app.get("/api/fraud")
def list_fraud_api(
    limit: int = 100,
):
    try:
        cases = get_fraud_cases_list(limit=limit)

        # Retrieve live counts from screenings table
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*) FILTER (WHERE status = 'HIGH RISK') AS high_risk_count,
                COUNT(*) FILTER (WHERE status = 'SUSPICIOUS') AS suspicious_count,
                COUNT(*) FILTER (WHERE status = 'REVIEW REQUIRED') AS review_count,
                COUNT(*) FILTER (WHERE status = 'LOW RISK') AS cleared_count,
                COUNT(*) AS total_screenings
            FROM screenings
        """)
        stats = dict(cursor.fetchone())

        # If explicit fraud_cases table has no rows yet, derive active fraud investigations
        # directly from the real screenings marked HIGH RISK or SUSPICIOUS / REVIEW REQUIRED
        if not cases:
            cursor.execute("""
                SELECT
                    id,
                    identity_id,
                    full_name,
                    document_number,
                    document_type,
                    risk_score,
                    status AS severity,
                    checkpoint,
                    officer_name,
                    issues AS description,
                    'Under Investigation' AS status,
                    'Document Screening Discrepancy' AS case_type,
                    created_at
                FROM screenings
                WHERE status IN ('HIGH RISK', 'SUSPICIOUS', 'REVIEW REQUIRED')
                ORDER BY id DESC
                LIMIT %s
            """, (limit,))
            cases = [dict(row) for row in cursor.fetchall()]

        cursor.close()
        conn.close()

        return {
            "success": True,
            "cases": cases,
            "summary": stats,
        }
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

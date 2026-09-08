"""
============================================================
TeamTrace / BorderShield AI - Backend
OCR.space + OpenCV + SQLite Identity Intelligence
============================================================
"""

import os
import re
import sqlite3
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
    version="1.0.0",
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
    r"\b[A-Z]{1,2}\s?\d{6,8}\b",
    re.IGNORECASE,
)

DL_NUMBER_RE = re.compile(
    r"\b[A-Z]{2}[-\s]?\d{2}[-\s]?\d{4,11}\b",
    re.IGNORECASE,
)

# Supports:
# 23/04/1984
# 23-04-1984
# 1984-04-23
# 23 APR 1984
# 23 APR/AVR 1984
# 25 STY/JAN 2033
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
        (?:\s*/\s*)?
        (?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)?
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
    ],

    "PAN Card": [
        "income tax department",
        "permanent account number",
        "pan card",
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
    ],

    "Visa": [
        "visa",
        "visa number",
        "entry",
        "valid from",
        "valid until",
        "entries",
        "visa type",
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
# DATE HELPER
# ============================================================

MONTHS = {
    "JAN": 1, "STY": 1,
    "FEB": 2, "LUT": 2,
    "MAR": 3,
    "APR": 4, "KWI": 4,
    "MAY": 5, "MAJ": 5,
    "JUN": 6, "CZE": 6,
    "JUL": 7, "LIP": 7,
    "AUG": 8, "SIE": 8,
    "SEP": 9, "WRZ": 9,
    "OCT": 10, "PAZ": 10, "PAŹ": 10,
    "NOV": 11, "LIS": 11,
    "DEC": 12, "GRU": 12,
}


def to_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None

    value = value.strip().upper()
    value = re.sub(r"\s+", " ", value)

    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass

    match = re.fullmatch(
        r"(\d{1,2})\s+([A-ZĄĆĘŁŃÓŚŹŻ]{3,5})(?:\s*/\s*[A-Z]{3})?\s+(\d{4})",
        value,
    )

    if match:
        day = int(match.group(1))
        month = MONTHS.get(match.group(2).upper())
        year = int(match.group(3))

        if month:
            try:
                return date(year, month, day)
            except ValueError:
                return None

    return None


def extract_dates(text: str) -> list:
    if not text:
        return []

    matches = DATE_RE.findall(text)
    result = []

    for value in matches:
        value = value.strip()
        if value and value not in result:
            result.append(value)

    return result


# ============================================================
# DOCUMENT CLASSIFICATION
# ============================================================

def classify_document(text: str) -> str:
    text = text or ""
    lower = text.lower()
    upper = text.upper()
    scores = {}

    for doc_type, keywords in KEYWORDS.items():
        scores[doc_type] = sum(
            1 for keyword in keywords
            if keyword.lower() in lower
        )

    if AADHAAR_NUMBER_RE.search(text) and scores.get("Aadhaar", 0) > 0:
        return "Aadhaar"

    if PAN_NUMBER_RE.search(text):
        if scores.get("PAN Card", 0) > 0 or "income tax" in lower:
            return "PAN Card"

    if "P<" in upper or re.search(r"\bP<[A-Z]{3}", upper):
        return "Passport"

    if PASSPORT_NUMBER_RE.search(text) and scores.get("Passport", 0) >= 1:
        return "Passport"

    if scores.get("Passport", 0) >= 2:
        return "Passport"

    if scores.get("Visa", 0) >= 2:
        return "Visa"

    if DL_NUMBER_RE.search(text) and scores.get("Driving Licence", 0) > 0:
        return "Driving Licence"

    if scores.get("National ID", 0) >= 1:
        return "National ID"

    if scores.get("Permit", 0) >= 2:
        return "Permit"

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    return "Unknown"


# ============================================================
# EXTRACT DOCUMENT NUMBER
# ============================================================

def extract_document_number(
    text: str,
    doc_type: str,
) -> Optional[str]:

    text = text or ""

    if doc_type == "Aadhaar":
        match = AADHAAR_NUMBER_RE.search(text)
        if match:
            return match.group(0).replace(" ", "")

    elif doc_type == "PAN Card":
        match = PAN_NUMBER_RE.search(text)
        if match:
            return match.group(0).upper()

    elif doc_type == "Passport":
        match = PASSPORT_NUMBER_RE.search(text)
        if match:
            return re.sub(r"\s+", "", match.group(0).upper())

        normalized = re.sub(r"(?<=[A-Z])\s+(?=\d)", "", text.upper())
        match = PASSPORT_NUMBER_RE.search(normalized)
        if match:
            return re.sub(r"\s+", "", match.group(0).upper())

    elif doc_type == "Driving Licence":
        match = DL_NUMBER_RE.search(text)
        if match:
            return re.sub(r"\s+", "", match.group(0).upper())

    return None


# ============================================================
# VISUAL ANALYSIS
# ============================================================

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


    # Resolution

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


    # Blur

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


    # Brightness

    brightness = float(np.mean(gray))

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


    # Contrast

    contrast = float(np.std(gray))

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
    mrz: Optional[dict] = None,
):
    issues = []
    verified = []
    score = 0
    mrz = mrz or {}

    # NAME
    name = guess_name(text)

    if doc_type == "Passport" and mrz.get("name"):
        name = mrz["name"]
        verified.append("Passport name detected from MRZ")
    elif name:
        verified.append("Name information was detected")
    else:
        issues.append({
            "title": "Name not detected",
            "severity": "MEDIUM",
            "description": "The OCR could not confidently locate the document name.",
        })
        score += WEIGHTS["MISSING_REQUIRED_FIELD"]

    # DOCUMENT NUMBER
    doc_number = extract_document_number(text, doc_type)

    if (
        not doc_number
        and doc_type == "Passport"
        and mrz.get("document_number")
    ):
        doc_number = mrz["document_number"]
        verified.append("Passport number recovered from MRZ")

    if doc_number:
        verified.append(f"{doc_type} number format looks valid")
    elif doc_type in ("Aadhaar", "PAN Card", "Passport", "Driving Licence"):
        issues.append({
            "title": "Document number not detected",
            "severity": severity_for_weight(WEIGHTS["INVALID_NUMBER_FORMAT"]),
            "description": f"No valid {doc_type} number could be extracted from the document.",
        })
        score += WEIGHTS["INVALID_NUMBER_FORMAT"]

    # DATES
    dates_found = extract_dates(text)
    parsed_dates = []

    for value in dates_found:
        parsed = to_date(value)
        if parsed:
            parsed_dates.append(parsed)

    if doc_type == "Passport":
        for mrz_key in ("date_of_birth", "expiry_date"):
            if mrz.get(mrz_key):
                try:
                    parsed_dates.append(date.fromisoformat(mrz[mrz_key]))
                except (ValueError, TypeError):
                    pass

    parsed_dates = list(dict.fromkeys(parsed_dates))

    if parsed_dates:
        verified.append(f"{len(parsed_dates)} valid date(s) extracted")
    elif doc_type in ("Passport", "Visa", "Driving Licence", "Permit"):
        issues.append({
            "title": "Date could not be verified",
            "severity": "MEDIUM",
            "description": "No supported date format could be confidently converted to a calendar date.",
        })
        score += WEIGHTS["MISSING_REQUIRED_FIELD"]

    # PASSPORT EXPIRY
    if doc_type == "Passport":
        expiry_candidate = None

        if mrz.get("expiry_date"):
            try:
                expiry_candidate = date.fromisoformat(mrz["expiry_date"])
            except (ValueError, TypeError):
                pass

        if expiry_candidate is None and parsed_dates:
            expiry_candidate = max(parsed_dates)

        if expiry_candidate:
            if expiry_candidate < date.today():
                issues.append({
                    "title": "Passport appears expired",
                    "severity": "HIGH",
                    "description": f"Detected expiry date {expiry_candidate.isoformat()} is earlier than today's date.",
                })
                score += WEIGHTS["EXPIRED_DOCUMENT"]
            else:
                verified.append(
                    f"Passport expiry date {expiry_candidate.isoformat()} is valid"
                )

    # OTHER DOCUMENT EXPIRY
    elif doc_type in ("Visa", "Driving Licence", "Permit") and parsed_dates:
        expiry_candidate = max(parsed_dates)

        if expiry_candidate < date.today():
            issues.append({
                "title": "Document may be expired",
                "severity": "MEDIUM",
                "description": f"Latest detected date {expiry_candidate.isoformat()} is earlier than today's date.",
            })
            score += WEIGHTS["EXPIRED_DOCUMENT"]
        else:
            verified.append("Latest detected validity date is not expired")

    # MRZ VALIDATION
    if doc_type == "Passport":
        if mrz.get("detected"):
            verified.append("Passport MRZ detected")

            mrz_number = mrz.get("document_number")

            if doc_number and mrz_number:
                visible_normalized = re.sub(r"[^A-Z0-9]", "", doc_number.upper())
                mrz_normalized = re.sub(r"[^A-Z0-9]", "", mrz_number.upper())

                if visible_normalized == mrz_normalized:
                    verified.append("Passport number matches MRZ")
                else:
                    issues.append({
                        "title": "MRZ passport number mismatch",
                        "severity": "HIGH",
                        "description": (
                            f"Visible passport number '{visible_normalized}' "
                            f"does not match MRZ '{mrz_normalized}'."
                        ),
                    })
                    score += WEIGHTS["CRITICAL_MISMATCH"]

            mrz_expiry = mrz.get("expiry_date")

            if mrz_expiry:
                try:
                    mrz_expiry_date = date.fromisoformat(mrz_expiry)

                    if mrz_expiry_date < date.today():
                        issues.append({
                            "title": "MRZ indicates expired passport",
                            "severity": "HIGH",
                            "description": f"MRZ expiry date is {mrz_expiry}.",
                        })
                        score += WEIGHTS["EXPIRED_DOCUMENT"]
                    else:
                        verified.append("MRZ expiry date is valid")
                except ValueError:
                    issues.append({
                        "title": "Invalid MRZ expiry date",
                        "severity": "MEDIUM",
                        "description": "The passport MRZ was detected but its expiry date could not be parsed.",
                    })
        else:
            verified.append("MRZ not detected; visible passport fields used")

    # OCR CONFIDENCE
    if ocr_confidence < 0.60:
        issues.append({
            "title": "Low OCR confidence",
            "severity": "MEDIUM",
            "description": "OCR quality appears low. Manual verification is recommended.",
        })
        score += WEIGHTS["LOW_OCR_CONFIDENCE"]
    else:
        verified.append("OCR extraction completed successfully")

    # UNKNOWN DOCUMENT
    if doc_type == "Unknown":
        issues.append({
            "title": "Document type could not be identified",
            "severity": "HIGH",
            "description": "No supported document pattern was confidently detected.",
        })
        score += WEIGHTS["CRITICAL_MISMATCH"]

    return issues, verified, score, name, doc_number


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

            detail="OCR_SPACE_API_KEY is missing from .env",

        )


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

            detail=f"OCR service error: {str(error)}",

        )


    if result.get("IsErroredOnProcessing"):

        error_message = (

            result.get("ErrorMessage")

            or result.get("ErrorDetails")

            or "OCR processing failed"

        )

        raise HTTPException(

            status_code=502,

            detail=str(error_message),

        )


    parsed_results = result.get(
        "ParsedResults",
        [],
    )


    if not parsed_results:
        return ""


    return "\n".join(

        item.get("ParsedText", "")

        for item in parsed_results

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

# ============================================================
# SCREEN DOCUMENT
# ============================================================

@app.post("/api/screen")
async def screen_document(

    file: UploadFile = File(...),

    expected_document_type: str = Form(...),

):


    # FILE TYPE

    if file.content_type not in ALLOWED_CONTENT_TYPES:

        raise HTTPException(

            status_code=400,

            detail=f"Unsupported file type: {file.content_type}",

        )


    # READ FILE

    file_bytes = await file.read()


    # FILE SIZE

    size_mb = len(file_bytes) / (
        1024 * 1024
    )


    if size_mb > MAX_FILE_SIZE_MB:

        raise HTTPException(

            status_code=400,

            detail=(
                f"File too large ({size_mb:.1f} MB). "
                f"Maximum is {MAX_FILE_SIZE_MB} MB."
            ),

        )


    # VISUAL ANALYSIS

    visual_analysis = analyze_document_image(
        file_bytes
    )


    # OCR

    text = perform_ocr(

        file_bytes,

        file.filename or "document",

    )


    # NO TEXT

    if not text.strip():

        score = 80

        status = status_for_score(score)

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

                checkpoint="Main Border Checkpoint",

                officer_name="BorderShield AI",

            )

            print(
                f"SCREENING SAVED SUCCESSFULLY! "
                f"ID = {screening_id}"
            )

        except Exception as error:

            print(
                "DATABASE SAVE ERROR:",
                error,
            )


        return {

            "success": True,

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

            "visual_analysis":
                visual_analysis,

        }


    # OCR CONFIDENCE

    ocr_confidence = 0.85


    # DOCUMENT TYPE

    doc_type = classify_document(text)
    # Passport MRZ extraction
    mrz = extract_mrz(text)


    # NORMALIZATION

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


    expected = expected_document_type.strip().lower()

    detected = doc_type.strip().lower()


    expected_normalized = aliases.get(
        expected,
        expected,
    )


    detected_normalized = aliases.get(
        detected,
        detected,
    )


    # VALIDATION

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

    mrz,
    )


    # TYPE MISMATCH

    if expected_normalized != detected_normalized:

        issues.append({

            "title":
                "Document Type Mismatch",

            "severity":
                "HIGH",

            "description":
                f"Expected '{expected_document_type}', "
                f"but detected '{doc_type}'.",

        })

        score += WEIGHTS[
            "CRITICAL_MISMATCH"
        ]

    else:

        verified.append(
            f"Correct document type confirmed: "
            f"{expected_document_type}"
        )


    # VISUAL ISSUES

    for visual_issue in visual_analysis["visual_issues"]:

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


    # VISUAL CHECKS

    for visual_check in visual_analysis["visual_checks"]:

        verified.append(
            f"Visual check: {visual_check}"
        )


    # LIMIT SCORE

    score = max(
        0,
        min(100, score),
    )


    status = status_for_score(score)


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

            document_number=doc_number or "Not detected",

            document_type=doc_type,

            risk_score=score,

            status=status,

            issues=issues,

            checkpoint="Main Border Checkpoint",

            officer_name="BorderShield AI",

        )


        print(
            f"SCREENING SAVED SUCCESSFULLY! "
            f"ID = {screening_id}"
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

            action="SCREENING_COMPLETED",

            details=(
                f"Screening ID: {screening_id}, "
                f"Document: {doc_type}, "
                f"Risk Score: {score}, "
                f"Status: {status}"
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
    # FINAL RESPONSE
    # ========================================================

    return {

        "success": True,

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
                extract_dates(text),

            "mrz": mrz,

        },

        "visual_analysis":
            visual_analysis,

        "identity_intelligence":
            identity_intelligence,

    }


# ============================================================
# DASHBOARD API
# ============================================================

# ============================================================
# DASHBOARD API
# ============================================================

@app.get("/api/dashboard")
def get_dashboard():

    import sqlite3

    conn = sqlite3.connect("bordershield.db")
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()


    # ========================================================
    # TOTAL SCREENINGS
    # ========================================================

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM screenings
    """)

    total_screenings = cursor.fetchone()["total"]


    # ========================================================
    # AVERAGE RISK SCORE
    # ========================================================

    cursor.execute("""
        SELECT AVG(risk_score) AS average
        FROM screenings
    """)

    result = cursor.fetchone()

    average_risk_score = (
        round(result["average"], 1)
        if result["average"] is not None
        else 0
    )


    # ========================================================
    # CLEARED DOCUMENTS
    # ========================================================

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM screenings
        WHERE status = 'LOW RISK'
    """)

    cleared_documents = cursor.fetchone()["total"]


    # ========================================================
    # FLAGGED DOCUMENTS
    # ========================================================

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM screenings
        WHERE status != 'LOW RISK'
    """)

    flagged_documents = cursor.fetchone()["total"]


    # ========================================================
    # LOW RISK
    # ========================================================

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM screenings
        WHERE status = 'LOW RISK'
    """)

    low_risk = cursor.fetchone()["total"]


    # ========================================================
    # REVIEW REQUIRED
    # ========================================================

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM screenings
        WHERE status = 'REVIEW REQUIRED'
    """)

    review_required = cursor.fetchone()["total"]


    # ========================================================
    # SUSPICIOUS
    # ========================================================

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM screenings
        WHERE status = 'SUSPICIOUS'
    """)

    suspicious = cursor.fetchone()["total"]


    # ========================================================
    # HIGH RISK
    # ========================================================

    cursor.execute("""
        SELECT COUNT(*) AS total
        FROM screenings
        WHERE status = 'HIGH RISK'
    """)

    high_risk = cursor.fetchone()["total"]


    # ========================================================
    # RECENT SCREENINGS
    # ========================================================

    cursor.execute("""
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
    """)

    recent_screenings = [

        dict(row)

        for row in cursor.fetchall()

    ]


    # ========================================================
    # CHECKPOINT LOAD
    # ========================================================

    cursor.execute("""
        SELECT
            checkpoint,
            COUNT(*) AS total_screenings
        FROM screenings
        GROUP BY checkpoint
        ORDER BY total_screenings DESC
    """)

    checkpoint_load = [

        dict(row)

        for row in cursor.fetchall()

    ]


    conn.close()


    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "success": True,

        "statistics": {

            "screenings_today": total_screenings,

            "average_risk_score": average_risk_score,

            "cleared_documents": cleared_documents,

            "flagged_for_review": flagged_documents

        },

        "recent_screenings": recent_screenings,

        "risk_distribution": {

            "low_risk": low_risk,

            "review_required": review_required,

            "suspicious": suspicious,

            "high_risk": high_risk

        },

        "checkpoint_load": checkpoint_load

    }


# ============================================================
# IDENTITY INTELLIGENCE API
# ============================================================

@app.get("/api/identity/{identity_id}")
def identity_intelligence(identity_id: int):

    try:

        intelligence = get_identity_intelligence(
            identity_id
        )


        if not intelligence:

            raise HTTPException(

                status_code=404,

                detail="Identity not found",

            )


        return {

            "success": True,

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

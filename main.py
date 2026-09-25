"""
============================================================
TeamTrace / BorderShield AI - Backend
OCR.space + OpenCV + PostgreSQL Identity Intelligence
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


def status_for_score(score: int, critical_missing: int = 0) -> str:
    """Conservative status: missing core fields cannot be LOW RISK."""
    if critical_missing >= 3:
        return "HIGH RISK"
    if critical_missing >= 2:
        return "SUSPICIOUS"
    if score >= 76:
        return "HIGH RISK"
    if score >= 51:
        return "SUSPICIOUS"
    if score >= 21:
        return "REVIEW REQUIRED"
    return "LOW RISK"



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

DATE_RE = re.compile(
    r"\b("
    r"\d{2}[/-]\d{2}[/-]\d{4}"
    r"|"
    r"\d{4}[/-]\d{2}[/-]\d{2}"
    r")\b"
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
        "republic of india",
        "nationality",
        "surname",
        "given name",
        "place of birth",
        "date of issue",
        "date of expiry",
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
# TEXT / DATE HELPERS
# ============================================================

def normalize_text(text: str) -> str:
    text = (text or "").replace("\r", "\n")
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_value(value: str) -> str:
    value = (value or "").strip()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" :-#|")


def lines_of(text: str) -> list[str]:
    return [clean_value(x) for x in normalize_text(text).splitlines() if clean_value(x)]


MONTH_MAP = {
    "STY/JAN": "JAN", "LUT/FEB": "FEB", "KWI/APR": "APR",
    "MAJ/MAY": "MAY", "CZE/JUN": "JUN", "LIP/JUL": "JUL",
    "SIE/AUG": "AUG", "WRZ/SEP": "SEP", "PAZ/OCT": "OCT",
    "LIS/NOV": "NOV", "GRU/DEC": "DEC",
}


def to_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    value = value.strip().upper().replace(".", "")
    value = re.sub(r"\s+", " ", value)
    for old, new in MONTH_MAP.items():
        value = value.replace(old, new)
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    return None


def parse_mrz_date(raw: str, dob: bool = False) -> Optional[str]:
    if not raw or not re.fullmatch(r"\d{6}", raw):
        return None
    yy, mm, dd = int(raw[:2]), int(raw[2:4]), int(raw[4:6])
    if not (1 <= mm <= 12 and 1 <= dd <= 31):
        return None
    year = (1900 + yy if yy >= 30 else 2000 + yy) if dob else 2000 + yy
    try:
        return date(year, mm, dd).isoformat()
    except ValueError:
        return None


# ============================================================
# PASSPORT MRZ EXTRACTION - OCR-TOLERANT
# ============================================================

def _parse_mrz_line2(line: str) -> Optional[dict]:
    compact = re.sub(r"[^A-Z0-9]", "", (line or "").upper())
    if len(compact) < 25:
        return None

    # Standard TD3 layout begins with 9-character passport/document number.
    doc_no = compact[:9].rstrip("<")
    if not doc_no:
        return None

    # Prefer the standard nationality position, then fall back to nearby 3-letter token.
    nationality = None
    if len(compact) >= 13 and re.fullmatch(r"[A-Z]{3}", compact[10:13]):
        nationality = compact[10:13]

    # Find DOB + optional check + sex + expiry. OCR may lose '<' fillers.
    m = re.search(r"(\d{6})\d?[MF](\d{6})", compact)
    if not m:
        m = re.search(r"(\d{6})(?:[MF])?(\d{6})", compact)
    if not m:
        return None

    if not nationality:
        prefix = compact[:m.start()]
        tokens = re.findall(r"[A-Z]{3}", prefix)
        if tokens:
            nationality = tokens[-1]

    sex = None
    between = compact[m.start()+6:m.start()+8]
    if "M" in between:
        sex = "M"
    elif "F" in between:
        sex = "F"

    return {
        "document_number": doc_no,
        "nationality": nationality,
        "dob_raw": m.group(1),
        "sex": sex,
        "expiry_raw": m.group(2),
    }


def is_valid_person_name(value: str) -> bool:
    value = clean_value(value)
    if not (3 <= len(value) <= 80) or not re.search(r"[A-Za-z]", value):
        return False
    if re.search(r"\d", value):
        return False
    upper = value.upper()
    bad = {
        "PASSPORT","PASZPORT","VISA","PERMIT","LICENCE","LICENSE",
        "NATIONALITY","GOVERNMENT","REPUBLIC","IDENTIFICATION","AUTHORITY",
        "TRANSPORT","DEPARTMENT","AADHAAR","AADHAR","UIDAI","PAN","CARD",
        "DATE","BIRTH","EXPIRY","VALID","NUMBER","ISSUE","ISSUED","SURNAME",
        "GIVEN","NAME","SEX","TYPE","ENTRY","ENTRIES","INCOME","TAX",
    }
    if upper in bad or set(re.findall(r"[A-Z]+", upper)) & bad:
        return False
    return bool(re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ .'-]+", value))


def extract_passport_mrz(text: str) -> dict:
    result = {
        "detected": False, "document_number": None, "nationality": None,
        "date_of_birth": None, "sex": None, "expiry_date": None,
        "name": None, "line1": None, "line2": None,
    }
    candidates=[]
    for raw in lines_of(text):
        squashed=re.sub(r"[^A-Z0-9<]", "", raw.upper())
        if len(squashed)>=25:
            candidates.append((raw,squashed))

    for i,(raw1,line1) in enumerate(candidates):
        if not re.match(r"^P(?:<|K)?[A-Z]{3}", line1):
            continue
        for raw2,line2 in candidates[i+1:i+3]:
            parsed=_parse_mrz_line2(line2)
            if not parsed:
                continue
            result["detected"]=True
            result["line1"]=raw1
            result["line2"]=raw2
            result["document_number"]=parsed["document_number"]
            result["nationality"]=parsed["nationality"]
            result["sex"]=parsed["sex"]
            result["date_of_birth"]=parse_mrz_date(parsed["dob_raw"],True)
            result["expiry_date"]=parse_mrz_date(parsed["expiry_raw"],False)
            name_line=re.sub(r"[^A-Z<]", "", line1)
            nm=re.match(r"^P(?:<|K)?[A-Z]{3}(.*)$",name_line)
            if nm:
                candidate=clean_value(re.sub(r"<+", " ", nm.group(1)))
                if is_valid_person_name(candidate):
                    result["name"]=candidate
            return result
    return result


# ============================================================
# DOCUMENT CLASSIFICATION
# ============================================================

def classify_document(text: str) -> str:
    lower=normalize_text(text).lower()
    scores={doc:sum(1 for k in keys if k in lower) for doc,keys in KEYWORDS.items()}

    mrz=extract_passport_mrz(text)
    if mrz["detected"]:
        return "Passport"
    if AADHAAR_NUMBER_RE.search(text) and scores["Aadhaar"]>0:
        return "Aadhaar"
    if PAN_NUMBER_RE.search(text) and (scores["PAN Card"]>0 or "income tax" in lower):
        return "PAN Card"
    if scores["Passport"]>=2:
        return "Passport"
    if scores["Visa"]>=2:
        return "Visa"
    if DL_NUMBER_RE.search(text) and scores["Driving Licence"]>0:
        return "Driving Licence"
    if scores["Driving Licence"]>=2:
        return "Driving Licence"
    if scores["National ID"]>=1:
        return "National ID"
    if scores["Permit"]>=2:
        return "Permit"
    best=max(scores,key=scores.get)
    return best if scores[best]>0 else "Unknown"



# ============================================================
# DOCUMENT NUMBER EXTRACTION
# ============================================================

def _label_pattern(label: str) -> str:
    return r"\s*".join(re.escape(x) for x in re.findall(r"[A-Z0-9]+", label.upper()))


def _value_after_labels(text: str, labels: list[str], min_len=4, max_len=30) -> Optional[str]:
    upper=normalize_text(text).upper()
    for label in labels:
        p=_label_pattern(label)
        m=re.search(rf"{p}\s*[:#\-]?\s*([A-Z0-9][A-Z0-9 ./\-]{{{min_len-1},{max_len-1}}})",upper,re.I)
        if m:
            return clean_value(m.group(1)).rstrip("./")
        lines=lines_of(upper)
        for i,line in enumerate(lines):
            if re.fullmatch(rf"{p}\s*[:#\-]?",line,re.I) and i+1<len(lines):
                candidate=clean_value(lines[i+1])
                if min_len<=len(candidate)<=max_len:
                    return candidate
    return None


def extract_document_number(text: str, doc_type: str) -> Optional[str]:
    upper=normalize_text(text).upper()

    if doc_type=="Passport":
        mrz=extract_passport_mrz(text)
        if mrz["document_number"]:
            return mrz["document_number"]
        candidate=_value_after_labels(text,["PASSPORT NUMBER","PASSPORT NO","DOCUMENT NUMBER"],6,12)
        if candidate and re.fullmatch(r"[A-Z0-9]{6,10}",re.sub(r"\W","",candidate)):
            return re.sub(r"\W","",candidate)
        m=PASSPORT_NUMBER_RE.search(upper)
        return m.group(0).replace(" ","") if m else None

    if doc_type=="Visa":
        candidate=_value_after_labels(text,["VISA NUMBER","VISA NO","VISA ID","DOCUMENT NUMBER","DOCUMENT NO","CONTROL NUMBER","CONTROL NO","REFERENCE NUMBER"],5,25)
        if candidate:
            return re.sub(r"[^A-Z0-9-]","",candidate)
        for line in lines_of(upper):
            if "VISA" in line:
                for token in re.findall(r"\b[A-Z0-9]{5,20}\b",line):
                    if token not in {"VISA","NUMBER","DOCUMENT","ENTRY","TYPE"}:
                        return token
        return None

    if doc_type=="Aadhaar":
        m=AADHAAR_NUMBER_RE.search(text)
        if m: return m.group(0).replace(" ","")
        candidate=_value_after_labels(text,["AADHAAR NUMBER","AADHAAR NO","AADHAAR","UID NUMBER","UID"],8,20)
        if candidate:
            digits=re.sub(r"\D","",candidate)
            if len(digits)==12: return digits
        return None

    if doc_type=="PAN Card":
        m=PAN_NUMBER_RE.search(upper)
        if m: return m.group(0).upper()
        candidate=_value_after_labels(text,["PAN NUMBER","PAN NO","PERMANENT ACCOUNT NUMBER"],10,15)
        if candidate:
            candidate=re.sub(r"[^A-Z0-9]","",candidate)
            if re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]",candidate): return candidate
        return None

    if doc_type=="Driving Licence":
        candidate=_value_after_labels(text,["DRIVING LICENCE NUMBER","DRIVING LICENSE NUMBER","LICENCE NUMBER","LICENSE NUMBER","LICENCE NO","LICENSE NO","DL NUMBER","DL NO"],8,25)
        if candidate: return re.sub(r"[^A-Z0-9-]","",candidate)
        m=DL_NUMBER_RE.search(upper)
        return m.group(0).replace(" ","") if m else None

    if doc_type=="National ID":
        candidate=_value_after_labels(text,["NATIONAL ID NUMBER","NATIONAL ID NO","IDENTITY NUMBER","IDENTITY NO","IDENTIFICATION NUMBER","IDENTIFICATION NO","ID NUMBER","ID NO"],6,25)
        return re.sub(r"[^A-Z0-9-]","",candidate) if candidate else None

    if doc_type=="Permit":
        candidate=_value_after_labels(text,["PERMIT NUMBER","PERMIT NO","PERMIT ID","REFERENCE NUMBER","REFERENCE NO","APPLICATION NUMBER","APPLICATION NO"],5,25)
        return re.sub(r"[^A-Z0-9-]","",candidate) if candidate else None

    return None


# ============================================================
# DATE EXTRACTION
# ============================================================

DATE_PATTERN=(r"\b(?:\d{2}[/-]\d{2}[/-]\d{4}|\d{4}[/-]\d{2}[/-]\d{2}|"
              r"\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s+\d{4}|"
              r"\d{1,2}\s+(?:STY/JAN|LUT/FEB|KWI/APR|MAJ/MAY|CZE/JUN|LIP/JUL|SIE/AUG|WRZ/SEP|PAZ/OCT|LIS/NOV|GRU/DEC)\s+\d{4})\b")


def extract_dates(text: str, doc_type: Optional[str]=None) -> list:
    upper=normalize_text(text).upper()
    found=re.findall(DATE_PATTERN,upper)
    if doc_type=="Passport":
        mrz=extract_passport_mrz(text)
        if mrz["date_of_birth"]: found.append(mrz["date_of_birth"])
        if mrz["expiry_date"]: found.append(mrz["expiry_date"])
    out=[]
    for value in found:
        value=clean_value(value)
        if value and value not in out: out.append(value)
    return out


def _extract_labeled_date(text: str, labels: list[str]) -> Optional[str]:
    upper=normalize_text(text).upper()
    for label in labels:
        p=_label_pattern(label)
        m=re.search(rf"{p}.{{0,60}}?({DATE_PATTERN})",upper,re.S)
        if m: return clean_value(m.group(1))
    return None


def extract_date_fields(text: str, doc_type: str) -> dict:
    result={"date_of_birth":None,"issue_date":None,"expiry_date":None}
    if doc_type=="Passport":
        mrz=extract_passport_mrz(text)
        result["date_of_birth"]=mrz["date_of_birth"]
        result["expiry_date"]=mrz["expiry_date"]
    if not result["date_of_birth"]:
        v=_extract_labeled_date(text,["DATE OF BIRTH","DOB","BIRTH DATE","BORN"])
        d=to_date(v) if v else None
        result["date_of_birth"]=d.isoformat() if d else v
    v=_extract_labeled_date(text,["DATE OF ISSUE","ISSUE DATE","ISSUED ON","VALID FROM"])
    d=to_date(v) if v else None
    result["issue_date"]=d.isoformat() if d else v
    if not result["expiry_date"]:
        v=_extract_labeled_date(text,["DATE OF EXPIRY","EXPIRY DATE","EXPIRY","VALID UNTIL","VALID TILL","VALID UP TO","VALID THROUGH","VALID TO","VISA EXPIRY","DATE OF VALIDITY","VALIDITY"])
        d=to_date(v) if v else None
        result["expiry_date"]=d.isoformat() if d else v
    return result



# ============================================================
# NAME EXTRACTION
# ============================================================

def guess_name(text: str, doc_type: Optional[str]=None) -> Optional[str]:
    if doc_type=="Passport":
        mrz=extract_passport_mrz(text)
        if mrz["name"]: return mrz["name"]

    labels=["FULL NAME","FULLNAME","GIVEN NAME","GIVEN NAMES","APPLICANT NAME","HOLDER NAME","PERSON NAME","LICENSE HOLDER","LICENCE HOLDER","NAME OF HOLDER","NAME OF APPLICANT","SURNAME","NAME"]
    lines=lines_of(text)

    for i,line in enumerate(lines):
        upper=line.upper()
        for label in labels:
            p=_label_pattern(label)
            m=re.search(rf"{p}\s*[:#\-]?\s*(.+)$",upper,re.I)
            if m:
                candidate=clean_value(m.group(1))
                if is_valid_person_name(candidate): return candidate
            if re.fullmatch(rf"{p}\s*[:#\-]?",upper,re.I) and i+1<len(lines):
                candidate=clean_value(lines[i+1])
                if is_valid_person_name(candidate): return candidate

    for line in lines:
        if is_valid_person_name(line):
            return line
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

def run_validation(doc_type: str, text: str, ocr_confidence: float):
    issues=[]; verified=[]; score=0; critical_missing=0
    name=guess_name(text,doc_type)
    doc_number=extract_document_number(text,doc_type)
    dates_found=extract_dates(text,doc_type)
    date_fields=extract_date_fields(text,doc_type)

    if name:
        verified.append(f"Name detected: {name}")
    else:
        critical_missing+=1; score+=WEIGHTS["MISSING_REQUIRED_FIELD"]
        issues.append({"title":"Name not detected","severity":"HIGH","description":"No reliable person-name field could be extracted from the OCR text."})

    if doc_number:
        verified.append(f"{doc_type} number detected: {doc_number}")
    else:
        critical_missing+=1; score+=WEIGHTS["INVALID_NUMBER_FORMAT"]
        issues.append({"title":"Document number not detected","severity":"HIGH","description":f"No reliable {doc_type} number could be extracted from the OCR text."})

    if dates_found:
        verified.append(f"Date information detected: {len(dates_found)} date value(s)")
    else:
        critical_missing+=1; score+=WEIGHTS["MISSING_REQUIRED_FIELD"]
        issues.append({"title":"Date information not detected","severity":"HIGH","description":"No readable date could be extracted from the document."})

    if date_fields["date_of_birth"]:
        verified.append(f"Date of birth detected: {date_fields['date_of_birth']}")
    if date_fields["issue_date"]:
        verified.append(f"Issue date detected: {date_fields['issue_date']}")

    expiry=date_fields["expiry_date"]
    if expiry:
        parsed=to_date(expiry)
        if parsed and parsed<date.today():
            score+=WEIGHTS["EXPIRED_DOCUMENT"]
            issues.append({"title":"Possible expired document","severity":"MEDIUM","description":f"Extracted expiry date is {parsed.isoformat()}, which is before today's date."})
        elif parsed:
            verified.append(f"Expiry date detected and currently valid: {parsed.isoformat()}")
        else:
            issues.append({"title":"Expiry date needs review","severity":"MEDIUM","description":f"An expiry value was found but could not be safely parsed: {expiry}"})

    if ocr_confidence<0.60:
        score+=WEIGHTS["LOW_OCR_CONFIDENCE"]
        issues.append({"title":"Low OCR confidence","severity":"MEDIUM","description":"OCR quality appears low."})
    else:
        verified.append("OCR extraction completed successfully")

    if doc_type=="Unknown":
        score+=WEIGHTS["CRITICAL_MISMATCH"]
        issues.append({"title":"Document type could not be identified","severity":"HIGH","description":"No supported document pattern was confidently detected."})

    return issues,verified,min(score,100),name,doc_number,dates_found,date_fields,critical_missing



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
# DATABASE HEALTH
# ============================================================

@app.get("/api/database-health")
def database_health():
    conn = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1 AS ok")
        result = cursor.fetchone()
        return {
            "success": True,
            "database": "PostgreSQL",
            "connected": bool(result and result["ok"] == 1),
        }
    except Exception as error:
        return {
            "success": False,
            "database": "PostgreSQL",
            "connected": False,
            "error": str(error),
        }
    finally:
        if conn:
            conn.close()


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
        dates_found,
        date_fields,
        critical_missing,
    ) = run_validation(

        doc_type,

        text,

        ocr_confidence,

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


    status = status_for_score(score, critical_missing)


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
        "screening_id": screening_id,
        "identity_id": identity_id,
        "document_type": doc_type,
        "expected_document_type": expected_document_type,
        "risk_score": score,
        "status": status,
        "message": "Document received and analyzed successfully.",
        "issues": issues,
        "verified_checks": verified,
        "extracted_information": {
            "name": name or "Not detected",
            "document_number": doc_number or "Not detected",
            "dates_found": dates_found,
            "date_of_birth": date_fields.get("date_of_birth"),
            "issue_date": date_fields.get("issue_date"),
            "expiry_date": date_fields.get("expiry_date"),
        },
        "extracted_fields": {
            "name_guess": name,
            "document_number": doc_number,
            "dates_found": dates_found,
            "date_of_birth": date_fields.get("date_of_birth"),
            "issue_date": date_fields.get("issue_date"),
            "expiry_date": date_fields.get("expiry_date"),
        },
        "passport_mrz": extract_passport_mrz(text) if doc_type == "Passport" else None,
        "ocr_text": text,
        "ocr_confidence": ocr_confidence,
        "visual_analysis": visual_analysis,
        "identity_intelligence": identity_intelligence,
    }


# ============================================================
# DASHBOARD API - POSTGRESQL
# ============================================================

@app.get("/api/dashboard")
def get_dashboard():

    conn = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) AS total FROM screenings")
        total_screenings = cursor.fetchone()["total"] or 0

        cursor.execute("SELECT AVG(risk_score) AS average FROM screenings")
        average_result = cursor.fetchone()
        average_risk_score = (
            round(float(average_result["average"]), 1)
            if average_result["average"] is not None
            else 0
        )

        cursor.execute(
            "SELECT COUNT(*) AS total FROM screenings WHERE status = %s",
            ("LOW RISK",),
        )
        cleared_documents = cursor.fetchone()["total"] or 0

        cursor.execute(
            "SELECT COUNT(*) AS total FROM screenings WHERE status <> %s",
            ("LOW RISK",),
        )
        flagged_documents = cursor.fetchone()["total"] or 0

        cursor.execute(
            "SELECT COUNT(*) AS total FROM screenings WHERE status = %s",
            ("LOW RISK",),
        )
        low_risk = cursor.fetchone()["total"] or 0

        cursor.execute(
            "SELECT COUNT(*) AS total FROM screenings WHERE status = %s",
            ("REVIEW REQUIRED",),
        )
        review_required = cursor.fetchone()["total"] or 0

        cursor.execute(
            "SELECT COUNT(*) AS total FROM screenings WHERE status = %s",
            ("SUSPICIOUS",),
        )
        suspicious = cursor.fetchone()["total"] or 0

        cursor.execute(
            "SELECT COUNT(*) AS total FROM screenings WHERE status = %s",
            ("HIGH RISK",),
        )
        high_risk = cursor.fetchone()["total"] or 0

        cursor.execute("""
            SELECT
                id, identity_id, full_name, document_number, document_type,
                risk_score, status, issues, checkpoint, officer_name, created_at
            FROM screenings
            ORDER BY id DESC
            LIMIT 10
        """)
        recent_screenings = [dict(row) for row in cursor.fetchall()]

        cursor.execute("""
            SELECT checkpoint, COUNT(*) AS total_screenings
            FROM screenings
            GROUP BY checkpoint
            ORDER BY total_screenings DESC
        """)
        checkpoint_load = [dict(row) for row in cursor.fetchall()]

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

    except Exception as error:
        print("DASHBOARD DATABASE ERROR:", error)
        raise HTTPException(
            status_code=500,
            detail=f"Dashboard database error: {str(error)}",
        )

    finally:
        if conn:
            conn.close()


# ============================================================
# AUDIT LOG API - POSTGRESQL
# ============================================================

@app.get("/api/audit")
def get_audit_logs(limit: int = 100):

    limit = max(1, min(limit, 500))
    conn = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, officer_name, action, details, created_at
            FROM audit_logs
            ORDER BY id DESC
            LIMIT %s
        """, (limit,))

        logs = [dict(row) for row in cursor.fetchall()]

        return {
            "success": True,
            "count": len(logs),
            "audit_logs": logs,
        }

    except Exception as error:
        print("AUDIT DATABASE ERROR:", error)
        raise HTTPException(
            status_code=500,
            detail=f"Audit database error: {str(error)}",
        )

    finally:
        if conn:
            conn.close()


@app.get("/api/audit/verify")
def verify_audit_chain():
    """
    Verifies that the audit log can be read consistently from PostgreSQL.

    The current database schema stores audit entries as normal records
    (id/officer/action/details/created_at); it does not contain hash-chain
    columns, so this endpoint does not claim cryptographic verification.
    """

    conn = None

    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, officer_name, action, details, created_at
            FROM audit_logs
            ORDER BY id ASC
        """)

        logs = [dict(row) for row in cursor.fetchall()]

        ids_are_sequential = all(
            logs[i]["id"] < logs[i + 1]["id"]
            for i in range(len(logs) - 1)
        )

        return {
            "success": True,
            "verified": True,
            "cryptographic_hash_chain": False,
            "message": (
                "Audit records were read successfully from PostgreSQL. "
                "The current schema does not include cryptographic hash fields."
            ),
            "total_records": len(logs),
            "record_order_valid": ids_are_sequential,
        }

    except Exception as error:
        print("AUDIT VERIFY ERROR:", error)
        raise HTTPException(
            status_code=500,
            detail=f"Audit verification error: {str(error)}",
        )

    finally:
        if conn:
            conn.close()


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

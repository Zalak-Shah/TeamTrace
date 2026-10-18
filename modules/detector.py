"""
============================================================
BorderShield AI - Document Type Detector
============================================================
Detects document type using OCR keywords, regex patterns,
and layout indicators, then routes to the appropriate module.
============================================================
"""

import re
from typing import Dict, Any, Tuple
from modules.base import DocumentType


# Regex patterns for fast identification
AADHAAR_NUM_RE = re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")
PAN_NUM_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE)
INDIAN_PASSPORT_NUM_RE = re.compile(r"\b[A-PR-WY][0-9]{7}\b")
GENERIC_PASSPORT_NUM_RE = re.compile(r"\b[A-Z]{1,2}[0-9]{6,8}\b", re.IGNORECASE)
DL_NUM_RE = re.compile(
    r"\b(?:AN|AP|AR|AS|BR|CG|CH|DD|DL|DN|GA|GJ|HP|HR|JH|JK|KA|KL"
    r"|LA|LD|MH|ML|MN|MP|MZ|NL|OD|PB|PY|RJ|SK|TN|TR|TS|UK|UP|WB)"
    r"[-\s/]?\d{2}[-\s/]?\d{4}[-\s/]?\d{5,8}\b",
    re.IGNORECASE,
)
# Generic/Older state formats (e.g. MH-04-2019-12345 or GJ0119980012345 or DL04123456789)
GENERIC_DL_RE = re.compile(r"\b[A-Z]{2}[-\s/]?\d{2}[-\s/]?\d{4,11}\b", re.IGNORECASE)
MRZ_TD3_RE = re.compile(r"P[<K][A-Z<]{3}[A-Z0-9<]{39}")

KEYWORDS: Dict[DocumentType, list] = {
    "aadhaar": [
        "unique identification authority",
        "government of india",
        "aadhaar",
        "aadhar",
        "uidai",
        "meraaadhaar",
        "enrollment no",
        "enrolment no",
        "my aadhaar",
        "vid",
    ],
    "pan": [
        "income tax department",
        "permanent account number",
        "govt of india",
        "incometax",
        "pan card",
        "pan",
    ],
    "passport": [
        "passport",
        "republic of india",
        "type p",
        "code ind",
        "given names",
        "surname",
        "nationality",
        "place of birth",
        "place of issue",
        "date of expiry",
        "date of issue",
    ],
    "driving_licence": [
        "driving licence",
        "driving license",
        "licence to drive",
        "license to drive",
        "union of india",
        "driving",
        "transport department",
        "sarathi",
        "parivahan",
        "motor vehicles",
        "motor vehicle",
        "form 7",
        "rto",
        "licencing authority",
        "licensing authority",
        "dl no",
        "d.l. no",
        "licence no",
        "license no",
        "non-transport",
        "nt valid",
        "cov",
        "lmv",
        "mcwg",
        "blood group",
    ],

    "visa": [
        "visa",
        "entry permit",
        "visa type",
        "number of entries",
        "valid for",
        "stay duration",
        "visa number",
        "immigration",
    ],
}


def detect_document_type(
    ocr_text: str,
    declared_type: str = "auto",
) -> Tuple[DocumentType, float, Dict[str, Any]]:
    """
    Detects the document type from OCR text and layout cues.
    Returns (detected_type, confidence_score, detection_details)
    """
    text = (ocr_text or "").lower()
    scores: Dict[DocumentType, int] = {
        "aadhaar": 0,
        "pan": 0,
        "passport": 0,
        "driving_licence": 0,
        "visa": 0,
    }

    # 1. Check strong structural markers (MRZ, specific number formats)
    if "p<" in text or "pk" in text or MRZ_TD3_RE.search(ocr_text or ""):
        scores["passport"] += 50

    if PAN_NUM_RE.search(ocr_text or ""):
        scores["pan"] += 40

    if DL_NUM_RE.search(ocr_text or "") or GENERIC_DL_RE.search(ocr_text or ""):
        scores["driving_licence"] += 45

    if AADHAAR_NUM_RE.search(ocr_text or ""):
        scores["aadhaar"] += 30

    # 2. Check keyword frequencies
    for doc_type, kws in KEYWORDS.items():
        for kw in kws:
            if kw in text:
                scores[doc_type] += 12

    # Strong overrides
    if "income tax department" in text and "permanent account number" in text:
        scores["pan"] += 40
    if "unique identification authority" in text or "uidai" in text:
        scores["aadhaar"] += 40
    if "republic of india" in text and "passport" in text:
        scores["passport"] += 40
    if any(k in text for k in ["driving licence", "driving license", "licence to drive", "license to drive", "transport department", "sarathi"]):
        scores["driving_licence"] += 50


    # 3. Find top candidate
    sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_type, top_score = sorted_candidates[0]

    # If declared type was manually chosen by officer and has at least moderate score, give it a boost
    normalized_declared = declared_type.lower().replace(" ", "_")
    if normalized_declared in scores and scores[normalized_declared] >= 15:
        if top_score < 40 or scores[normalized_declared] >= (top_score - 10):
            best_type = normalized_declared
            top_score = max(top_score, scores[normalized_declared] + 10)

    if top_score < 15:
        # Unable to determine conclusively
        if normalized_declared in scores:
            return normalized_declared, 30.0, {
                "detected": normalized_declared,
                "confidence": 30.0,
                "scores": scores,
                "fallback_to_declared": True,
            }
        return "unknown", 0.0, {
            "detected": "unknown",
            "confidence": 0.0,
            "scores": scores,
        }

    confidence = min(round((top_score / 100.0) * 100, 1), 99.0)
    return best_type, confidence, {
        "detected": best_type,
        "confidence": confidence,
        "scores": scores,
    }

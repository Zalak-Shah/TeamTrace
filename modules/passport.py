"""
============================================================
BorderShield AI - Passport Verification Module (Tier 3)
============================================================
Features:
1. ICAO Doc 9303 TD3 MRZ Parsing:
   - Identifies two 44-character MRZ lines (Line 1: P<..., Line 2: doc_no + checks)
   - Computes 7-3-1 weighting algorithm on:
     - Document Number check digit
     - Date of Birth check digit
     - Date of Expiry check digit
     - Composite check digit
2. Cross-Verification:
   - Compares MRZ extracted fields against Visual Inspection Zone (VIZ) OCR text
3. Limitations Reporting (CRITICAL PRINCIPLE):
   - Explicitly notes that MRZ verification alone CANNOT prove document authenticity
     against generated or photoshopped fakes.
   - Notes that physical biometric chip (e-Passport NFC) authentication is impossible
     from a 2D image upload.
============================================================
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from modules.base import BaseDocumentVerifier, VerificationResult


# ICAO 7-3-1 Weights
ICAO_WEIGHTS = [7, 3, 1]


def icao_char_value(c: str) -> int:
    """Returns numeric value of a character per ICAO Doc 9303."""
    if c == "<":
        return 0
    if c.isdigit():
        return int(c)
    if "A" <= c.upper() <= "Z":
        return ord(c.upper()) - ord("A") + 10
    return 0


def compute_icao_check_digit(data: str) -> int:
    """Computes check digit using 7-3-1 repeating weights modulo 10."""
    total = 0
    for idx, ch in enumerate(data):
        weight = ICAO_WEIGHTS[idx % 3]
        total += icao_char_value(ch) * weight
    return total % 10


def parse_and_validate_td3_mrz(ocr_text: str) -> Tuple[Optional[Dict[str, Any]], List[str], List[str]]:
    """
    Parses TD3 (Passport) MRZ and validates all check digits.
    Returns (mrz_data, issues, verified_checks)
    """
    issues = []
    verified = []

    lines = [re.sub(r"[^A-Z0-9<]", "", l.upper()) for l in (ocr_text or "").splitlines() if len(re.sub(r"[^A-Z0-9<]", "", l)) >= 30]

    line1, line2 = None, None
    for i in range(len(lines) - 1):
        l1, l2 = lines[i], lines[i + 1]
        if (l1.startswith("P<") or l1.startswith("PK")) and len(l1) >= 42 and len(l2) >= 42:
            line1, line2 = l1[:44], l2[:44]
            break

    if not line1 or not line2:
        return None, ["No valid 44-character 2-line TD3 MRZ detected in document."], []

    # ---------------- Line 1 Parsing ----------------
    issuing_country = line1[2:5].replace("<", "")
    name_segment = line1[5:]
    name_parts = name_segment.split("<<", 1)
    surname = name_parts[0].replace("<", " ").strip() if len(name_parts) >= 1 else ""
    given_names = name_parts[1].replace("<", " ").strip() if len(name_parts) >= 2 else ""
    full_name = f"{given_names} {surname}".strip()

    # ---------------- Line 2 Parsing ----------------
    raw_doc_num = line2[0:9]
    doc_num_check = line2[9]
    nationality = line2[10:13].replace("<", "")
    raw_dob = line2[13:19]
    dob_check = line2[19]
    gender = line2[20]
    raw_expiry = line2[21:27]
    expiry_check = line2[27]
    raw_optional = line2[28:42]
    composite_check = line2[43]

    clean_doc_num = raw_doc_num.replace("<", "")

    # ---------------- Validate Check Digits ----------------
    # 1. Document Number Check Digit
    expected_doc_check = compute_icao_check_digit(raw_doc_num)
    if doc_num_check.isdigit() and int(doc_num_check) == expected_doc_check:
        verified.append(f"Passport number check digit valid ({doc_num_check})")
    else:
        issues.append(f"Passport number check digit failed: expected {expected_doc_check}, found '{doc_num_check}'")

    # 2. Date of Birth Check Digit
    expected_dob_check = compute_icao_check_digit(raw_dob)
    if dob_check.isdigit() and int(dob_check) == expected_dob_check:
        verified.append(f"Date of birth check digit valid ({dob_check})")
    else:
        issues.append(f"Date of birth check digit failed: expected {expected_dob_check}, found '{dob_check}'")

    # 3. Expiry Date Check Digit
    expected_expiry_check = compute_icao_check_digit(raw_expiry)
    if expiry_check.isdigit() and int(expiry_check) == expected_expiry_check:
        verified.append(f"Expiry date check digit valid ({expiry_check})")
    else:
        issues.append(f"Expiry date check digit failed: expected {expected_expiry_check}, found '{expiry_check}'")

    # 4. Composite Check Digit (covers positions 0-10, 13-20, 21-43)
    composite_data = line2[0:10] + line2[13:20] + line2[21:43]
    expected_comp_check = compute_icao_check_digit(composite_data)
    if composite_check.isdigit() and int(composite_check) == expected_comp_check:
        verified.append(f"Overall composite MRZ check digit valid ({composite_check})")
    else:
        issues.append(f"Overall composite MRZ check digit failed: expected {expected_comp_check}, found '{composite_check}'")

    mrz_data = {
        "line1": line1,
        "line2": line2,
        "document_number": clean_doc_num,
        "issuing_country": issuing_country,
        "nationality": nationality,
        "name": full_name,
        "dob_raw": raw_dob,
        "expiry_raw": raw_expiry,
        "gender": gender,
    }

    return mrz_data, issues, verified


class PassportVerifier(BaseDocumentVerifier):
    """
    Dedicated Passport Verification Module.
    """

    def verify(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        ocr_text: str,
        extracted_fields: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        reasons = []
        limitations = [
            "MRZ checksum verification only proves syntactic consistency. Generated or altered documents with mathematically correct check digits cannot be detected by MRZ alone.",
            "Physical ICAO 9303 RFID contactless chip (e-Passport cryptographic signature) CANNOT be verified from 2D scans or images.",
            "Cross-checkpoint border database lookups (INTERPOL SLTD) require secure Tier 2 gateway integration.",
        ]
        checks = {}
        risk_score = 0
        extracted_fields = extracted_fields or {}

        # 1. Parse and validate MRZ
        mrz_data, mrz_issues, mrz_verified = parse_and_validate_td3_mrz(ocr_text)
        checks["mrz_analysis"] = {
            "mrz_detected": mrz_data is not None,
            "data": mrz_data,
            "issues": mrz_issues,
            "verified": mrz_verified,
        }

        if not mrz_data:
            reasons.append("CRITICAL: No readable 2-line ICAO TD3 MRZ could be parsed.")
            return VerificationResult(
                document_type="passport",
                verification_method="mrz_check",
                verification_tier=3,
                status="suspicious",
                risk_score=75,
                reasons=reasons,
                limitations=limitations,
                checks=checks,
            )

        # Check for check digit failures
        if mrz_issues:
            for issue in mrz_issues:
                reasons.append(f"MRZ Checksum Failure: {issue}")
            risk_score += 45
            status = "suspicious"
        else:
            reasons.append("All ICAO 9303 MRZ check digits (document number, DOB, expiry, composite) mathematically valid.")
            risk_score = 10
            status = "unverified"  # Tier 3 can only be 'unverified' or 'suspicious'

        # 2. Cross-check MRZ with Visual Inspection Zone (VIZ) OCR
        cross_mismatches = []
        mrz_num = mrz_data.get("document_number")
        viz_num = extracted_fields.get("document_number")
        if mrz_num and viz_num and mrz_num.upper() != viz_num.upper():
            cross_mismatches.append(f"Passport Number mismatch: MRZ has '{mrz_num}', visual field has '{viz_num}'.")

        mrz_name = mrz_data.get("name")
        viz_name = extracted_fields.get("name_guess")
        if mrz_name and viz_name and mrz_name.split()[0].lower() not in viz_name.lower():
            cross_mismatches.append(f"Name mismatch: MRZ primary surname '{mrz_name.split()[0]}' not found in printed card name.")

        checks["viz_cross_check"] = {
            "mismatches": cross_mismatches,
        }

        if cross_mismatches:
            reasons.extend(cross_mismatches)
            risk_score += 40
            status = "suspicious"
        elif not mrz_issues:
            reasons.append("MRZ and Visual Zone (VIZ) fields are mutually consistent.")

        # Cap score
        risk_score = min(max(risk_score, 0), 100)
        if risk_score >= 50:
            status = "suspicious"

        return VerificationResult(
            document_type="passport",
            verification_method="mrz_check",
            verification_tier=3,
            status=status,
            risk_score=risk_score,
            reasons=reasons,
            limitations=limitations,
            checks=checks,
        )

"""
============================================================
BorderShield AI - Driving Licence Verification Module
============================================================
Features:
1. Pluggable Issuer Lookup Interface (Tier 2):
   - Pluggable provider interface (Parivahan / Sarathi / DigiLocker).
   - Mock provider returns status "unverified: issuer check not connected".
   - Never fakes a pass.
2. State & RTO Format Heuristics (Tier 3):
   - Validates Indian State Codes (e.g. DL, MH, GJ, KA, UP, etc.).
   - Validates 15-character sequence and issue year plausibility.
============================================================
"""

import re
from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Dict, Optional, Tuple

from modules.base import BaseDocumentVerifier, VerificationResult


INDIAN_STATE_CODES = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Daman and Diu",
    "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "UK": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}

DL_REGEX = re.compile(
    r"\b([A-Z]{2})[-\s]?(\d{2})[-\s]?(\d{4})[-\s]?(\d{7})\b",
    re.IGNORECASE,
)


class DlIssuerProvider(ABC):
    """
    Abstract pluggable interface for Parivahan Sarathi driving licence verification.
    """

    @abstractmethod
    def verify_licence(self, dl_number: str, dob: Optional[str] = None) -> Tuple[bool, str, Dict[str, Any]]:
        pass


class MockParivahanProvider(DlIssuerProvider):
    """
    Default mock implementation returning unverified when government credentials are not configured.
    """

    def verify_licence(self, dl_number: str, dob: Optional[str] = None) -> Tuple[bool, str, Dict[str, Any]]:
        return False, "unverified: issuer check not connected", {
            "provider": "MoRTH Parivahan / Sarathi Gateway",
            "connected": False,
            "note": "Requires Ministry of Road Transport & Highways or DigiLocker institutional API access.",
        }


class DrivingLicenceVerifier(BaseDocumentVerifier):
    """
    Dedicated Driving Licence Verification Module.
    """

    def __init__(self, provider: Optional[DlIssuerProvider] = None):
        self.provider = provider or MockParivahanProvider()

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
            "Physical driving licence cards lack cryptographic digital signatures.",
            "Visual examination cannot verify licence validity, suspension, or endorsements without live Sarathi/Parivahan portal connectivity.",
        ]
        checks = {}
        risk_score = 0
        extracted_fields = extracted_fields or {}

        # Extract DL number
        doc_num = extracted_fields.get("document_number")
        dl_match = DL_REGEX.search(ocr_text) if not doc_num else DL_REGEX.search(doc_num)

        # -------------------------------------------------------------
        # 1. Issuer Lookup (Tier 2 Pluggable Provider)
        # -------------------------------------------------------------
        if dl_match or doc_num:
            target_num = dl_match.group(0) if dl_match else doc_num
            is_valid, issuer_msg, issuer_details = self.provider.verify_licence(target_num)
            checks["issuer_lookup"] = issuer_details
            if is_valid:
                reasons.append(f"Driving licence verified active in Sarathi database: {target_num}")
                return VerificationResult(
                    document_type="driving_licence",
                    verification_method="issuer_lookup",
                    verification_tier=2,
                    status="verified",
                    risk_score=5,
                    reasons=reasons,
                    limitations=[],
                    checks=checks,
                )
            else:
                reasons.append(f"Issuer Lookup: {issuer_msg}")

        # -------------------------------------------------------------
        # 2. State & RTO Format Heuristics (Tier 3)
        # -------------------------------------------------------------
        if dl_match:
            state_code = dl_match.group(1).upper()
            rto_code = dl_match.group(2)
            year_str = dl_match.group(3)
            seq_num = dl_match.group(4)

            state_name = INDIAN_STATE_CODES.get(state_code)
            checks["dl_format"] = {
                "dl_number": f"{state_code}{rto_code} {year_str}{seq_num}",
                "state_code": state_code,
                "state_name": state_name,
                "issue_year": int(year_str) if year_str.isdigit() else None,
            }

            if state_name:
                reasons.append(f"Parivahan jurisdiction valid: {state_name} ('{state_code}')")
                issue_year = int(year_str)
                current_year = date.today().year
                if 1960 <= issue_year <= current_year:
                    reasons.append(f"Licence issuance year plausibility confirmed ({issue_year}).")
                    risk_score = 25
                    status = "unverified"
                else:
                    reasons.append(f"Abnormal licence issue year: {issue_year}")
                    risk_score = 65
                    status = "suspicious"
            else:
                reasons.append(f"Invalid state jurisdiction prefix: '{state_code}' does not exist.")
                risk_score = 80
                status = "suspicious"
        else:
            reasons.append("No recognized Indian Driving Licence format (e.g. MH01 20180012345) detected.")
            risk_score = 75
            status = "suspicious"

        # Check transport department keywords
        has_transport = any(k in ocr_text.lower() for k in ["transport department", "driving licence", "driving license", "motor vehicles", "sarathi"])
        checks["dl_format_markers"] = has_transport
        if has_transport:
            reasons.append("Transport department statutory markings present.")
        else:
            reasons.append("Transport department statutory header markings missing.")
            risk_score += 15

        risk_score = min(max(risk_score, 0), 100)
        return VerificationResult(
            document_type="driving_licence",
            verification_method="forensics_only",
            verification_tier=3,
            status=status,
            risk_score=risk_score,
            reasons=reasons,
            limitations=limitations,
            checks=checks,
        )

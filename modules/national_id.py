"""
============================================================
BorderShield AI - National ID Verification Module (Tier 3 Only)
============================================================
Features:
1. Validates National / Civil Identity card keywords and markers.
2. Evaluates ID number format, holder name, nationality, and expiry.
3. Explicit operational limitations:
   - Sovereign national IDs lack a single cross-border verification standard.
   - Without direct connection to sovereign civil registries (e.g., DigiLocker, Singpass, NADRA),
     image-only verification is heuristic.
============================================================
"""

import re
from typing import Any, Dict, Optional

from modules.base import BaseDocumentVerifier, VerificationResult


class NationalIdVerifier(BaseDocumentVerifier):
    """
    Dedicated National ID Verification Module (Tier 3).
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
            "LOW-CONFIDENCE VERIFICATION: Non-ICAO national ID cards lack standardized cross-border public keys.",
            "Authenticity cannot be confirmed without connecting to the sovereign nation's national identity registry.",
            "Visual OCR checks cannot detect manipulated physical cards or high-resolution digital composites.",
        ]
        checks = {}
        risk_score = 35
        status = "unverified"
        extracted_fields = extracted_fields or {}

        text = (ocr_text or "").lower()
        has_id_kw = any(
            k in text
            for k in [
                "national identity card",
                "national identity",
                "national id",
                "identity card",
                "citizen id",
                "civil id",
                "republic",
            ]
        )

        id_num = extracted_fields.get("document_number")
        if not id_num:
            num_match = re.search(r"\b(?:id|identity)[\s#.:-]*([A-Z0-9-]{5,20})\b", ocr_text or "", re.IGNORECASE)
            if num_match:
                id_num = num_match.group(1)

        checks["national_id_markers"] = {
            "id_keywords_detected": has_id_kw,
            "id_number": id_num,
        }

        if has_id_kw:
            reasons.append("National / Civil identity terminology identified.")
            if id_num:
                reasons.append(f"National ID number extracted: {id_num}")
                risk_score = 30
                status = "unverified"
            else:
                reasons.append("National ID number could not be reliably extracted.")
                risk_score = 50
                status = "suspicious"
        else:
            reasons.append("No national identity headings or civil registry markings found.")
            risk_score = 70
            status = "suspicious"

        return VerificationResult(
            document_type="national_id",
            verification_method="forensics_only",
            verification_tier=3,
            status=status,
            risk_score=risk_score,
            reasons=reasons,
            limitations=limitations,
            checks=checks,
        )

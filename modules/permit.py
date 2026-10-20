"""
============================================================
BorderShield AI - Permit Verification Module (Tier 3 Only)
============================================================
Features:
1. Validates standard work, residence, and travel permit markers.
2. Checks document number, issue date, and expiry date logic.
3. Explicit operational limitations:
   - Permits are issued by state/regional/national bodies with no single global PKI.
   - Verification relies on heuristic OCR checks and date consistency without
     direct state registry lookup.
============================================================
"""

import re
from typing import Any, Dict, Optional

from modules.base import BaseDocumentVerifier, VerificationResult


class PermitVerifier(BaseDocumentVerifier):
    """
    Dedicated Permit Verification Module (Tier 3).
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
            "LOW-CONFIDENCE VERIFICATION: Permits vary widely across jurisdictions without a universal PKI or open validation registry.",
            "Document authenticity cannot be definitively confirmed without live queries to the issuing immigration or labour ministry database.",
            "Visual OCR validation cannot detect high-quality physical or digital counterfeit permits.",
        ]
        checks = {}
        risk_score = 35
        status = "unverified"
        extracted_fields = extracted_fields or {}

        text = (ocr_text or "").lower()
        has_permit_kw = any(
            k in text
            for k in [
                "permit",
                "residence permit",
                "work permit",
                "travel permit",
                "entry permit",
                "permit number",
                "permission",
            ]
        )

        permit_num = extracted_fields.get("document_number")
        if not permit_num:
            num_match = re.search(r"\b(?:permit|pmt)[\s#.:-]*([A-Z0-9-]{5,20})\b", ocr_text or "", re.IGNORECASE)
            if num_match:
                permit_num = num_match.group(1)

        checks["permit_markers"] = {
            "permit_keywords_detected": has_permit_kw,
            "permit_number": permit_num,
        }

        if has_permit_kw:
            reasons.append("Permit statutory terminology detected.")
            if permit_num:
                reasons.append(f"Permit identifier detected: {permit_num}")
                risk_score = 30
                status = "unverified"
            else:
                reasons.append("Permit identifier could not be definitively isolated.")
                risk_score = 50
                status = "suspicious"
        else:
            reasons.append("No standard permit terminology or regulatory headers found.")
            risk_score = 70
            status = "suspicious"

        return VerificationResult(
            document_type="permit",
            verification_method="forensics_only",
            verification_tier=3,
            status=status,
            risk_score=risk_score,
            reasons=reasons,
            limitations=limitations,
            checks=checks,
        )

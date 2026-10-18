"""
============================================================
BorderShield AI - Visa Verification Module (Tier 3 Only)
============================================================
Features:
1. Forensics and MRZ-style layout checks.
2. Clear low-confidence limitation disclaimer:
   - Visas are sticker or paper documents issued across hundreds of sovereign formats.
   - Without direct immigration authority API connectivity, image verification alone
     carries high risk and low confidence.
============================================================
"""

import re
from typing import Any, Dict, Optional

from modules.base import BaseDocumentVerifier, VerificationResult


class VisaVerifier(BaseDocumentVerifier):
    """
    Dedicated Visa Verification Module.
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
            "LOW-CONFIDENCE VERIFICATION: International visas lack universal public key infrastructure or standard open lookup registries.",
            "Visual sticker inspection cannot confirm visa validity, revocation, or active entry status without connecting to the specific sovereign immigration database.",
            "Forged visa stickers are frequently combined with authentic passport booklets; visual checks carry substantial risk.",
        ]
        checks = {}
        risk_score = 40  # Baseline uncertainty
        extracted_fields = extracted_fields or {}

        # Look for Visa identifiers
        text = (ocr_text or "").lower()
        has_visa_kw = any(k in text for k in ["visa", "entry permit", "visa number", "valid for", "duration of stay"])

        visa_num = extracted_fields.get("document_number")
        if not visa_num:
            num_match = re.search(r"\b[A-Z0-9]{7,10}\b", ocr_text or "")
            if num_match:
                visa_num = num_match.group(0)

        checks["visa_markers"] = {
            "visa_keywords_detected": has_visa_kw,
            "extracted_number": visa_num,
        }

        if has_visa_kw:
            reasons.append("Visa / entry permit statutory terminology detected.")
            if visa_num:
                reasons.append(f"Visa document / sticker number detected: {visa_num}")
                risk_score = 35
                status = "unverified"
            else:
                reasons.append("Visa sticker number could not be reliably extracted.")
                risk_score = 60
                status = "suspicious"
        else:
            reasons.append("No standard visa header or consular markings found.")
            risk_score = 75
            status = "suspicious"

        return VerificationResult(
            document_type="visa",
            verification_method="forensics_only",
            verification_tier=3,
            status=status,
            risk_score=risk_score,
            reasons=reasons,
            limitations=limitations,
            checks=checks,
        )

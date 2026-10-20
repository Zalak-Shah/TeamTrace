"""
============================================================
BorderShield AI - Base Verification Interface & Common Result Schema
============================================================
Guarantees consistent, standardized schema across all document types.

TIERS:
- Tier 1: Cryptographic / Issuer-signed (QR digital signature, PDF digital certificate)
- Tier 2: Issuer database lookup (e.g. Parivahan, NSDL, DigiLocker API)
- Tier 3: Forensics & heuristic checks only (ELA, EXIF, OCR, MRZ syntax)
============================================================
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


DocumentType = Literal[
    "aadhaar",
    "pan",
    "passport",
    "driving_licence",
    "visa",
    "national_id",
    "permit",
    "unknown",
]

VerificationMethod = Literal[
    "aadhaar_qr_signature",
    "pdf_signature",
    "issuer_lookup",
    "mrz_check",
    "forensics_only",
]

VerificationStatus = Literal[
    "verified",
    "suspicious",
    "failed",
    "unverified",
]


class VerificationResult(BaseModel):
    """
    Standard verification result returned by every document module.
    """
    document_type: DocumentType
    verification_method: VerificationMethod
    verification_tier: int = Field(..., ge=1, le=3, description="1=Cryptographic, 2=Issuer DB, 3=Forensics only")
    status: VerificationStatus
    risk_score: int = Field(..., ge=0, le=100, description="0=Clean, 100=Critical Risk/Forged")
    reasons: List[str] = Field(default_factory=list, description="Positive or negative diagnostic reasons")
    limitations: List[str] = Field(default_factory=list, description="Known operational limits of this verification method")
    checks: Dict[str, Any] = Field(default_factory=dict, description="Detailed per-check breakdown")

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class BaseDocumentVerifier(ABC):
    """
    Abstract base class for all document verification modules.
    """

    @abstractmethod
    def verify(
        self,
        file_bytes: bytes,
        filename: str,
        content_type: str,
        ocr_text: str,
        extracted_fields: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        """
        Verify the document and return a standardized VerificationResult.
        """
        pass

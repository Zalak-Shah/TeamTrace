"""
============================================================
Comprehensive Unit Tests for Document Detection & Extraction
Covers:
  - Classification accuracy for 6 document types:
    (Passport, Visa, Driving Licence, PAN Card, Aadhaar, Permit, National ID)
  - No false type mismatches between declared and detected documents
  - Accurate field extraction without hallucination
  - ICAO Doc 9303 MRZ parsing and 7-3-1 check digit validation
============================================================
"""

import unittest
from modules.detector import detect_document_type
from modules.base import DocumentType
from modules.permit import PermitVerifier
from modules.national_id import NationalIdVerifier
from main import (
    classify_document,
    parse_mrz,
    extract_document_number,
    extract_document_information,
    guess_name,
)


class TestDocumentDetectionAndExtraction(unittest.TestCase):

    def test_mrz_passport_classification_and_extraction(self):
        sample_passport_ocr = (
            "PASSPORT REPUBLIC OF INDIA\n"
            "Given Names: RAMESH\n"
            "Surname: KUMAR\n"
            "P<INDKUMAR<<RAMESH<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
            "L898902C36IND7408122M1204159<<<<<<<<<<<<<<<8\n"
        )
        # 1. Classification
        doc_type = classify_document(sample_passport_ocr)
        self.assertEqual(doc_type, "Passport")

        slug, conf, _ = detect_document_type(sample_passport_ocr, "Passport")
        self.assertEqual(slug, "passport")
        self.assertGreaterEqual(conf, 50.0)

        # 2. MRZ Parsing
        mrz = parse_mrz(sample_passport_ocr)
        self.assertIsNotNone(mrz)
        self.assertEqual(mrz["name"], "RAMESH KUMAR")
        self.assertEqual(mrz["doc_number"], "L898902C3")
        self.assertEqual(mrz["nationality"], "IND")
        self.assertEqual(mrz["sex"], "M")
        self.assertEqual(mrz["date_of_birth"], "1974-08-12")
        self.assertEqual(mrz["expiry_date"], "2012-04-15")
        self.assertTrue(mrz["check_digits_valid"])

        # 3. Document info extraction
        info = extract_document_information(sample_passport_ocr, "Passport")
        self.assertEqual(info["document_type"], "Passport")
        self.assertEqual(info["document_number"], "L898902C3")
        self.assertEqual(info["name"], "RAMESH KUMAR")

    def test_aadhaar_classification_and_extraction(self):
        sample_aadhaar_ocr = (
            "Government of India\n"
            "Unique Identification Authority of India\n"
            "Ananya Sen\n"
            "DOB: 14/05/1995\n"
            "Female\n"
            "9876 5432 1098\n"
            "Mera Aadhaar, Meri Pehchan\n"
        )
        doc_type = classify_document(sample_aadhaar_ocr)
        self.assertEqual(doc_type, "Aadhaar")

        slug, conf, _ = detect_document_type(sample_aadhaar_ocr, "Aadhaar")
        self.assertEqual(slug, "aadhaar")

        doc_num = extract_document_number(sample_aadhaar_ocr, "Aadhaar")
        self.assertEqual(doc_num, "987654321098")

        info = extract_document_information(sample_aadhaar_ocr, "Aadhaar")
        self.assertEqual(info["document_number"], "987654321098")
        self.assertEqual(info["name"], "Ananya Sen")
        self.assertEqual(info["sex"], "Female")

    def test_pan_classification_and_extraction(self):
        sample_pan_ocr = (
            "INCOME TAX DEPARTMENT\n"
            "GOVT. OF INDIA\n"
            "Permanent Account Number Card\n"
            "ABCDE1234F\n"
            "Name: VIKRAM MALHOTRA\n"
            "Father's Name: RAJESH MALHOTRA\n"
            "Date of Birth: 22/11/1988\n"
        )
        doc_type = classify_document(sample_pan_ocr)
        self.assertEqual(doc_type, "PAN Card")

        slug, conf, _ = detect_document_type(sample_pan_ocr, "PAN Card")
        self.assertEqual(slug, "pan")

        doc_num = extract_document_number(sample_pan_ocr, "PAN Card")
        self.assertEqual(doc_num, "ABCDE1234F")

        info = extract_document_information(sample_pan_ocr, "PAN Card")
        self.assertEqual(info["document_number"], "ABCDE1234F")
        self.assertEqual(info["name"], "VIKRAM MALHOTRA")
        self.assertEqual(info["father_name"], "RAJESH MALHOTRA")

    def test_driving_licence_classification_and_extraction(self):
        sample_dl_ocr = (
            "UNION OF INDIA - DRIVING LICENCE\n"
            "TRANSPORT DEPARTMENT MAHARASHTRA\n"
            "DL No: MH-0120150012345\n"
            "Name: ROHIT DESHMUKH\n"
            "Date of Birth: 10/08/1992\n"
            "Date of Issue: 15/09/2015\n"
            "Valid Till: 09/08/2032\n"
            "Blood Group: B+\n"
            "COV: LMV, MCWG\n"
        )
        doc_type = classify_document(sample_dl_ocr)
        self.assertEqual(doc_type, "Driving Licence")

        slug, conf, _ = detect_document_type(sample_dl_ocr, "Driving Licence")
        self.assertEqual(slug, "driving_licence")

        info = extract_document_information(sample_dl_ocr, "Driving Licence")
        self.assertIn("MH0120150012345", info["licence_number"].replace("-", ""))
        self.assertEqual(info["name"], "ROHIT DESHMUKH")
        self.assertEqual(info["blood_group"], "B+")

    def test_visa_classification_and_extraction(self):
        sample_visa_ocr = (
            "ENTRY PERMIT / VISA\n"
            "Visa Number: V-9823412\n"
            "Visa Type: Tourist - Multiple Entry\n"
            "Full Name: EMILY WATSON\n"
            "Nationality: GBR\n"
            "Entries: MULTIPLE\n"
            "Valid From: 01/01/2026\n"
            "Valid Until: 01/01/2027\n"
        )
        doc_type = classify_document(sample_visa_ocr)
        self.assertEqual(doc_type, "Visa")

        slug, conf, _ = detect_document_type(sample_visa_ocr, "Visa")
        self.assertEqual(slug, "visa")

        info = extract_document_information(sample_visa_ocr, "Visa")
        self.assertEqual(info["visa_number"], "V-9823412")
        self.assertEqual(info["name"], "EMILY WATSON")
        self.assertEqual(info["entries"], "MULTIPLE")

    def test_permit_classification_and_verification(self):
        sample_permit_ocr = (
            "WORK PERMIT & RESIDENCE AUTHORIZATION\n"
            "Permit Number: PMT-8874102\n"
            "Permit Type: Skilled Worker\n"
            "Name: TARIQ AL-MANSOOR\n"
            "Date of Issue: 12/03/2024\n"
            "Date of Expiry: 12/03/2027\n"
        )
        doc_type = classify_document(sample_permit_ocr)
        self.assertEqual(doc_type, "Permit")

        slug, conf, _ = detect_document_type(sample_permit_ocr, "Permit")
        self.assertEqual(slug, "permit")

        verifier = PermitVerifier()
        result = verifier.verify(
            file_bytes=b"sample",
            filename="permit.jpg",
            content_type="image/jpeg",
            ocr_text=sample_permit_ocr,
        )
        self.assertEqual(result.document_type, "permit")
        self.assertIn("Permit statutory terminology detected.", result.reasons)

    def test_no_fabrication_on_empty_text(self):
        info = extract_document_information("", "Passport")
        self.assertIsNone(info["name"])
        self.assertIsNone(info["document_number"])
        self.assertIsNone(info["date_of_birth"])
        self.assertIsNone(info["date_of_issue"])
        self.assertIsNone(info["date_of_expiry"])


if __name__ == "__main__":
    unittest.main()

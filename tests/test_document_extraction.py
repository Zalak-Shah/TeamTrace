"""
============================================================
Comprehensive Document Extraction & MRZ Parsing Test Suite
============================================================
Verifies:
1. ICAO Doc 9303 TD3 Passport MRZ parsing & check digit calculation
2. TD1 ID Card / Border Pass MRZ parsing
3. Field-specific OCR character correction (0 vs O, 1 vs I)
4. Surname and given names extraction from MRZ ('<<' separation)
5. Date separation and plausibility (DOB must never be Issue or Expiry; adult 2-digit years)
6. Gender and Nationality extraction
7. Rejection of invalid MRZ check digits
8. Aadhaar, Driving Licence, and PAN card field extraction
9. MRZ vs Visual zone (VIZ) mismatch detection without destructive overwrites
============================================================
"""

import pytest
from datetime import date
from main import (
    compute_icao_check_digit,
    parse_mrz,
    to_date,
    extract_document_information,
    classify_document,
)


class TestMRZParsing:
    """Test ICAO Doc 9303 standard MRZ parsing and check digit logic."""

    def test_valid_td3_passport_mrz(self):
        """Test standard 2-line TD3 passport MRZ with valid check digits."""
        mrz_text = (
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\n"
            "L898902C36UTO7408122F1204159ZE184226B<<<<<10\n"
        )
        res = parse_mrz(mrz_text)
        assert res is not None
        assert res["format"] == "TD3"
        assert res["name"] == "ANNA MARIA ERIKSSON"
        assert res["surname"] == "ERIKSSON"
        assert res["given_names"] == "ANNA MARIA"
        assert res["doc_number"] == "L898902C3"
        assert res["nationality"] == "UTO"
        assert res["issuing_country"] == "UTO"
        assert res["sex"] == "F"
        assert res["gender"] == "Female"
        assert res["date_of_birth"] == "1974-08-12"
        assert res["expiry_date"] == "2012-04-15"
        assert res["check_digits_valid"] is True
        assert res["mrz_checks"]["document_number"] is True
        assert res["mrz_checks"]["date_of_birth"] is True
        assert res["mrz_checks"]["expiry_date"] is True

    def test_mrz_with_ocr_typos_in_digits(self):
        """Test OCR mistakes like letter 'O' instead of '0', 'I' instead of '1' in numeric fields."""
        # 'O' in DOB '74O812', 'I' in check digit, etc.
        mrz_text = (
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\n"
            "L898902C36UTO74O8122F12O4159ZE184226B<<<<<1O\n"
        )
        res = parse_mrz(mrz_text)
        assert res is not None
        assert res["date_of_birth"] == "1974-08-12"
        assert res["expiry_date"] == "2012-04-15"

    def test_invalid_mrz_check_digits(self):
        """Test invalid check digit detection."""
        # Deliberately modify the doc_number check digit from 6 to 9
        mrz_text = (
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\n"
            "L898902C39UTO7408122F1204159ZE184226B<<<<<10\n"
        )
        res = parse_mrz(mrz_text)
        assert res is not None
        assert res["check_digits_valid"] is False
        assert res["mrz_checks"]["document_number"] is False

    def test_td1_3line_id_card_mrz(self):
        """Test 3-line TD1 ID Card / Border pass MRZ."""
        td1_text = (
            "I<UTOD231458907<<<<<<<<<<<<<<<\n"
            "7408122F1204159UTO<<<<<<<<<<<6\n"
            "ERIKSSON<<ANNA<MARIA<<<<<<<<<<\n"
        )
        res = parse_mrz(td1_text)
        assert res is not None
        assert res["format"] == "TD1"
        assert res["surname"] == "ERIKSSON"
        assert res["given_names"] == "ANNA MARIA"
        assert res["doc_number"] == "D23145890"
        assert res["gender"] == "Female"
        assert res["date_of_birth"] == "1974-08-12"
        assert res["expiry_date"] == "2012-04-15"


class TestDateExtractionAndPlausibility:
    """Test date parsing, 2-digit years, and separation of DOB, DOI, and DOE."""

    def test_two_digit_year_plausibility(self):
        """DOB 2-digit year should be past; Expiry 2-digit year should be future."""
        # 14/05/92 -> Born in 1992
        d_dob = to_date("14/05/92", default_future=False)
        assert d_dob == date(1992, 5, 14)
        assert d_dob <= date.today()

        # 14/05/32 -> Expires in 2032
        d_exp = to_date("14/05/32", default_future=True)
        assert d_exp == date(2032, 5, 14)

    def test_international_and_bilingual_dates(self):
        """Verify support for bilingual and textual international date formats."""
        d1 = to_date("23 APR/AVR 1984")
        assert d1 == date(1984, 4, 23)

        d2 = to_date("04 Aug 1995")
        assert d2 == date(1995, 8, 4)

        d3 = to_date("August 04, 1995")
        assert d3 == date(1995, 8, 4)

    def test_dob_never_assigned_to_issue_or_expiry(self):
        """Verify that extract_document_information strictly isolates DOB from Issue and Expiry."""
        ocr_text = (
            "PASSPORT\n"
            "UNITED STATES OF AMERICA\n"
            "Surname: SMITH\n"
            "Given Names: JOHN DAVID\n"
            "Date of Birth: 15 JAN 1985\n"
            "Date of Issue: 20 MAR 2018\n"
            "Date of Expiry: 19 MAR 2028\n"
            "Sex: M\n"
            "Passport No: C12345678\n"
        )
        info = extract_document_information(ocr_text, "Passport")
        assert info["surname"] == "SMITH"
        assert info["given_names"] == "JOHN DAVID"
        assert info["name"] == "JOHN DAVID SMITH"
        assert info["date_of_birth"] == "15/01/1985"
        assert info["date_of_issue"] == "20/03/2018"
        assert info["date_of_expiry"] == "19/03/2028"
        assert info["date_of_birth"] != info["date_of_issue"]
        assert info["date_of_birth"] != info["date_of_expiry"]
        assert info["sex"] == "Male"
        assert info["document_number"] == "C12345678"


class TestMultiDocumentExtraction:
    """Verify field mapping across diverse supported document types."""

    def test_aadhaar_extraction(self):
        ocr_text = (
            "Government of India\n"
            "Unique Identification Authority of India\n"
            "Rahul Kumar Sharma\n"
            "DOB: 12/08/1990\n"
            "Male\n"
            "9876 5432 1098\n"
            "Address: 42 MG Road, Bengaluru, Karnataka 560001\n"
        )
        info = extract_document_information(ocr_text, "Aadhaar")
        assert info["document_number"] == "987654321098"
        assert info["name"] == "Rahul Kumar Sharma"
        assert info["date_of_birth"] == "12/08/1990"
        assert info["sex"] == "Male"
        assert "Bengaluru" in (info["address"] or "")

    def test_driving_licence_extraction(self):
        ocr_text = (
            "UNION OF INDIA DRIVING LICENCE\n"
            "DL No: DL-0420110149646\n"
            "Name: ARJUN VERMA\n"
            "DOB: 05/11/1988\n"
            "Issue Date: 14/02/2011\n"
            "Valid Till (NT): 04/11/2038\n"
            "Blood Group: B+\n"
            "Class of Vehicle: LMV, MCWG\n"
        )
        info = extract_document_information(ocr_text, "Driving Licence")
        assert info["licence_number"] == "DL0420110149646"
        assert info["name"] == "ARJUN VERMA"
        assert info["date_of_birth"] == "05/11/1988"
        assert info["date_of_issue"] == "14/02/2011"
        assert info["date_of_expiry"] == "04/11/2038"
        assert info["blood_group"] == "B+"
        assert info["vehicle_class"] == "LMV, MCWG"

    def test_pan_card_extraction(self):
        ocr_text = (
            "INCOME TAX DEPARTMENT\n"
            "GOVT OF INDIA\n"
            "Permanent Account Number Card\n"
            "ABCDE1234F\n"
            "Name: PRIYA PATEL\n"
            "Father's Name: SURESH PATEL\n"
            "Date of Birth: 22/07/1995\n"
        )
        info = extract_document_information(ocr_text, "PAN Card")
        assert info["document_number"] == "ABCDE1234F"
        assert info["name"] == "PRIYA PATEL"
        assert info["father_name"] == "SURESH PATEL"
        assert info["date_of_birth"] == "22/07/1995"

    def test_mrz_and_viz_cross_check_mismatch(self):
        """If MRZ and printed OCR document numbers mismatch, report mismatch without overwriting."""
        ocr_text = (
            "PASSPORT\n"
            "Passport No: Z9999999\n"
            "Surname: ERIKSSON\n"
            "Given Names: ANNA\n"
            "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\n"
            "L898902C36UTO7408122F1204159ZE184226B<<<<<10\n"
        )
        info = extract_document_information(ocr_text, "Passport")
        # MRZ doc is L898902C3, VIZ had Z9999999
        assert len(info["mrz_mismatches"]) >= 1
        mismatch = info["mrz_mismatches"][0]
        assert mismatch["field"] == "document_number"
        assert mismatch["mrz_value"] == "L898902C3"
        assert mismatch["viz_value"] == "Z9999999"

    def test_multilingual_us_passport_screenshot_regression(self):
        """Regression test for passport2 screenshot with trilingual headers."""
        ocr_text = (
            "UNITED STATES OF AMERICA\n"
            "PASSPORT PASSEPORT PASAPORTE\n"
            "Type/Type/Tipo Code/Code/Codigo Passport No./No. du Passeport/No. de Pasaporte\n"
            "P USA 910239248\n"
            "Surname/Nom / Apellidos\n"
            "DE LA PAZ\n"
            "Given Names/Prenoms/Nombres\n"
            "MICHELLE\n"
            "Nationality/Nationalite/Nacionalidad\n"
            "UNITED STATES OF AMERICA\n"
            "Date of birth/Date de naissance / Fecha de nacimiento\n"
            "07 Aug 1999\n"
            "Place of birth/Lieu de naissance/Lugar de nacimiento Sex/Sexe/Sexo\n"
            "CALIFORNIA, U.S.A. F\n"
            "Date of issue/Date de delivrance / Fecha de expedicion Authority/Autorite/Autoridad\n"
            "06 Feb 2008 United States\n"
            "Date of expiration/Date d'expiration / Fecha de caducidad Department of State\n"
            "05 Feb 2018\n"
            "SEE PAGE 27 USA\n"
            "P<DELAPAZ<<MICHELLE<<<<<<<<<<<<<<<<<<<<<<<<<\n"
            "9102392482USA6401171F1812051900781200<129676\n"
        )
        assert classify_document(ocr_text) == "Passport"
        info = extract_document_information(ocr_text, "Passport")
        assert info["surname"] == "DE LA PAZ"
        assert info["given_names"] == "MICHELLE"
        assert info["name"] == "MICHELLE DE LA PAZ"
        assert info["document_number"] == "910239248"
        assert info["nationality"] == "USA"
        assert info["sex"] == "Female"
        assert info["place_of_birth"] == "CALIFORNIA, U.S.A."
        assert info["issuing_country"] in ("USA", "United States of America")
        assert info["date_of_birth"] == "07/08/1999"
        assert info["date_of_issue"] == "06/02/2008"

    def test_mrv_a_visa_screenshot_regression(self):
        """Regression test for visa.jpg screenshot with grid fields and MRV-A."""
        ocr_text = (
            "UNITED STATES OF AMERICA\n"
            "VISA\n"
            "Issuing Post Name\n"
            "FRANKFURT Control Number\n"
            "Surname 20241234560001\n"
            "SMITH\n"
            "Given Name Visa Type/Class\n"
            "JOE R B1\n"
            "Passport Number Sex Birth Date Nationality\n"
            "CZ6311T47 M 21MAR1996 GER\n"
            "Entries Issue Date Expiration Date\n"
            "M 14FEB2024 13FEB2034 1010\n"
            "Annotation\n"
            "U1234567\n"
            "VNUSA SMITH<<JOE<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
            "CZ6311T47<9BER15001013F1304071B3FRN00AB4243934\n"
        )
        assert classify_document(ocr_text) == "Visa"
        info = extract_document_information(ocr_text, "Visa")
        assert info["surname"] == "SMITH"
        assert info["given_names"] == "JOE"
        assert info["name"] == "JOE SMITH"
        assert info["document_number"] == "CZ6311T47"
        assert info["nationality"] == "GER"
        assert info["sex"] == "Male"
        assert info["visa_number"] == "U1234567"
        assert info["entries"] == "M"
        assert info["visa_type"] == "R B1"
        assert info["date_of_birth"] == "21/03/1996"
        assert info["date_of_issue"] == "14/02/2024"
        assert info["date_of_expiry"] == "13/02/2034"

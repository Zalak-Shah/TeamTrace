import unittest
from modules.passport import (
    compute_icao_check_digit,
    parse_and_validate_td3_mrz,
    PassportVerifier,
)


class TestPassportModule(unittest.TestCase):
    def test_icao_check_digit_calculation(self):
        # Known test vector: "L898902C3" with check digit 6
        # Data: L(21)*7 + 8*3 + 9*1 + 8*7 + 9*3 + 0*1 + 2*7 + C(12)*3 + 3*1 = 147 + 24 + 9 + 56 + 27 + 0 + 14 + 36 + 3 = 316 % 10 = 6
        computed = compute_icao_check_digit("L898902C3")
        self.assertEqual(computed, 6)

        # Date of birth 740812 with check digit 2
        # 7*7 + 4*3 + 0*1 + 8*7 + 1*3 + 2*1 = 49 + 12 + 0 + 56 + 3 + 2 = 122 % 10 = 2
        self.assertEqual(compute_icao_check_digit("740812"), 2)

    def test_passport_mrz_parsing_and_checks(self):
        # Specimen Indian Passport MRZ TD3
        # Line 1: P<INDKUMAR<<RAMESH<<<<<<<<<<<<<<<<<<<<<<<<<<
        # Line 2: L898902C36IND7408122M1204159<<<<<<<<<<<<<<<4
        mrz_text = (
            "PASSPORT REPUBLIC OF INDIA\n"
            "P<INDKUMAR<<RAMESH<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
            "L898902C36IND7408122M1204159<<<<<<<<<<<<<<<4\n"
        )
        data, issues, verified = parse_and_validate_td3_mrz(mrz_text)
        self.assertIsNotNone(data)
        self.assertEqual(data["document_number"], "L898902C3")
        self.assertEqual(data["issuing_country"], "IND")
        self.assertEqual(data["name"], "RAMESH KUMAR")
        # Check that check digit verified correctly
        self.assertTrue(any("Passport number check digit valid" in v for v in verified))

    def test_passport_tampered_check_digit_flagged(self):
        # Tampered line 2 with invalid check digit (changed '6' to '9')
        tampered_mrz_text = (
            "P<INDKUMAR<<RAMESH<<<<<<<<<<<<<<<<<<<<<<<<<<\n"
            "L898902C39IND7408122M1204159<<<<<<<<<<<<<<<4\n"
        )
        verifier = PassportVerifier()
        result = verifier.verify(b"", "doc.jpg", "image/jpeg", tampered_mrz_text, {})
        self.assertEqual(result.status, "suspicious")
        self.assertTrue(any("failed" in r for r in result.reasons))
        self.assertEqual(result.verification_tier, 3)
        self.assertTrue(len(result.limitations) > 0)


if __name__ == "__main__":
    unittest.main()

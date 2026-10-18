import unittest
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from modules.aadhaar import (
    AadhaarVerifier,
    mask_aadhaar_number,
    parse_decompressed_qr,
)


class TestAadhaarModule(unittest.TestCase):
    def test_aadhaar_masking(self):
        # 12 digits
        self.assertEqual(mask_aadhaar_number("1234 5678 9012"), "XXXX-XXXX-9012")
        self.assertEqual(mask_aadhaar_number("123456789012"), "XXXX-XXXX-9012")
        self.assertEqual(mask_aadhaar_number(""), "XXXX-XXXX-XXXX")
        self.assertEqual(mask_aadhaar_number(None), "XXXX-XXXX-XXXX")

    def test_parse_decompressed_qr(self):
        # Delimited format: flag\xffname\xffdob\xffgender\xffcare_of
        payload = b"Y\xffRamesh Kumar\xff1990-05-12\xffM\xffS/O Suresh Kumar\xffDelhi\xff\xff\xff\xff110001"
        fields = parse_decompressed_qr(payload)
        self.assertEqual(fields.get("name"), "Ramesh Kumar")
        self.assertEqual(fields.get("dob"), "1990-05-12")
        self.assertEqual(fields.get("gender"), "M")
        self.assertEqual(fields.get("pincode"), "110001")

    def test_aadhaar_qr_signature_validation_flow(self):
        verifier = AadhaarVerifier()

        # Generate test RSA key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        data = b"sample_uidai_qr_decompressed_data"
        valid_signature = private_key.sign(
            data,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        self.assertEqual(len(valid_signature), 256)

        # Corrupt the signature by flipping bytes
        corrupted_signature = bytearray(valid_signature)
        corrupted_signature[0] ^= 0xFF
        corrupted_signature = bytes(corrupted_signature)

        # With corrupted signature, verification must fail
        qr_blob = data + corrupted_signature
        qr_res = verifier._verify_secure_qr(qr_blob, "Ramesh Kumar 1990-05-12")
        self.assertFalse(qr_res["signature_valid"])


if __name__ == "__main__":
    unittest.main()

import unittest
import hashlib
from database import compute_block_hash, GENESIS_HASH


class TestAuditHashChain(unittest.TestCase):
    def test_compute_block_hash_deterministic(self):
        h1 = compute_block_hash(
            GENESIS_HASH, "Officer Sharma", "Screening completed", "All valid", "2026-09-05T12:00:00", 101
        )
        h2 = compute_block_hash(
            GENESIS_HASH, "Officer Sharma", "Screening completed", "All valid", "2026-09-05T12:00:00", 101
        )
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)

    def test_tamper_detection_in_chain(self):
        # Build 3 blocks
        prev = GENESIS_HASH
        b1_hash = compute_block_hash(prev, "Officer 1", "Screening 1", "OK", "2026-09-05T10:00:00", 1)
        b2_hash = compute_block_hash(b1_hash, "Officer 2", "Screening 2", "Flagged", "2026-09-05T10:05:00", 2)
        b3_hash = compute_block_hash(b2_hash, "Officer 3", "Screening 3", "Cleared", "2026-09-05T10:10:00", 3)

        chain = [
            {"id": 1, "prev_hash": prev, "hash": b1_hash, "officer_name": "Officer 1", "action": "Screening 1", "details": "OK", "created_at": "2026-09-05T10:00:00", "screening_id": 1},
            {"id": 2, "prev_hash": b1_hash, "hash": b2_hash, "officer_name": "Officer 2", "action": "Screening 2", "details": "Flagged", "created_at": "2026-09-05T10:05:00", "screening_id": 2},
            {"id": 3, "prev_hash": b2_hash, "hash": b3_hash, "officer_name": "Officer 3", "action": "Screening 3", "details": "Cleared", "created_at": "2026-09-05T10:10:00", "screening_id": 3},
        ]

        # Verify untouched chain
        expected = GENESIS_HASH
        broken = []
        for row in chain:
            rec = compute_block_hash(expected, row["officer_name"], row["action"], row["details"], row["created_at"], row["screening_id"])
            if row["prev_hash"] != expected or row["hash"] != rec:
                broken.append(row["id"])
            expected = row["hash"]
        self.assertEqual(len(broken), 0)

        # Tamper with block 2 action/decision details
        chain[1]["action"] = "Tampered Action"
        expected = GENESIS_HASH
        first_broken = None
        for row in chain:
            rec = compute_block_hash(expected, row["officer_name"], row["action"], row["details"], row["created_at"], row["screening_id"])
            if row["prev_hash"] != expected or row["hash"] != rec:
                if first_broken is None:
                    first_broken = row["id"]
            expected = row["hash"]

        self.assertEqual(first_broken, 2, "Tampered block 2 should be caught as the first broken block")


if __name__ == "__main__":
    unittest.main()

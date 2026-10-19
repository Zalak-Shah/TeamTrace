"""
============================================================
Test Suite: OpenCV Forensics Checks
============================================================
Tests:
1. Clean baseline image (uniform document scan)
2. Pasted patch image (synthetic spliced forgery for ELA)
3. Copied region image (synthetic duplicated region for copy-move)
Asserts that tampered versions score higher than clean ones.
"""

import cv2
import numpy as np
import pytest
from modules.opencv_forensics import (
    run_ela_check,
    run_copy_move_check,
    run_qr_check,
    run_face_match_check,
    rectify_document,
)


def create_clean_document_image() -> bytes:
    """Creates a synthetic uniform document image saved as JPEG at quality 95."""
    img = np.full((500, 700, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (680, 480), (60, 60, 60), 2)
    cv2.putText(img, "PASSPORT REPUBLIC OF BORDERSHIELD", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(img, "GIVEN NAMES: ALEXIS JORDAN", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    cv2.putText(img, "DOCUMENT NO: P9842103", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    cv2.putText(img, "DATE OF BIRTH: 14 MAY 1988", (40, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    cv2.putText(img, "NATIONALITY: IND  SEX: FEMALE", (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)
    cv2.putText(img, "DATE OF EXPIRY: 28 OCT 2032", (40, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)

    # Unique complex emblem in clean document (not duplicated)
    cv2.circle(img, (550, 150), 40, (10, 20, 80), 2)
    for i in range(8):
        ang = i * (np.pi / 4)
        cv2.line(img, (550, 150), (int(550 + 35 * np.cos(ang)), int(150 + 35 * np.sin(ang))), (20, 60, 150), 2)

    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return enc.tobytes()


def create_pasted_patch_image() -> bytes:
    """
    Creates a document image with an alien, re-compressed / foreign patch pasted onto it.
    This introduces high localized compression discrepancy for ELA.
    """
    clean_bytes = create_clean_document_image()
    arr = np.frombuffer(clean_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # Create a patch compressed at a wildly different quality level (e.g. Q30 with strong noise)
    patch = np.zeros((100, 150, 3), dtype=np.uint8)
    noise = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    patch = cv2.addWeighted(patch, 0.2, noise, 0.8, 0)
    cv2.putText(patch, "FORGED", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    _, patch_enc = cv2.imencode(".jpg", patch, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    degraded_patch = cv2.imdecode(patch_enc, cv2.IMREAD_COLOR)

    # Paste the degraded patch into the clean document
    img[120:220, 300:450] = degraded_patch

    _, final_enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return final_enc.tobytes()


def create_copy_move_image() -> bytes:
    """
    Creates a document image where a detailed texture region is copied and pasted elsewhere
    to trigger SIFT/ORB duplicated feature detection.
    """
    clean_bytes = create_clean_document_image()
    arr = np.frombuffer(clean_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    # Copy the emblem from region (550, 150) and duplicate it to (250, 360)
    # The source region is [100:200, 500:600]
    region_seal = img[100:200, 500:600].copy()
    img[320:420, 200:300] = region_seal

    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return enc.tobytes()


def test_ela_score_clean_vs_tampered():
    """Verify that a spliced/pasted image yields a higher ELA anomaly score than a clean image."""
    clean_bytes = create_clean_document_image()
    tampered_bytes = create_pasted_patch_image()

    res_clean = run_ela_check(clean_bytes)
    res_tampered = run_ela_check(tampered_bytes)

    assert res_clean["status"] == "completed"
    assert res_tampered["status"] == "completed"

    # Tampered image must score higher on localized compression anomaly
    assert res_tampered["score"] > res_clean["score"], (
        f"Expected tampered score ({res_tampered['score']}) > clean score ({res_clean['score']})"
    )
    # The localized region ratio or block variance must reflect the tampering
    assert (
        res_tampered["details"]["block_variance"] > res_clean["details"]["block_variance"]
        or res_tampered["details"]["localized_region_ratio"] > res_clean["details"]["localized_region_ratio"]
    )


def test_copy_move_clean_vs_cloned():
    """Verify that duplicated copied regions yield a higher copy-move score than a clean image."""
    clean_bytes = create_clean_document_image()
    cloned_bytes = create_copy_move_image()

    res_clean = run_copy_move_check(clean_bytes)
    res_cloned = run_copy_move_check(cloned_bytes)

    assert res_clean["status"] == "completed"
    assert res_cloned["status"] == "completed"

    # Cloned image must have more duplicated RANSAC inliers and higher score
    assert res_cloned["score"] >= res_clean["score"]
    assert res_cloned["details"]["ransac_inliers"] > res_clean["details"]["ransac_inliers"], (
        f"Expected cloned inliers ({res_cloned['details']['ransac_inliers']}) > clean inliers ({res_clean['details']['ransac_inliers']})"
    )


def test_qr_check():
    """Verify QR detection handles images gracefully."""
    clean_bytes = create_clean_document_image()
    res = run_qr_check(clean_bytes)
    assert res["status"] == "completed"
    assert res["details"]["qr_found"] is False


def test_face_match_graceful_skip():
    """Verify face match returns skipped when no selfie is passed."""
    clean_bytes = create_clean_document_image()
    res = run_face_match_check(clean_bytes, None)
    assert res["status"] == "skipped"
    assert res["score"] == 0.0


def test_rectification_fallback():
    """Verify rectification returns original image and completes cleanly when no quad is detected."""
    clean_bytes = create_clean_document_image()
    rectified, res = rectify_document(clean_bytes)
    assert res["status"] == "completed"
    assert isinstance(rectified, np.ndarray)

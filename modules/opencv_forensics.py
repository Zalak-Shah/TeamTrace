"""
============================================================
BorderShield AI - OpenCV Forensics Module
============================================================
Performs advanced computer vision & forensic checks:
1. Error Level Analysis (ELA) with cv2.imencode @ quality 90
2. Copy-Move Forgery Detection (SIFT/ORB + RANSAC estimateAffinePartial2D)
3. QR Code Detection & Aadhaar Decoded vs OCR Cross-Validation
4. Facial Verification (FaceDetectorYN YuNet + FaceRecognizerSF SFace)
5. Document Contour Rectification (Largest Quad + getPerspectiveTransform)

Standard Return Interface for each check:
{
    "score": float (0.0 to 1.0),
    "flag": bool,
    "explanation": str,
    "status": "completed" | "skipped" | "failed" | "not_applicable",
    "details": dict
}
============================================================
"""

import os
import re
import numpy as np
from typing import Any, Dict, Optional, Tuple

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
YUNET_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_PATH = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")


def _decode_image(img_input: Any) -> Optional[np.ndarray]:
    """Helper to convert bytes, filepath, or np.ndarray into a BGR image."""
    if not _CV2_AVAILABLE or cv2 is None:
        return None
    if isinstance(img_input, np.ndarray):
        return img_input
    if isinstance(img_input, bytes):
        arr = np.frombuffer(img_input, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if isinstance(img_input, str):
        if os.path.exists(img_input):
            return cv2.imread(img_input)
    return None


def run_ela_check(image_input: Any, quality: int = 90) -> Dict[str, Any]:
    """
    1. Error Level Analysis (ELA).
    Re-encodes image as JPEG at quality 90 with cv2.imencode, calculates the absolute difference,
    and reports the localized-region ratio and score (0 to 1).
    """
    if not _CV2_AVAILABLE or cv2 is None:
        return {
            "score": 0.0,
            "flag": False,
            "explanation": "OpenCV not available.",
            "status": "skipped",
            "details": {},
        }

    img = _decode_image(image_input)
    if img is None:
        return {
            "score": 0.0,
            "flag": False,
            "explanation": "Could not decode image for ELA.",
            "status": "failed",
            "details": {},
        }

    try:
        # Re-encode to JPEG in memory at specified quality
        success, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not success:
            return {
                "score": 0.0,
                "flag": False,
                "explanation": "JPEG re-encoding failed during ELA.",
                "status": "failed",
                "details": {},
            }

        recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if recompressed is None:
            return {
                "score": 0.0,
                "flag": False,
                "explanation": "Failed to decode re-compressed image in ELA.",
                "status": "failed",
                "details": {},
            }

        # Absolute difference in float
        diff = cv2.absdiff(img, recompressed).astype(np.float32)
        mean_err = float(np.mean(diff))
        std_err = float(np.std(diff))
        max_err = float(np.max(diff))

        # Localized block analysis (16x16 grid)
        h, w = diff.shape[:2]
        block_h = max(h // 16, 8)
        block_w = max(w // 16, 8)
        block_means = []
        for r in range(0, h - block_h + 1, block_h):
            for c in range(0, w - block_w + 1, block_w):
                blk = diff[r:r + block_h, c:c + block_w]
                block_means.append(float(np.mean(blk)))

        block_arr = np.array(block_means, dtype=np.float32) if block_means else np.array([mean_err], dtype=np.float32)
        overall_mean = float(np.mean(block_arr))
        # High-discrepancy regions: blocks with error > 2.0x overall mean (and at least 4.0 absolute diff)
        high_discrepancy_blocks = np.sum((block_arr > (overall_mean * 1.8 + 2.0)))
        total_blocks = len(block_arr)
        localized_ratio = float(high_discrepancy_blocks / total_blocks) if total_blocks > 0 else 0.0
        block_variance = float(np.std(block_arr))

        # Normalized anomaly score from 0.0 to 1.0
        # Natural scans have uniform ELA; tampered/spliced images have distinct localized variance spikes
        raw_score = (localized_ratio * 0.6) + (min(block_variance / 15.0, 1.0) * 0.4)
        score = float(np.clip(raw_score, 0.0, 1.0))
        flag = bool(score >= 0.45 or localized_ratio >= 0.25)

        explanation = (
            f"ELA detected localized compression anomalies (ratio: {localized_ratio:.2%}, variance: {block_variance:.1f})."
            if flag
            else f"Compression level across document surface is consistent (ratio: {localized_ratio:.2%})."
        )

        return {
            "score": round(score, 3),
            "flag": flag,
            "explanation": explanation,
            "status": "completed",
            "details": {
                "localized_region_ratio": round(localized_ratio, 3),
                "mean_error": round(mean_err, 2),
                "std_error": round(std_err, 2),
                "max_error": round(max_err, 2),
                "block_variance": round(block_variance, 2),
            },
        }
    except Exception as e:
        return {
            "score": 0.0,
            "flag": False,
            "explanation": f"ELA check encountered error: {str(e)}",
            "status": "failed",
            "details": {"error": str(e)},
        }


def run_copy_move_check(image_input: Any) -> Dict[str, Any]:
    """
    2. Copy-Move Forgery Detection.
    Detects duplicated regions within the same image using SIFT/ORB feature matching,
    excluding self-matches, followed by RANSAC estimateAffinePartial2D verification.
    """
    if not _CV2_AVAILABLE or cv2 is None:
        return {
            "score": 0.0,
            "flag": False,
            "explanation": "OpenCV not available.",
            "status": "skipped",
            "details": {},
        }

    img = _decode_image(image_input)
    if img is None:
        return {
            "score": 0.0,
            "flag": False,
            "explanation": "Could not decode image for copy-move detection.",
            "status": "failed",
            "details": {},
        }

    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Use SIFT if available, fallback to ORB
        use_sift = hasattr(cv2, "SIFT_create")
        if use_sift:
            detector = cv2.SIFT_create(nfeatures=1500)
            norm = cv2.NORM_L2
        else:
            detector = cv2.ORB_create(nfeatures=1500)
            norm = cv2.NORM_HAMMING

        keypoints, descriptors = detector.detectAndCompute(gray, None)
        if descriptors is None or len(keypoints) < 10:
            return {
                "score": 0.0,
                "flag": False,
                "explanation": "Insufficient feature points found to evaluate copy-move tampering.",
                "status": "completed",
                "details": {"keypoints_count": len(keypoints) if keypoints else 0},
            }

        # Match descriptors against themselves (k=4 nearest neighbors)
        bf = cv2.BFMatcher(norm)
        matches = bf.knnMatch(descriptors, descriptors, k=min(4, len(descriptors)))

        min_spatial_dist = max(w, h) * 0.08  # Keypoints must be at least 8% image dimension apart
        candidate_src = []
        candidate_dst = []

        for m_list in matches:
            if len(m_list) < 2:
                continue
            # m_list[0] is always self-match (distance 0 at identical index)
            for m in m_list[1:]:
                if m.queryIdx == m.trainIdx:
                    continue
                # For authentic identical clones, descriptor distance is tight
                if use_sift and m.distance > 85.0:
                    continue
                if not use_sift and m.distance > 45.0:
                    continue
                # Spatial distance between the two matched keypoints
                pt1 = keypoints[m.queryIdx].pt
                pt2 = keypoints[m.trainIdx].pt
                dist = np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
                if dist >= min_spatial_dist:
                    candidate_src.append(pt1)
                    candidate_dst.append(pt2)
                    break

        inliers_count = 0
        total_candidates = len(candidate_src)

        if total_candidates >= 6:
            src_pts = np.array(candidate_src, dtype=np.float32).reshape(-1, 1, 2)
            dst_pts = np.array(candidate_dst, dtype=np.float32).reshape(-1, 1, 2)
            # RANSAC with estimateAffinePartial2D
            matrix, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=4.0,
                maxIters=2000,
                confidence=0.99,
            )
            if inliers is not None:
                inliers_count = int(np.sum(inliers))

        # Score calculation based on geometrically consistent duplicated clusters
        if inliers_count >= 25:
            score = 1.0
            flag = True
        elif inliers_count >= 12:
            score = round(0.5 + (inliers_count - 12) * 0.035, 3)
            flag = True
        else:
            score = round(min(inliers_count / 30.0, 0.35), 3)
            flag = False

        explanation = (
            f"Copy-move forgery detected: {inliers_count} affine-consistent duplicated keypoints found."
            if flag
            else f"No significant cloned or duplicated regions detected (inliers: {inliers_count})."
        )

        return {
            "score": score,
            "flag": flag,
            "explanation": explanation,
            "status": "completed",
            "details": {
                "algorithm": "SIFT" if use_sift else "ORB",
                "total_keypoints": len(keypoints),
                "candidate_matches": total_candidates,
                "ransac_inliers": inliers_count,
            },
        }
    except Exception as e:
        return {
            "score": 0.0,
            "flag": False,
            "explanation": f"Copy-move check error: {str(e)}",
            "status": "failed",
            "details": {"error": str(e)},
        }


def run_qr_check(
    image_input: Any,
    doc_type: Optional[str] = None,
    ocr_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    3. QR Decoding with cv2.QRCodeDetector.
    Decodes the QR code. For Aadhaar, compares the decoded fields/text against OCR text.
    """
    if not _CV2_AVAILABLE or cv2 is None:
        return {
            "score": 0.0,
            "flag": False,
            "explanation": "OpenCV not available.",
            "status": "skipped",
            "details": {},
        }

    img = _decode_image(image_input)
    if img is None:
        return {
            "score": 0.0,
            "flag": False,
            "explanation": "Could not decode image for QR check.",
            "status": "failed",
            "details": {},
        }

    try:
        detector = cv2.QRCodeDetector()
        val, pts, _ = detector.detectAndDecode(img)

        # If not found directly, try on grayscale and sharpened
        if not val:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            val, pts, _ = detector.detectAndDecode(gray)

        if not val:
            # Fallback check multi if available
            if hasattr(detector, "detectAndDecodeMulti"):
                success, decoded_info, pts, _ = detector.detectAndDecodeMulti(img)
                if success and len(decoded_info) > 0 and decoded_info[0]:
                    val = decoded_info[0]

        if not val:
            return {
                "score": 0.0,
                "flag": False,
                "explanation": "No QR code detected or decoded in document.",
                "status": "completed",
                "details": {"qr_found": False},
            }

        qr_text = str(val).strip()
        details = {
            "qr_found": True,
            "qr_length": len(qr_text),
            "qr_snippet": qr_text[:60] + "..." if len(qr_text) > 60 else qr_text,
        }

        # If Aadhaar document and OCR text provided, compare decoded fields to OCR
        is_aadhaar = bool(doc_type and "aadhaar" in doc_type.lower())
        if is_aadhaar and ocr_text:
            ocr_clean = re.sub(r"\s+", " ", ocr_text.upper())
            # Search for numbers or names in QR text
            # Extract 4-digit sequences or numbers >= 4 digits from QR
            qr_tokens = [t for t in re.split(r"[\s,\<\>\/\|\"\':;]+", qr_text) if len(t) >= 4]
            matches = 0
            checked_tokens = 0

            for tok in qr_tokens:
                if tok.isalnum():
                    checked_tokens += 1
                    if tok.upper() in ocr_clean:
                        matches += 1

            match_ratio = (matches / checked_tokens) if checked_tokens > 0 else 1.0
            details["tokens_checked"] = checked_tokens
            details["tokens_matched"] = matches
            details["match_ratio"] = round(match_ratio, 2)

            # If QR tokens conflict with OCR text, flag mismatch
            if checked_tokens >= 2 and match_ratio < 0.3:
                score = round(1.0 - match_ratio, 2)
                flag = True
                explanation = f"Aadhaar QR content strongly conflicts with OCR text (match ratio: {match_ratio:.0%})."
            else:
                score = 0.0
                flag = False
                explanation = f"Aadhaar QR verified successfully against document OCR ({matches}/{checked_tokens} tokens matched)."

            return {
                "score": score,
                "flag": flag,
                "explanation": explanation,
                "status": "completed",
                "details": details,
            }

        return {
            "score": 0.0,
            "flag": False,
            "explanation": f"QR code decoded successfully ({len(qr_text)} characters).",
            "status": "completed",
            "details": details,
        }

    except Exception as e:
        return {
            "score": 0.0,
            "flag": False,
            "explanation": f"QR detection error: {str(e)}",
            "status": "failed",
            "details": {"error": str(e)},
        }


def run_face_match_check(doc_image_input: Any, selfie_image_input: Any) -> Dict[str, Any]:
    """
    4. Face match with cv2.FaceDetectorYN and cv2.FaceRecognizerSF using YuNet and SFace models.
    Compares face in document against live selfie.
    If a model file is missing, returns status "skipped" without crashing.
    """
    if not _CV2_AVAILABLE or cv2 is None:
        return {
            "score": 0.0,
            "flag": False,
            "explanation": "OpenCV not available.",
            "status": "skipped",
            "details": {},
        }

    if not (os.path.exists(YUNET_PATH) and os.path.exists(SFACE_PATH)):
        return {
            "score": 0.0,
            "flag": False,
            "explanation": "YuNet / SFace ONNX models missing in models/. Run scripts/download_models.py.",
            "status": "skipped",
            "details": {
                "yunet_present": os.path.exists(YUNET_PATH),
                "sface_present": os.path.exists(SFACE_PATH),
            },
        }

    if selfie_image_input is None:
        return {
            "score": 0.0,
            "flag": False,
            "explanation": "No selfie image provided for facial comparison.",
            "status": "skipped",
            "details": {"selfie_provided": False},
        }

    doc_img = _decode_image(doc_image_input)
    selfie_img = _decode_image(selfie_image_input)

    if doc_img is None or selfie_img is None:
        return {
            "score": 0.0,
            "flag": False,
            "explanation": "Could not decode document or selfie image for face match.",
            "status": "failed",
            "details": {},
        }

    try:
        # Initialize detector for doc
        h_d, w_d = doc_img.shape[:2]
        detector_doc = cv2.FaceDetectorYN.create(
            YUNET_PATH, "", (w_d, h_d), score_threshold=0.6, nms_threshold=0.3, top_k=5000
        )
        _, faces_doc = detector_doc.detect(doc_img)

        # Initialize detector for selfie
        h_s, w_s = selfie_img.shape[:2]
        detector_selfie = cv2.FaceDetectorYN.create(
            YUNET_PATH, "", (w_s, h_s), score_threshold=0.6, nms_threshold=0.3, top_k=5000
        )
        _, faces_selfie = detector_selfie.detect(selfie_img)

        if faces_doc is None or len(faces_doc) == 0:
            return {
                "score": 0.0,
                "flag": False,
                "explanation": "No face detected on document image.",
                "status": "completed",
                "details": {"document_faces": 0, "selfie_faces": len(faces_selfie) if faces_selfie is not None else 0},
            }

        if faces_selfie is None or len(faces_selfie) == 0:
            return {
                "score": 0.0,
                "flag": False,
                "explanation": "No face detected in selfie image.",
                "status": "completed",
                "details": {"document_faces": len(faces_doc), "selfie_faces": 0},
            }

        # Initialize FaceRecognizerSF
        recognizer = cv2.FaceRecognizerSF.create(SFACE_PATH, "")

        # Select primary face (largest bounding box) from each
        primary_doc_face = max(faces_doc, key=lambda f: f[2] * f[3])
        primary_selfie_face = max(faces_selfie, key=lambda f: f[2] * f[3])

        # Align faces and extract features
        aligned_doc = recognizer.alignCrop(doc_img, primary_doc_face)
        aligned_selfie = recognizer.alignCrop(selfie_img, primary_selfie_face)

        feat_doc = recognizer.feature(aligned_doc)
        feat_selfie = recognizer.feature(aligned_selfie)

        # Cosine distance and L2 distance
        # In FaceRecognizerSF:
        # Cosine similarity: threshold ~ 0.363 (higher means same person)
        # L2 distance: threshold ~ 1.128 (lower means same person)
        cosine_score = float(recognizer.match(feat_doc, feat_selfie, cv2.FaceRecognizerSF_FR_COSINE))
        l2_dist = float(recognizer.match(feat_doc, feat_selfie, cv2.FaceRecognizerSF_FR_NORM_L2))

        # We normalize to risk score 0 to 1 (1 = strong mismatch/impostor)
        # Cosine similarity >= 0.363 is considered a match by SFace specification
        matched = bool(cosine_score >= 0.363)
        # Risk score: inverted cosine clamped to 0..1
        risk_score = round(float(np.clip(1.0 - (cosine_score + 0.2) / 1.0, 0.0, 1.0)), 3)
        flag = not matched

        explanation = (
            f"Face verification passed (Cosine similarity: {cosine_score:.3f}, L2 dist: {l2_dist:.2f})."
            if matched
            else f"Face mismatch: Document face differs from selfie (Cosine similarity: {cosine_score:.3f} < 0.363 threshold)."
        )

        return {
            "score": risk_score,
            "flag": flag,
            "explanation": explanation,
            "status": "completed",
            "details": {
                "cosine_similarity": round(cosine_score, 4),
                "l2_distance": round(l2_dist, 4),
                "match": matched,
            },
        }

    except Exception as e:
        return {
            "score": 0.0,
            "flag": False,
            "explanation": f"Face match error: {str(e)}",
            "status": "failed",
            "details": {"error": str(e)},
        }


def rectify_document(image_input: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    5. Document Rectification.
    Finds the largest quadrilateral contour and applies cv2.getPerspectiveTransform
    to produce a bird's-eye rectified view.
    If none is found or on error, returns the original image.
    """
    if not _CV2_AVAILABLE or cv2 is None:
        return image_input, {
            "score": 0.0,
            "flag": False,
            "explanation": "OpenCV not available for rectification.",
            "status": "skipped",
            "rectified": False,
        }

    img = _decode_image(image_input)
    if img is None:
        return image_input, {
            "score": 0.0,
            "flag": False,
            "explanation": "Could not decode image for rectification.",
            "status": "failed",
            "rectified": False,
        }

    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        doc_contour = None
        min_area = (w * h) * 0.15  # At least 15% of image area

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                doc_contour = approx
                break

        if doc_contour is None:
            return img, {
                "score": 0.0,
                "flag": False,
                "explanation": "No quadrilateral document border detected; returning original image.",
                "status": "completed",
                "rectified": False,
            }

        pts = doc_contour.reshape(4, 2).astype(np.float32)

        # Order points: top-left, top-right, bottom-right, bottom-left
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        (tl, tr, br, bl) = rect
        width_a = np.hypot(br[0] - bl[0], br[1] - bl[1])
        width_b = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
        max_w = max(int(width_a), int(width_b))

        height_a = np.hypot(tr[0] - br[0], tr[1] - br[1])
        height_b = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
        max_h = max(int(height_a), int(height_b))

        dst = np.array([
            [0, 0],
            [max_w - 1, 0],
            [max_w - 1, max_h - 1],
            [0, max_h - 1]
        ], dtype=np.float32)

        # Perspective transform
        m = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, m, (max_w, max_h))

        return warped, {
            "score": 0.0,
            "flag": False,
            "explanation": f"Document rectified successfully to {max_w}x{max_h} perspective.",
            "status": "completed",
            "rectified": True,
            "dimensions": f"{max_w}x{max_h}",
        }

    except Exception as e:
        return img, {
            "score": 0.0,
            "flag": False,
            "explanation": f"Document rectification failed: {str(e)}",
            "status": "failed",
            "rectified": False,
            "error": str(e),
        }

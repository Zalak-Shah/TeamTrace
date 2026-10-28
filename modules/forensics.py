"""
============================================================
BorderShield AI - AI & Tamper Forensics Module (Shared Fallback)
============================================================
Features:
1. Error Level Analysis (ELA):
   - Computes compression artifact variance across blocks at 90% JPEG quality.
   - Highlights spliced, pasted, or cloned regions.
   - Safely handles PNG / PDF with format conversion or limitation note.
2. EXIF / Metadata Inspection:
   - Detects generative AI or photo-editing signatures (Photoshop, GIMP, Canva,
     StableDiffusion, Midjourney, DALL-E, etc.).
   - Mildly notes missing EXIF without marking document as fraudulent.
3. Frequency / Noise Anomaly Analysis (FFT):
   - Analyzes high-frequency 2D Fourier power spectrum for synthetic periodic grid patterns.
4. Returns standardized `ai_tamper_analysis` object with strict 'risk' nomenclature.
============================================================
"""

import io
import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None
    _NUMPY_AVAILABLE = False

from PIL import Image, ImageChops, ImageEnhance


KNOWN_MANIPULATION_SOFTWARE = [
    "photoshop",
    "gimp",
    "canva",
    "picsart",
    "pixlr",
    "stablediffusion",
    "midjourney",
    "dall-e",
    "automatic1111",
    "comfyui",
    "faceapp",
    "remaker",
]


def analyze_metadata(image_bytes: bytes) -> Tuple[bool, int, List[str], Dict[str, Any]]:
    """
    Inspects image metadata and EXIF for digital manipulation or AI software footprints.
    Returns (manipulation_flag, risk_score, findings, details)
    """
    findings = []
    details = {}
    risk_score = 0
    flag = False

    try:
        img = Image.open(io.BytesIO(image_bytes))
        info = img.info or {}
        details["format"] = img.format
        details["size"] = f"{img.width}x{img.height}"

        # 1. Search text comments / PNG info
        for key, val in info.items():
            if isinstance(val, (str, bytes)):
                val_str = str(val).lower()
                for sw in KNOWN_MANIPULATION_SOFTWARE:
                    if sw in val_str:
                        flag = True
                        risk_score = max(risk_score, 80)
                        findings.append(f"Editing / AI software signature detected in image metadata: '{sw}'")

        # 2. EXIF examination
        exif = img.getexif()
        if exif:
            details["exif_present"] = True
            software = exif.get(0x0131)  # Software tag
            if software:
                soft_str = str(software).lower()
                details["software_tag"] = str(software)
                for sw in KNOWN_MANIPULATION_SOFTWARE:
                    if sw in soft_str:
                        flag = True
                        risk_score = max(risk_score, 85)
                        findings.append(f"Image was modified using photo-editing or AI tool: '{software}'")
        else:
            details["exif_present"] = False
            # Missing EXIF is normal for WhatsApp, Telegram, or web scans - mild advisory only
            findings.append("No camera EXIF metadata present (common in web scans, messaging apps, or screenshots).")
            risk_score = max(risk_score, 10)

    except Exception as e:
        details["metadata_read_error"] = str(e)
        findings.append(f"Could not parse image header: {str(e)}")

    return flag, risk_score, findings, details


def compute_ela_score(image_bytes: bytes, quality: int = 90) -> Tuple[bool, float, List[str], Dict[str, Any]]:
    """
    Performs Error Level Analysis (ELA) by re-compressing at 90% quality and measuring difference.
    Returns (anomaly_detected, anomaly_score_0_100, findings, details)
    """
    findings = []
    details = {}

    try:
        original = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        details["original_format"] = original.format or "Converted RGB"

        # Re-save to JPEG in memory
        buffer = io.BytesIO()
        original.save(buffer, "JPEG", quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer)

        # Difference
        diff = ImageChops.difference(original, resaved)
        # Enhance for analysis
        enhanced = ImageEnhance.Brightness(diff).enhance(10.0)

        # Convert to numpy array to measure variance across image
        diff_arr = np.asarray(diff, dtype=np.float32)
        mean_err = float(np.mean(diff_arr))
        std_err = float(np.std(diff_arr))
        max_err = float(np.max(diff_arr))

        details["mean_error"] = round(mean_err, 2)
        details["std_error"] = round(std_err, 2)
        details["max_error"] = round(max_err, 2)

        # Localized block analysis (16x16 grid) to identify localized spliced tampering
        h, w = diff_arr.shape[:2]
        block_h = max(h // 16, 8)
        block_w = max(w // 16, 8)
        block_means = []
        for r in range(0, h - block_h + 1, block_h):
            for c in range(0, w - block_w + 1, block_w):
                blk = diff_arr[r:r+block_h, c:c+block_w]
                block_means.append(float(np.mean(blk)))

        block_variance = float(np.std(block_means)) if block_means else std_err
        details["block_variance"] = round(block_variance, 2)

        # In natural documents, error level is relatively uniform.
        # High standard deviation / spikes indicate spliced elements.
        anomaly_score = min(max(round((std_err / 18.0) * 80 + (block_variance / 12.0) * 20, 1), 0.0), 100.0)

        # Optional low-res base64 heatmap preview for visual auditing in frontend
        import base64
        thumb_buffer = io.BytesIO()
        enhanced.thumbnail((240, 240))
        enhanced.save(thumb_buffer, format="JPEG", quality=75)
        details["ela_preview_base64"] = base64.b64encode(thumb_buffer.getvalue()).decode("ascii")

        if anomaly_score >= 60.0:
            anomaly_detected = True
            findings.append(f"High compression variance detected (ELA score {anomaly_score}%, block variance {details['block_variance']}). Suggests non-uniform digital splicing or localized text edits.")
        elif anomaly_score >= 35.0:
            anomaly_detected = False
            findings.append(f"Moderate compression variance (ELA score {anomaly_score}%). May be caused by multi-generation compression or resizing.")
        else:
            anomaly_detected = False
            findings.append("Uniform compression levels across image blocks; no ELA splicing anomalies detected.")

        return anomaly_detected, anomaly_score, findings, details

    except Exception as e:
        findings.append(f"ELA could not be executed on this format: {str(e)}")
        return False, 0.0, findings, {"error": str(e)}


def analyze_frequency_spectrum(image_bytes: bytes) -> Tuple[bool, float, List[str]]:
    """
    2D Fourier Transform (FFT) analysis to detect synthetic periodic lattice artifacts
    produced by diffusion models (Midjourney, DALL-E, SD).
    """
    findings = []
    if not _NUMPY_AVAILABLE or np is None:
        return False, 0.0, ["Frequency analysis skipped (numpy unavailable)."]

    try:
        if _CV2_AVAILABLE and cv2 is not None:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return False, 0.0, ["Image decoding failed for frequency analysis."]
            resized = cv2.resize(img, (512, 512))
        else:
            # Fallback to PIL for decoding and resizing
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("L")
            resized_pil = pil_img.resize((512, 512))
            resized = np.array(resized_pil, dtype=np.uint8)

        f = np.fft.fft2(resized)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-9)

        # Calculate high-frequency energy ratio
        center_x, center_y = 256, 256
        radius = 180
        y, x = np.ogrid[:512, :512]
        dist_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        high_freq_mask = dist_from_center > radius

        high_freq_energy = float(np.mean(magnitude_spectrum[high_freq_mask]))
        overall_energy = float(np.mean(magnitude_spectrum))

        hf_ratio = high_freq_energy / (overall_energy + 1e-5)
        # Synthetic images often show unnatural high-frequency grid artifacts
        is_synthetic = hf_ratio > 1.35
        score = min(max(round((hf_ratio - 1.0) * 120, 1), 0.0), 99.0)

        if is_synthetic:
            findings.append("High-frequency power spectrum shows synthetic periodic grid artifacts characteristic of generative diffusion models.")
        else:
            findings.append("Frequency spectrum consistent with natural optical camera capture.")

        return is_synthetic, score, findings
    except Exception as e:
        return False, 0.0, [f"Frequency analysis error: {str(e)}"]


def run_ai_and_tamper_forensics(image_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Runs combined ELA, metadata inspection, and frequency analysis.
    Returns a comprehensive ai_tamper_analysis dictionary.
    """
    meta_flag, meta_score, meta_findings, meta_details = analyze_metadata(image_bytes)
    ela_flag, ela_score, ela_findings, ela_details = compute_ela_score(image_bytes)
    fft_flag, fft_score, fft_findings = analyze_frequency_spectrum(image_bytes)

    # Combined risk score calculation (Weighted: Metadata 35%, ELA 40%, FFT 25%)
    combined_risk = round(
        (meta_score * 0.35) + (ela_score * 0.40) + (fft_score * 0.25)
    )
    combined_risk = min(max(combined_risk, 0), 100)

    is_ai_risk = meta_flag or fft_flag or (combined_risk >= 65)

    all_findings = meta_findings + ela_findings + fft_findings

    limitations = [
        "Forensics signals indicate risk, NOT cryptographic proof of authenticity.",
        "Fully AI-generated synthetic documents contain no edits or splices and may appear completely uniform under ELA.",
        "Recompressed images (messaging apps, screenshots, low-quality camera sensors) frequently introduce compression artifacts that can mimic tampering.",
    ]

    return {
        "is_ai_generated_risk": is_ai_risk,
        "ela_anomaly_detected": ela_flag,
        "metadata_tampering_flag": meta_flag,
        "frequency_anomaly_detected": fft_flag,
        "combined_risk_score": combined_risk,
        "per_check_scores": {
            "metadata_risk": meta_score,
            "ela_variance_score": ela_score,
            "frequency_synthetic_score": fft_score,
        },
        "findings": all_findings,
        "limitations": limitations,
        "details": {
            "metadata": meta_details,
            "ela": ela_details,
        },
    }

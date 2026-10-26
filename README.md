# TeamTrace — BorderShield AI
### Automated Document Fraud Detection, Computer Vision Forensics & Tamper-Evident Audit Chains

[![Live Demo](https://img.shields.io/badge/Live%20Demo-team--trace.vercel.app-blue.svg)](https://team-trace.vercel.app)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Zalak--Shah%2FTeamTrace-black.svg)](https://github.com/Zalak-Shah/TeamTrace)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green.svg)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red.svg)](https://opencv.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Neon%20Serverless-007ACC.svg)](https://neon.tech/)

---

## Project Overview
**TeamTrace (BorderShield AI)** is an intelligent border checkpoint and document verification system designed to assist immigration and security personnel in detecting altered, counterfeit, and forged identification documents.

The system combines **OpenCV computer vision forensics**, multi-document verification rules (MRZ, Verhoeff checksums, format validation), OCR-based cross-checks, and a **self-hosted cryptographic hash chain** that provides tamper-evident audit logging for recorded screening decisions.

- **Live Deployment:** [https://team-trace.vercel.app](https://team-trace.vercel.app)
- **Source Code:** [https://github.com/Zalak-Shah/TeamTrace](https://github.com/Zalak-Shah/TeamTrace)

---

## OpenCV Computer Vision & Forensics Architecture

TeamTrace implements five forensic and computer vision capabilities in [modules/opencv_forensics.py](modules/opencv_forensics.py) and [main.py](main.py), powered by OpenCV:

### 1. Error Level Analysis (ELA) (`run_ela_check`)
- **How it works:** Re-compresses the input image as a JPEG at quality 90 using `cv2.imencode`, decodes it back with `cv2.imdecode`, and computes the pixel-level absolute difference matrix using `cv2.absdiff`.
- **Purpose:** Identifies regions with compression anomalies. Digitally spliced elements, pasted text, and inserted photo stamps often show distinct error levels relative to the original document background.

### 2. Copy-Move Forgery Detection (`run_copy_move_check`)
- **How it works:** Extracts keypoint features and descriptors across the document using **SIFT** (`cv2.SIFT_create`) with an **ORB** fallback (`cv2.ORB_create`). It matches features using `cv2.FlannBasedMatcher` (or `cv2.BFMatcher`) and filters out spatial self-matches.
- **Affine Geometric Filtering:** Employs `cv2.estimateAffinePartial2D` with RANSAC to detect cloned, duplicated, or repeated visual patterns indicative of duplicated text or stamps.

### 3. Document Contour Detection & Perspective Rectification (`rectify_document`)
- **How it works:** Applies bilateral filtering (`cv2.bilateralFilter`) and edge detection (`cv2.Canny`), finds contours (`cv2.findContours`), and calculates convex hulls and polygon approximations (`cv2.approxPolyDP`) to isolate the document boundary.
- **Warp Transform:** Extracts the quad corner points and applies `cv2.getPerspectiveTransform` and `cv2.warpPerspective` to correct perspective tilt, delivering a planar, rectified image to optimize OCR accuracy.

### 4. Facial Detection & 1:1 Biometric Verification (`run_face_verification`)
- **Face Detection:** Uses OpenCV Zoo's **YuNet** ONNX model (`cv2.FaceDetectorYN`) to detect faces and extract 5 facial landmarks from both the document photo and an optional live selfie.
- **Face Recognition:** Uses the **SFace** ONNX model (`cv2.FaceRecognizerSF`) to generate 128-dimensional L2-normalized feature embeddings and calculates cosine similarity (`cv2.FaceRecognizerSF.FR_COSINE`) to assess face match confidence.

### 5. QR Code Localization & Cross-Validation (`run_qr_check`)
- **How it works:** Utilizes `cv2.QRCodeDetector` to locate and decode 2D barcodes on Aadhaar cards and supported identity credentials.
- **Data Consistency:** Compares decoded data against visual OCR text to flag discrepancies between the digital payload and printed identity fields.

---

## Tamper-Evident Audit Hash Chain

TeamTrace enforces accountability using an **in-database, sequential SHA-256 hash chain** implemented in [database.py](database.py). 

> **Architecture Clarification:** This is a lightweight, self-hosted cryptographic hash chain stored in PostgreSQL. It does **not** use an external decentralized blockchain network, cryptocurrency tokens, or consensus mechanisms.

- **Genesis Block:** The initial hash is defined as 64 zeros (`GENESIS_HASH = "0" * 64`).
- **Sequential Block Hashing:** Each recorded audit log entry computes its SHA-256 block hash incorporating the preceding block's hash:
  `Hash_n = SHA256(prev_hash | officer_name | action | details | created_at | screening_id)`
- **Tamper Detection Implementation:** The `/api/audit/verify` endpoint executes `verify_audit_chain()`, which recalculates each hash in order from block 1 to N. If any past database row has been altered, reordered, or deleted, the recalculation detects the mismatch, reporting `valid: false` along with the specific broken block ID and reason. Note that row deletions or external rewrites can break the chain sequence, which the verifier flags as broken rather than auto-repairing.

---

## Multi-Document Rule Verification Engine

The modular verification engine in `modules/` applies document-specific rules:
- **Passports (`passport.py`):** ICAO Doc 9303 Machine Readable Zone (MRZ) extraction, line checksum verification, issuing authority matching, and expiration date checks.
- **Aadhaar Cards (`aadhaar.py`):** Verhoeff checksum algorithm validation on the 12-digit UID and format verification.
- **PAN Cards (`pan.py`):** 10-character alphanumeric structural validation (4th character entity type check, 5th character surname initial check).
- **Driving Licences (`driving_licence.py`):** Modern Sarathi 15/16-character format regex matching and legacy state licensing pattern validation.
- **Visas & Permits (`visa.py`, `permit.py`):** Entry limits, date span validity, and permit number extraction.

---

## Tech Stack

- **Backend:** FastAPI, Python 3.10+, Uvicorn, AsyncIO
- **Computer Vision:** OpenCV (`opencv-python-headless`), NumPy, Pillow
- **Database:** PostgreSQL (Neon Serverless) via Psycopg2
- **Frontend:** Vanilla HTML5, CSS (Glassmorphism design, Dark Mode), Vanilla JavaScript
- **Deployment:** Vercel (Frontend & Serverless API)

---

## Environment Configuration

TeamTrace uses the following environment variables (configured in a local `.env` file or hosting provider settings):

| Variable Name | Description |
| :--- | :--- |
| `DATABASE_URL` | PostgreSQL connection URI (e.g. Neon connection string). |
| `OCR_SPACE_API_KEY` | *(Optional)* API key for OCR.space fallback processing. |
| `REQUIRE_OPENCV` | Set to `1` in environments where OpenCV is strictly required at startup. |

*Note: Never commit secret values or `.env` files to source control.*

---

## Setup & Execution Guide

### Local Development (Full OpenCV & Computer Vision Support)

To run TeamTrace locally with full OpenCV forensics and biometric face models enabled:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Zalak-Shah/TeamTrace.git
   cd TeamTrace
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv

   # On Windows:
   .venv\Scripts\activate

   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   For local development with full OpenCV capabilities, use `requirements-aws.txt` (which includes `opencv-python-headless` and `numpy`):
   ```bash
   pip install -r requirements-aws.txt
   ```

4. **Verify Face Verification ONNX Models:**
   Ensure the two OpenCV Zoo ONNX model files are located in the `models/` folder:
   - `models/face_detection_yunet_2023mar.onnx`
   - `models/face_recognition_sface_2021dec.onnx`

5. **Configure `.env`:**
   Create a `.env` file in the root directory:
   ```env
   DATABASE_URL=postgresql://user:password@hostname/dbname?sslmode=require
   OCR_SPACE_API_KEY=your_ocr_space_api_key_here
   ```

6. **Start the backend server:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```
   Access the web interface at:
   - **Dashboard:** `http://localhost:8000/index.html`
   - **Document Screening:** `http://localhost:8000/screening.html`
   - **Audit Ledger:** `http://localhost:8000/audit.html`
   - **Interactive API Docs:** `http://localhost:8000/docs`

---

### Deployment Note (Vercel vs. Local / Server)

- **Vercel Serverless Deployment:** 
  The production web demo runs on Vercel using `requirements.txt`. Vercel serverless functions enforce a strict 50 MB uncompressed code package limit; heavy C++ binary dependencies (OpenCV) are lazy-loaded only if available to prevent cold-start timeouts.
- **Local / Container / Server Deployment:** 
  For full server or local deployments where OpenCV computer vision forensics (ELA, SIFT/ORB Copy-Move, and YuNet/SFace) are fully active, install via `requirements-aws.txt`.

---

## Testing

To run the offline unit test suite:
```bash
# Test cryptographic hash chain logic:
python -m unittest test_audit_chain.py

# Test document verifiers (Passport & Aadhaar):
python -m unittest test_passport.py test_aadhaar.py
```

---

## Authors
- **TeamTrace / BorderShield AI** — SIH 2026 & OpenCV Hackathon Submission

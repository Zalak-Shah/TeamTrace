"""
Download YuNet and SFace ONNX models from the OpenCV Zoo.
Models are saved in the models/ directory.
"""
import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

MODELS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}


def download_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, url in MODELS.items():
        dest = os.path.join(MODELS_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"[models] {filename} already exists ({os.path.getsize(dest)} bytes).")
            continue
        print(f"[models] Downloading {filename} from {url}...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[models] Saved {filename} ({os.path.getsize(dest)} bytes).")
        except Exception as e:
            print(f"[models] Failed to download {filename}: {e}")


if __name__ == "__main__":
    download_models()

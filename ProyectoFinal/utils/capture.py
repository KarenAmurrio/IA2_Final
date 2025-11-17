# utils/capture.py
from PIL import Image
from pathlib import Path

from .helpers import save_face_image


def capture_and_save(person_name: str, image: Image.Image) -> Path:
    return save_face_image(person_name, image)

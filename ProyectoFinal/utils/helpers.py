# utils/helpers.py
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import cv2

from .config import FACES_DIR, EMBEDDINGS_DIR


def ensure_directories() -> None:
    """
    Crea las carpetas necesarias si no existen.
    """
    for folder in [FACES_DIR, EMBEDDINGS_DIR]:
        os.makedirs(folder, exist_ok=True)


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """
    Convierte una imagen PIL (RGB) a un array numpy en BGR para usar con OpenCV / InsightFace.
    """
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def save_face_image(person_name: str, image: Image.Image) -> Path:
    """
    Guarda una imagen de una persona en faces/<nombre>/img_XXX.jpg
    y devuelve la ruta.
    """
    ensure_directories()
    person_dir = FACES_DIR / person_name
    person_dir.mkdir(parents=True, exist_ok=True)

    existing = list(person_dir.glob("img_*.jpg"))
    next_index = len(existing) + 1
    filename = person_dir / f"img_{next_index:03d}.jpg"

    image.save(filename, format="JPEG")
    return filename


def load_embeddings_db() -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    """
    Carga la base de embeddings guardada en embeddings/embeddings.npz.
    Devuelve (embeddings, labels) o (None, None) si no existe.
    """
    db_path = EMBEDDINGS_DIR / "embeddings.npz"
    if not db_path.exists():
        return None, None

    data = np.load(db_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    return embeddings, labels


def get_dataset_summary() -> Tuple[int, int]:
    """
    Devuelve (cantidad_personas, cantidad_imagenes) en la carpeta faces/.
    """
    if not FACES_DIR.exists():
        return 0, 0

    persons = [d for d in FACES_DIR.iterdir() if d.is_dir()]
    num_persons = len(persons)
    num_images = 0

    for p in persons:
        num_images += len(list(p.glob("*.jpg")))

    return num_persons, num_images

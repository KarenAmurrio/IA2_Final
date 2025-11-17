# utils/train.py
from pathlib import Path
from typing import List
import random

import numpy as np
from PIL import Image

from .config import FACES_DIR, EMBEDDINGS_DIR
from .helpers import pil_to_bgr
from .insight_model import get_insight_app

# Máximo de imágenes por persona para acelerar el entrenamiento
MAX_IMAGES_PER_PERSON = 5

def build_embeddings_db(progress_callback=None) -> str:
    """
    Recorre faces/<persona>/ y construye una base de embeddings con InsightFace.
    Optimizado:
      - Usa como máximo MAX_IMAGES_PER_PERSON fotos por persona.
      - Redimensiona las imágenes a 640x640 para acelerar.
    """
    app = get_insight_app()

    people_dirs: List[Path] = [d for d in FACES_DIR.iterdir() if d.is_dir()]

    if not people_dirs:
        return "No hay directorios en 'faces/'. Primero registra al menos una persona buscada."

    all_embeddings = []
    all_labels = []

    # Contar solo las que realmente vamos a usar
    total_images = 0
    for person_dir in people_dirs:
        image_paths = list(person_dir.glob("*.jpg"))
        total_images += min(len(image_paths), MAX_IMAGES_PER_PERSON)

    processed = 0

    for person_dir in people_dirs:
        person_name = person_dir.name
        image_paths = list(person_dir.glob("*.jpg"))

        if not image_paths:
            continue

        # Tomar solo hasta MAX_IMAGES_PER_PERSON (aleatorias si hay muchas)
        if len(image_paths) > MAX_IMAGES_PER_PERSON:
            image_paths = random.sample(image_paths, MAX_IMAGES_PER_PERSON)

        for img_path in image_paths:
            img = Image.open(img_path)

            # Redimensionar para acelerar
            img.thumbnail((640, 640))

            bgr = pil_to_bgr(img)

            try:
                faces = app.get(bgr)
            except Exception as e:
                print(f"Error procesando {img_path}: {e}")
                continue

            if not faces:
                continue

            face = faces[0]
            embedding = face.embedding.astype("float32")

            all_embeddings.append(embedding)
            all_labels.append(person_name)

            processed += 1
            if progress_callback:
                progress_callback(processed, total_images)

    if not all_embeddings:
        return "No se pudo generar ningún embedding. Revisa que las imágenes tengan rostros visibles."

    all_embeddings = np.stack(all_embeddings, axis=0)
    all_labels = np.array(all_labels)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    db_path = EMBEDDINGS_DIR / "embeddings.npz"
    np.savez_compressed(db_path, embeddings=all_embeddings, labels=all_labels)

    return f"Base de embeddings generada correctamente. Imágenes procesadas: {processed}."

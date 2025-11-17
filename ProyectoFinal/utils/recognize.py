# utils/recognize.py
from typing import Tuple, Optional

import numpy as np
from PIL import Image
import cv2

from .config import COSINE_THRESHOLD
from .helpers import load_embeddings_db, pil_to_bgr
from .insight_model import get_insight_app


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distancia coseno entre dos vectores.
    """
    a = a.astype("float32")
    b = b.astype("float32")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return 1.0 - float(np.dot(a, b) / denom)


def recognize_face(image: Image.Image) -> Tuple[Optional[str], float, str]:
    """
    Reconoce la persona en una imagen PIL usando InsightFace.
    Devuelve (label, distancia, mensaje_info).
    label = None si no hay base o si la distancia es > umbral.
    """
    embeddings_db, labels_db = load_embeddings_db()
    if embeddings_db is None or labels_db is None:
        return None, 0.0, "No hay base de embeddings. Genera la base en la pestaña de entrenamiento."

    app = get_insight_app()
    bgr = pil_to_bgr(image)

    try:
        faces = app.get(bgr)
    except Exception as e:
        return None, 0.0, f"Error obteniendo embedding de la imagen: {e}"

    if not faces:
        return None, 0.0, "No se detectó ningún rostro en la imagen."

    face = faces[0]
    embedding = face.embedding.astype("float32")

    best_label = None
    best_distance = 999.0

    for emb_db, label_db in zip(embeddings_db, labels_db):
        dist = cosine_distance(embedding, emb_db)
        if dist < best_distance:
            best_distance = dist
            best_label = label_db

    if best_distance <= COSINE_THRESHOLD:
        info = "Match por debajo del umbral, se considera la misma persona."
        return best_label, best_distance, info
    else:
        info = "La distancia supera el umbral, se considera desconocido."
        return None, best_distance, info


def recognize_and_annotate(image: Image.Image):
    """
    Reconoce el rostro y devuelve:
    - label: nombre o None
    - distance: distancia coseno
    - info: mensaje explicativo
    - annotated_img: imagen PIL con el recuadro y la etiqueta dibujados (si hay rostro)
    """
    embeddings_db, labels_db = load_embeddings_db()
    if embeddings_db is None or labels_db is None:
        return None, 0.0, "No hay base de embeddings. Genera la base en la pestaña de entrenamiento.", image

    app = get_insight_app()
    bgr = pil_to_bgr(image)

    try:
        faces = app.get(bgr)
    except Exception as e:
        return None, 0.0, f"Error obteniendo embedding de la imagen: {e}", image

    if not faces:
        return None, 0.0, "No se detectó ningún rostro en la imagen.", image

    face = faces[0]
    embedding = face.embedding.astype("float32")

    best_label = None
    best_distance = 999.0

    for emb_db, label_db in zip(embeddings_db, labels_db):
        dist = cosine_distance(embedding, emb_db)
        if dist < best_distance:
            best_distance = dist
            best_label = label_db

    # Dibujamos el recuadro
    x1, y1, x2, y2 = face.bbox.astype(int)
    annotated = bgr.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label_text = "Desconocido"
    if best_label is not None:
        label_text = f"{best_label} ({best_distance:.3f})"

    cv2.putText(
        annotated,
        label_text,
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA
    )

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    if best_distance <= COSINE_THRESHOLD:
        info = "Match por debajo del umbral, se considera la misma persona."
        return best_label, best_distance, info, annotated_pil
    else:
        info = "La distancia supera el umbral, se considera desconocido."
        return None, best_distance, info, annotated_pil

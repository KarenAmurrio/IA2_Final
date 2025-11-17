# utils/config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

FACES_DIR = BASE_DIR / "faces"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
MODELS_DIR = BASE_DIR / "models"

# Umbral para distancia coseno (ajústalo con pruebas):
COSINE_THRESHOLD = 0.45  # InsightFace suele tolerar un poco más

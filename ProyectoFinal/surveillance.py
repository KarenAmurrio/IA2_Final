import cv2
import time
from pathlib import Path
from datetime import datetime
import simpleaudio as sa
from pathlib import Path

import numpy as np
from PIL import Image

from utils.config import COSINE_THRESHOLD
from utils.helpers import load_embeddings_db, pil_to_bgr
from utils.insight_model import get_insight_app

ALERT_SOUND = None

def load_alert_sound():
    global ALERT_SOUND
    if ALERT_SOUND is None:
        sound_path = Path("assets/alert.wav")
        if sound_path.exists():
            ALERT_SOUND = sa.WaveObject.from_wave_file(str(sound_path))
        else:
            print("[!] No se encontró assets/alert.wav; no se reproducirá sonido.")


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype("float32")
    b = b.astype("float32")
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return 1.0 - float(np.dot(a, b) / denom)


def load_labels():
    """Carga solo las etiquetas (nombres) junto a embeddings."""
    embeddings_db, labels_db = load_embeddings_db()
    if embeddings_db is None or labels_db is None:
        print("[!] No hay base de embeddings. Ejecuta el entrenamiento antes.")
        return None, None
    return embeddings_db, labels_db


def log_alert(person_name: str, distance: float):
    """Guarda una alerta en data/alerts.csv"""
    from pathlib import Path
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    log_path = data_dir / "alerts.csv"

    now = datetime.now().isoformat(timespec="seconds")
    line = f"{now},{person_name},{distance:.4f}\n"

    if not log_path.exists():
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("timestamp,person_name,distance\n")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def main():
    load_alert_sound()
    embeddings_db, labels_db = load_labels()
    if embeddings_db is None:
        return

    app = get_insight_app()

    # 0 = webcam por defecto.
    # Si quieres usar una cámara IP (celu con app tipo IP Webcam):
    cap = cv2.VideoCapture("http://192.168.0.5:4747/video")
    # cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[!] No se pudo abrir la cámara.")
        return

    print("[INFO] Vigilancia iniciada. Presiona 'q' para salir.")

    last_alert_time = 0
    alert_cooldown = 5  # segundos entre alertas para la MISMA persona

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] No se pudo leer frame de la cámara.")
            break

        # InsightFace trabaja en BGR, ya lo tenemos así
        faces = app.get(frame)

        alert_text = None

        if faces:
            for face in faces:
                (x1, y1, x2, y2) = face.bbox.astype(int)
                embedding = face.embedding.astype("float32")

                # Buscar el match más cercano
                best_label = None
                best_distance = 999.0
                for emb_db, label_db in zip(embeddings_db, labels_db):
                    dist = cosine_distance(embedding, emb_db)
                    if dist < best_distance:
                        best_distance = dist
                        best_label = label_db

                if best_distance <= COSINE_THRESHOLD:
                    # ¡ALERTA!
                    alert_text = f"ALERTA: {best_label} ({best_distance:.3f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame, alert_text, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                    )

                    current_time = time.time()
                    if current_time - last_alert_time > alert_cooldown:
                        log_alert(best_label, best_distance)
                        last_alert_time = current_time
                         # 🔊 Reproducir sonido (no bloqueante)
                        if ALERT_SOUND is not None:
                            ALERT_SOUND.play()
                else:
                    # Rostro no reconocido → opcional: marcar en amarillo
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(
                        frame, "Desconocido", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                    )

        # Si hay alerta, mostrar banner arriba
        if alert_text:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 255), -1)
            cv2.putText(
                frame, alert_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
            )

        cv2.imshow("FaceGuardian - Vigilancia", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Vigilancia detenida.")


if __name__ == "__main__":
    main()

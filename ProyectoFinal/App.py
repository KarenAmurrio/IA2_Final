# App.py
import time
from io import BytesIO
from pathlib import Path
from datetime import datetime
import csv
import subprocess 

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from utils.config import FACES_DIR, EMBEDDINGS_DIR, COSINE_THRESHOLD
from utils.helpers import (
    ensure_directories,
    get_dataset_summary,
    save_face_image,
    load_embeddings_db,
    pil_to_bgr,
)
from utils.capture import capture_and_save
from utils.train import build_embeddings_db
from utils.recognize import recognize_and_annotate, cosine_distance
from utils.insight_model import get_insight_app


# ---------- Utilidades propias de la app ----------

def log_alert(person_name: str, distance: float):
    """
    Registra una alerta en data/alerts.csv con timestamp, nombre y distancia.
    """
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

def play_alert_sound():
    """
    Reproduce el sonido de alerta desde assets/alert.wav usando aplay.
    No bloquea la ejecución del programa.
    """
    sound_path = Path("assets/alert.wav")
    if sound_path.exists():
        subprocess.Popen(
            ["aplay", str(sound_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    else:
        print("[!] No se encontró assets/alert.wav, no se reproducirá sonido.")
# ---------- Páginas / Tabs ----------

def page_register():
    st.subheader("📸 Registrar persona buscada")

    with st.container():
        col1, col2 = st.columns([1, 1])
        with col1:
            person_name = st.text_input(
                "Nombre de la persona buscada",
                help="Este nombre se usará como etiqueta en la base de datos.",
                key="register_name",
            )
            st.write("Toma una o varias fotos de la persona:")

            camera_image = st.camera_input(
                "Capturar desde la cámara",
                key="register_camera",
            )
            uploaded_image = st.file_uploader(
                "O subir una imagen desde archivo",
                type=["jpg", "jpeg", "png"],
                key="register_uploader",
            )

        with col2:
            st.markdown(
                """
                **Recomendaciones para el registro:**
                - Usa 2–5 fotos por persona.
                - Varía un poco el ángulo (ligero perfil, frontal).
                - Buena iluminación y rostro visible.
                - Una carpeta por persona en `faces/`.
                """
            )

    if not person_name:
        st.info("Escribe un nombre para habilitar el guardado.")
        return

    img_to_use = None

    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        img_to_use = Image.open(BytesIO(bytes_data)).convert("RGB")
    elif uploaded_image is not None:
        img_to_use = Image.open(uploaded_image).convert("RGB")

    if img_to_use is not None:
        st.image(
            img_to_use,
            caption=f"Previsualización para {person_name}",
            use_column_width=True,
        )

        if st.button("💾 Guardar captura en dataset"):
            path = capture_and_save(person_name, img_to_use)
            st.success(f"Imagen guardada en: `{path}`")
    else:
        st.info("Toma una foto o sube una imagen para continuar.")


def page_train():
    st.subheader("🧠 Generar / Actualizar base de embeddings")

    st.write(
        "Esta acción recorre todas las imágenes en la carpeta `faces/` "
        "y genera una base de embeddings para el reconocimiento facial usando InsightFace."
    )

    if not FACES_DIR.exists() or not any(FACES_DIR.iterdir()):
        st.warning("No se encontraron personas en `faces/`. Registra al menos una persona primero.")
        return

    if st.button("⚙️ Construir base de embeddings"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(done, total):
            pct = int(done / total * 100)
            progress_bar.progress(pct)
            status_text.text(f"Procesando imágenes... {done}/{total} ({pct} %)")

        msg = build_embeddings_db(progress_callback=progress_callback)
        status_text.text("")
        progress_bar.progress(100)
        st.success(msg)


def page_recognize():
    st.subheader("🔍 Probar reconocimiento (foto única)")

    st.write("Toma una foto o sube una imagen para intentar detectar una **persona buscada**.")

    col1, col2 = st.columns([1, 1])

    with col1:
        camera_image = st.camera_input(
            "Capturar desde cámara",
            key="recognize_camera",
        )
        uploaded_image = st.file_uploader(
            "O subir una imagen desde archivo",
            type=["jpg", "jpeg", "png"],
            key="recognize_uploader",
        )

    img_to_use = None

    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        img_to_use = Image.open(BytesIO(bytes_data)).convert("RGB")
    elif uploaded_image is not None:
        img_to_use = Image.open(uploaded_image).convert("RGB")

    with col2:
        if img_to_use is not None:
            st.image(img_to_use, caption="Imagen original", use_column_width=True)
        else:
            st.info("Toma una foto o sube una imagen para ver aquí la previsualización.")

    st.write("---")

    if img_to_use is not None:
        if st.button("▶️ Ejecutar reconocimiento"):
            label, distance, info, annotated = recognize_and_annotate(img_to_use)

            st.image(annotated, caption="Resultado con detección", use_column_width=True)
            st.write(f"**Distancia calculada:** `{distance:.4f}`")
            st.write(info)

            if label is None:
                st.success("Resultado: Persona **desconocida** (no está en la base).")
            else:
                st.error(f"🚨 ALERTA: Persona buscada detectada: **{label}**")
    else:
        st.info("Toma una foto o sube una imagen para iniciar el reconocimiento.")


def page_live_monitor():
    st.subheader("🛡 Vigilancia en vivo (panel admin)")

    st.write(
        "Esta vista permite monitorear la cámara en tiempo casi real y generar "
        "alertas visuales cuando se detecta una **persona buscada**."
    )

    embeddings_db, labels_db = load_embeddings_db()
    if embeddings_db is None or labels_db is None:
        st.error("No hay base de embeddings. Primero genera la base en la pestaña **🧠 Generar embeddings**.")
        return

    # Asegurar que las etiquetas sean strings
    labels_db = np.array(
        [lbl.decode("utf-8") if isinstance(lbl, bytes) else str(lbl) for lbl in labels_db],
        dtype=object,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        source_option = st.selectbox(
            "Origen de la cámara",
            ["Webcam local (0)", "Cámara IP (URL)"],
            key="live_source",
        )

        default_url = "http://192.168.0.5:4747/video"
        ip_url = ""
        if source_option == "Cámara IP (URL)":
            ip_url = st.text_input(
                "URL del stream de la cámara IP",
                value=default_url,
                key="live_ip_url",
                help="Por ejemplo: http://IP_DEL_CELU:PUERTO/video",
            )

        duration = st.slider(
            "Duración del monitoreo (segundos)",
            min_value=5,
            max_value=30,
            value=10,
            step=5,
        )

        start_button = st.button("▶ Iniciar monitoreo")

    frame_placeholder = st.empty()
    alert_placeholder = st.empty()

    if start_button:
        app = get_insight_app()

        if source_option == "Webcam local (0)":
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(ip_url)

        if not cap.isOpened():
            st.error("No se pudo abrir la cámara. Verifica la URL o el dispositivo.")
            return

        alert_placeholder.info("Monitoreo en curso...")
        end_time = time.time() + duration
        last_alert_time = 0
        alert_cooldown = 5  # segundos entre alertas registradas

        while time.time() < end_time:
            ret, frame = cap.read()
            if not ret:
                alert_placeholder.error("No se pudo leer frame de la cámara.")
                break

            faces = app.get(frame)
            alert_text = None

            if faces:
                for face in faces:
                    (x1, y1, x2, y2) = face.bbox.astype(int)
                    embedding = face.embedding.astype("float32")

                    best_label = None
                    best_distance = 999.0
                    for emb_db, label_db in zip(embeddings_db, labels_db):
                        dist = cosine_distance(embedding, emb_db)
                        if dist < best_distance:
                            best_distance = dist
                            best_label = label_db

                    if best_distance <= COSINE_THRESHOLD:
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
                            play_alert_sound()
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(
                            frame, "Desconocido", (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                        )

            # Franja roja arriba si hubo alerta
            if alert_text:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 255), -1)
                cv2.putText(
                    frame, alert_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )
                alert_placeholder.error(alert_text)
            else:
                alert_placeholder.info("Sin alertas por el momento.")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            time.sleep(0.03)

        cap.release()
        alert_placeholder.success("Monitoreo finalizado.")


def page_alerts():
    st.subheader("📜 Historial de alertas")

    log_path = Path("data/alerts.csv")
    if not log_path.exists():
        st.info("Aún no hay alertas registradas. Ejecuta la vigilancia para generar alertas.")
        return

    rows = []
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        st.info("El archivo de alertas está vacío.")
        return

    st.write("Alertas registradas (las más recientes al final):")
    st.dataframe(rows, use_container_width=True)


# ---------- MAIN ----------

def main():
    st.set_page_config(
        page_title="FaceGuardian",
        page_icon="🛡️",
        layout="wide",
    )

    ensure_directories()

    # Métricas principales
    num_persons, num_images = get_dataset_summary()
    embeddings_exist = (EMBEDDINGS_DIR / "embeddings.npz").exists()

    st.title("FaceGuardian 🛡️")
    st.markdown("### Panel de administración y vigilancia facial para personas buscadas")

    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.metric("Personas buscadas registradas", num_persons)
    with mcol2:
        st.metric("Imágenes en el dataset", num_images)
    with mcol3:
        st.metric("Embeddings generados", "Sí" if embeddings_exist else "No")

    st.markdown("---")

    st.sidebar.title("FaceGuardian – Menú")
    st.sidebar.markdown(
        """
        Usa las pestañas de arriba para:

        1. **Registrar** personas buscadas.
        2. **Generar embeddings** de la base.
        3. **Probar reconocimiento** con una foto.
        4. **Vigilar en vivo** la cámara.
        5. **Revisar historial** de alertas.
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("Modelo: **InsightFace (buffalo_l)**  \nMétrica: **distancia coseno**")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "✅ Registrar persona buscada",
        "🧠 Generar embeddings",
        "🔍 Probar reconocimiento",
        "🛡 Vigilancia en vivo",
        "📜 Historial de alertas",
    ])

    with tab1:
        page_register()
    with tab2:
        page_train()
    with tab3:
        page_recognize()
    with tab4:
        page_live_monitor()
    with tab5:
        page_alerts()


if __name__ == "__main__":
    main()

# App.py
import time
from io import BytesIO
from pathlib import Path
from datetime import datetime
import csv
import subprocess 
from insightface.app import FaceAnalysis
import torch
from quality_cnn import QualityCNN, preprocess_face

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

# ================== MODELO CNN DE CALIDAD ==================

QUALITY_THRESHOLD_DEFAULT = 0.7  # umbral por defecto para la calidad

quality_model = QualityCNN()
try:
    state_dict = torch.load("quality_cnn.pt", map_location="cpu")
    quality_model.load_state_dict(state_dict)
    print("[OK] Modelo QualityCNN cargado correctamente.")
except FileNotFoundError:
    print("[!] quality_cnn.pt no encontrado. La CNN usará pesos sin entrenar (solo demo).")

quality_model.eval()


def is_good_quality(face_img, model=quality_model, threshold=QUALITY_THRESHOLD_DEFAULT):
    """
    Evalúa la calidad de un recorte de rostro usando la CNN.
    - face_img: recorte BGR (como viene de OpenCV)
    - threshold: umbral de probabilidad [0..1]
    Devuelve: (es_buena_calidad: bool, probabilidad: float)
    """
    tensor = preprocess_face(face_img)  # [1, 1, 112, 112]
    with torch.no_grad():
        prob = model(tensor).item()
    return prob >= threshold, prob


# ================== ESTILOS GLOBALES ==================

def set_custom_style():
    """
    Estilos básicos para un tema oscuro sencillo y profesional.
    """
    st.markdown("""
    <style>
        .main {
            background-color: #020617;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        h1, h2, h3, h4 {
            color: #e5e7eb;
        }
        p, label, span, .stMarkdown {
            color: #d1d5db;
        }
        .fg-card {
            background: #020617;
            border-radius: 0.9rem;
            border: 1px solid #1f2937;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
        }
        .fg-metric-card {
            background: #020617;
            border-radius: 0.9rem;
            border: 1px solid #1f2937;
            padding: 0.9rem 1rem;
            margin-bottom: 0.5rem;
        }
        .fg-badge-ok {
            background: rgba(34, 197, 94, 0.15);
            color: #22c55e;
            border-radius: 999px;
            padding: 0.25rem 0.75rem;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
        .fg-badge-warn {
            background: rgba(234, 179, 8, 0.15);
            color: #eab308;
            border-radius: 999px;
            padding: 0.25rem 0.75rem;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
        }
        button[role="tab"] {
            padding-top: 0.4rem !important;
            padding-bottom: 0.4rem !important;
        }
        .dataframe table {
            font-size: 0.85rem;
        }
        [data-testid="stSidebar"] {
            background-color: #020617;
            border-right: 1px solid #1f2937;
        }
        .stButton > button {
            border-radius: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)


# ================== UTILIDADES DE ALERTA ==================

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

def log_low_quality_detection(q_prob: float, bbox):
    """
    Registra en data/low_quality_faces.csv los casos donde la CNN
    detectó rostros de baja calidad (no confiables).
    """
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    log_path = data_dir / "low_quality_faces.csv"

    now = datetime.now().isoformat(timespec="seconds")

    if not log_path.exists():
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("timestamp,q_prob,x1,y1,x2,y2\n")

    x1, y1, x2, y2 = bbox
    line = f"{now},{q_prob:.4f},{x1},{y1},{x2},{y2}\n"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def play_alert_sound():
    """
    Reproduce el sonido de alerta desde assets/alert.wav usando aplay (no bloqueante).
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


# ================== PÁGINAS / TABS ==================

def page_register():
    st.markdown("### 📸 Registrar persona buscada")
    st.caption(
        "Captura o sube imágenes de personas reportadas como buscadas para "
        "alimentar la base de datos del sistema."
    )

    with st.container():
        col1, col2 = st.columns([1.2, 1])
        with col1:
            st.markdown('<div class="fg-card">', unsafe_allow_html=True)
            person_name = st.text_input(
                "Nombre completo de la persona",
                help="Este nombre se usará como etiqueta en el sistema.",
                key="register_name",
            )

            st.markdown("**Captura una o varias fotos:**")
            camera_image = st.camera_input(
                "Capturar desde la cámara",
                key="register_camera",
            )
            uploaded_image = st.file_uploader(
                "O subir una imagen desde archivo",
                type=["jpg", "jpeg", "png"],
                key="register_uploader",
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="fg-card">', unsafe_allow_html=True)
            st.markdown("#### Recomendaciones")
            st.markdown(
                """
                - Usa entre **2 y 5 fotos** por persona.  
                - Varía ligeramente el ángulo (frontal, leve perfil).  
                - Asegura **buena iluminación** y rostro visible.  
                - Una carpeta por persona en la carpeta `faces/`.  
                """
            )
            st.markdown('</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="fg-card">', unsafe_allow_html=True)
        st.image(
            img_to_use,
            caption=f"Previsualización – {person_name}",
            use_column_width=True,
        )

        if st.button("💾 Guardar imagen en el dataset"):
            path = capture_and_save(person_name, img_to_use)
            st.success(f"Imagen guardada en: `{path}`")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Toma una foto o sube una imagen para continuar.")


def page_train():
    st.markdown("### 🧠 Generar / actualizar base de embeddings")
    st.caption(
        "Este proceso recorre todas las imágenes en la carpeta `faces/` y genera "
        "la base de embeddings para el reconocimiento facial usando InsightFace."
    )

    if not FACES_DIR.exists() or not any(FACES_DIR.iterdir()):
        st.warning("No se encontraron personas en `faces/`. Registra al menos una persona primero.")
        return

    st.markdown('<div class="fg-card">', unsafe_allow_html=True)
    if st.button("⚙️ Construir base de embeddings"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(done, total):
            pct = int(done / total * 100)
            progress_bar.progress(pct)
            status_text.text(f"Procesando imágenes... {done}/{total} ({pct}%)")

        msg = build_embeddings_db(progress_callback=progress_callback)
        status_text.text("")
        progress_bar.progress(100)
        st.success(msg)
        time.sleep(1)
        st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def page_recognize():
    st.markdown("### 🔍 Prueba de reconocimiento (imagen única)")
    st.caption(
        "Toma una foto o sube una imagen para verificar si el sistema detecta "
        "a una persona previamente registrada como buscada."
    )

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown('<div class="fg-card">', unsafe_allow_html=True)
        camera_image = st.camera_input(
            "Capturar desde la cámara",
            key="recognize_camera",
        )
        uploaded_image = st.file_uploader(
            "O subir una imagen desde archivo",
            type=["jpg", "jpeg", "png"],
            key="recognize_uploader",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    img_to_use = None

    if camera_image is not None:
        bytes_data = camera_image.getvalue()
        img_to_use = Image.open(BytesIO(bytes_data)).convert("RGB")
    elif uploaded_image is not None:
        img_to_use = Image.open(uploaded_image).convert("RGB")

    with col2:
        st.markdown('<div class="fg-card">', unsafe_allow_html=True)
        if img_to_use is not None:
            st.image(img_to_use, caption="Imagen original", use_column_width=True)
        else:
            st.info("Toma una foto o sube una imagen para ver aquí la previsualización.")
        st.markdown('</div>', unsafe_allow_html=True)

    if img_to_use is not None:
        st.markdown("---")
        st.markdown('<div class="fg-card">', unsafe_allow_html=True)
        if st.button("▶️ Ejecutar reconocimiento"):
            label, distance, info, annotated = recognize_and_annotate(img_to_use)

            st.image(annotated, caption="Resultado con detección", use_column_width=True)
            st.write(f"**Distancia calculada:** `{distance:.4f}`")
            st.write(info)

            if label is None:
                st.success("Resultado: Persona **desconocida** (no está en la base).")
            else:
                st.error(f"🚨 ALERTA: Persona buscada detectada: **{label}**")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Toma una foto o sube una imagen para iniciar el reconocimiento.")


def page_live_monitor():
    st.markdown("### 🛡 Vigilancia en vivo – Panel de monitoreo")
    st.caption(
        "Monitorea la cámara en tiempo casi real y genera alertas visuales y sonoras "
        "cuando se detecta una persona buscada."
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
        st.markdown('<div class="fg-card">', unsafe_allow_html=True)
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
                help="Ejemplo: http://IP_DEL_CELULAR:PUERTO/video",
            )

        duration = st.slider(
            "Duración del monitoreo (segundos)",
            min_value=5,
            max_value=60,
            value=20,
            step=5,
        )

        quality_threshold = st.slider(
            "Umbral de calidad del rostro (CNN)",
            min_value=0.0,
            max_value=1.0,
            value=QUALITY_THRESHOLD_DEFAULT,
            step=0.05,
            help="Rostros con probabilidad menor a este valor se marcarán como baja calidad."
        )

        start_button = st.button("▶ Iniciar monitoreo")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="fg-card">', unsafe_allow_html=True)
        st.markdown("**Leyenda de colores:**")
        st.markdown(
            "- 🟥 **Rojo**: persona buscada detectada (alta confianza).\n"
            "- 🟨 **Amarillo**: rostro desconocido, pero de buena calidad.\n"
            "- 🟧 **Naranja**: rostro de **baja calidad**, el sistema muestra coincidencia "
            "pero advierte que puede haber falso positivo/negativo.\n"
            "- Todas las alertas confirmadas se registran en `data/alerts.csv`."
        )
        st.markdown('</div>', unsafe_allow_html=True)

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
        alert_cooldown = 5  # segundos entre alertas
        last_global_alert_text = None

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
                    face_crop = frame[y1:y2, x1:x2]

                    # ============ (1) Evaluación de calidad con CNN ============
                    good, q_prob = is_good_quality(
                        face_crop,
                        model=quality_model,
                        threshold=quality_threshold
                    )
                    q_text = f"Q={q_prob:.2f}"

                    # ============ (2) Embedding siempre, pero con interpretación distinta ============
                    embedding = face.embedding.astype("float32")

                    best_label = None
                    best_distance = 999.0
                    for emb_db, label_db in zip(embeddings_db, labels_db):
                        dist = cosine_distance(embedding, emb_db)
                        if dist < best_distance:
                            best_distance = dist
                            best_label = label_db

                    # ---------------------------------------------------------
                    # CASO A: calidad baja -> advertencia (NO alerta oficial)
                    # ---------------------------------------------------------
                    if not good:
                        warning_label = best_label if best_label is not None else "Desconocido"
                        warning_text = (
                            f"Baja calidad: posible FP/FN — {warning_label} "
                            f"(dist: {best_distance:.3f}, {q_text})"
                        )

                        # Rectángulo naranja
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
                        cv2.putText(
                            frame, warning_text, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2
                        )

                        # Opcional: registrar solo como estadística, no como alerta oficial
                        # log_low_quality_detection(q_prob, (x1, y1, x2, y2))

                        # No hay alerta global, solo visual
                        continue

                    # ---------------------------------------------------------
                    # CASO B: calidad buena -> proceso de reconocimiento normal
                    # ---------------------------------------------------------
                    if best_distance <= COSINE_THRESHOLD:
                        alert_text = f"ALERTA: {best_label} (dist: {best_distance:.3f}, {q_text})"
                        last_global_alert_text = alert_text

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
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
                        # Desconocido de buena calidad
                        unknown_text = f"Desconocido ({q_text})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cv2.putText(
                            frame, unknown_text, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                        )

            # Banner superior con la última alerta importante (si existe)
            if last_global_alert_text:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 255), -1)
                cv2.putText(
                    frame, last_global_alert_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                )
                alert_placeholder.error(last_global_alert_text)
            else:
                alert_placeholder.info("Sin alertas por el momento.")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            time.sleep(0.03)

        cap.release()
        alert_placeholder.success("Monitoreo finalizado.")



# ================== MAIN ==================

def main():
    st.set_page_config(
        page_title="FaceGuardian – Vigilancia Inteligente",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    set_custom_style()
    ensure_directories()

    # Métricas principales
    num_persons, num_images = get_dataset_summary()
    embeddings_exist = (EMBEDDINGS_DIR / "embeddings.npz").exists()

    # HEADER
    col_header, col_status = st.columns([2.5, 1.5])

    with col_header:
        st.markdown("## 🛡️ FaceGuardian")
        st.markdown(
            "Sistema de vigilancia con **reconocimiento facial** para apoyar la detección de "
            "**personas desaparecidas** en terminales de buses de Bolivia."
        )

    with col_status:
        st.markdown('<div class="fg-metric-card">', unsafe_allow_html=True)
        st.markdown("#### Estado del sistema")
        if embeddings_exist:
            st.markdown(
                '<span class="fg-badge-ok">Embeddings generados</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<span class="fg-badge-warn">Embeddings pendientes</span>',
                unsafe_allow_html=True
            )
        st.caption("Modelo: InsightFace (buffalo_l)")
        st.caption("Métrica: distancia coseno")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # MÉTRICAS EN CARDS
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.markdown('<div class="fg-metric-card">', unsafe_allow_html=True)
        st.caption("Personas registradas")
        st.markdown(f"### {num_persons}")
        st.markdown('</div>', unsafe_allow_html=True)

    with mcol2:
        st.markdown('<div class="fg-metric-card">', unsafe_allow_html=True)
        st.caption("Imágenes en el dataset")
        st.markdown(f"### {num_images}")
        st.markdown('</div>', unsafe_allow_html=True)

    with mcol3:
        st.markdown('<div class="fg-metric-card">', unsafe_allow_html=True)
        st.caption("Estado de embeddings")
        st.markdown(f"### {'Disponible' if embeddings_exist else 'No generado'}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # SIDEBAR
    with st.sidebar:
        st.markdown("## Menú")
        st.markdown(
            """
            1. **Registrar** personas buscadas.  
            2. **Generar embeddings** de la base.  
            3. **Probar reconocimiento** con una imagen.  
            4. **Vigilar en vivo** por cámara.  
            5. **Revisar historial** de alertas.  
            """
        )
        st.markdown("---")
        st.markdown("### Información del sistema")
        st.markdown(
            f"- Umbral de distancia: `{COSINE_THRESHOLD}`  \n"
            f"- Personas registradas: `{num_persons}`  \n"
            f"- Imágenes en dataset: `{num_images}`  \n"
        )

    # TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "✅ Registrar persona",
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

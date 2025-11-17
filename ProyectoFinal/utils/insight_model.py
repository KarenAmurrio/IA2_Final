# utils/insight_model.py
from insightface.app import FaceAnalysis

_app = None

def get_insight_app():
    """
    Devuelve una instancia única (singleton) de FaceAnalysis para evitar
    recargar el modelo en cada llamada.
    """
    global _app
    if _app is None:
        _app = FaceAnalysis(name="buffalo_l")  # modelo por defecto
        # ctx_id=-1 → CPU; si tienes GPU y drivers, puedes probar con 0
        _app.prepare(ctx_id=-1, det_size=(640, 640))
    return _app

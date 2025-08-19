"""
WSGI entrypoint for production servers (Gunicorn, uWSGI, etc.).
Usage:
    gunicorn -w 3 -b 0.0.0.0:$PORT src.web.wsgi:app
"""
from src.web.app import app  # Re-use the already-created Flask app

# Optional: warm-up tasks (uncomment if desired and fast)
# from src.storage.vector_store import ensure_index
# ensure_index()
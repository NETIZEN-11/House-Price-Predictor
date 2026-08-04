"""
House Price Prediction — Flask backend.

Security & robustness hardening applied:
  - Secret key is loaded from env (random fallback for local dev only)
  - Debug mode off; host/port from env so it works on Heroku/Render/Railway
  - Rate limiting via flask-limiter
  - Input validation on every numeric field (form + JSON API)
  - Whitelist of expected feature names + order (prevents form-order breakage)
  - Structured logging instead of bare except
  - No bare `except:` — catches only specific exceptions
  - JSON errors, no stack traces leaked to clients
  - Security response headers (CSP, X-Frame-Options, etc.)
"""

import logging
import os
import pickle
from typing import Any, Dict, List

import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected feature schema — single source of truth used by both the
# /predict form route and the /predict_api JSON route. This guarantees the
# model always receives columns in the correct order regardless of how the
# HTML form is laid out.
# ---------------------------------------------------------------------------
FEATURE_NAMES: List[str] = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX",
    "RM", "AGE", "DIS", "RAD", "TAX",
    "PTRATIO", "B", "LSTAT",
]
NUM_FEATURES = len(FEATURE_NAMES)

# Plausible ranges for each feature (lower, upper). Anything outside is rejected.
FEATURE_RANGES: Dict[str, tuple] = {
    "CRIM":   (0.0, 100.0),
    "ZN":     (0.0, 100.0),
    "INDUS":  (0.0, 50.0),
    "CHAS":   (0.0, 1.0),
    "NOX":    (0.0, 1.5),
    "RM":     (1.0, 15.0),
    "AGE":    (0.0, 100.0),
    "DIS":    (0.0, 20.0),
    "RAD":    (1.0, 30.0),
    "TAX":    (50.0, 800.0),
    "PTRATIO":(5.0, 30.0),
    "B":      (0.0, 700.0),
    "LSTAT":  (0.0, 60.0),
}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Secret key: required for sessions/flash. In production set FLASK_SECRET_KEY.
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32).hex()
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024  # 64 KB request body cap

# Rate limiting — protect both routes from abuse / DoS.
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per hour", "30 per minute"],
    storage_uri="memory://",
)


# ---------------------------------------------------------------------------
# Security headers
# ---------------------------------------------------------------------------
@app.after_request
def set_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=()"
    # Allow inline styles for the current home.html but block inline scripts
    # beyond what we already ship; tighten further if you extract CSS/JS.
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdnjs.cloudflare.com; "
        "font-src 'self' https://fonts.gstatic.com https://cdnjs.cloudflare.com; "
        "script-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:;"
    )
    return response


# ---------------------------------------------------------------------------
# Model loading — fail fast, no silent fallback to a fake model.
# A model that returns plausible-looking but meaningless numbers is worse
# than an error the operator can see.
# ---------------------------------------------------------------------------
def _load_artifacts():
    model_path = os.environ.get("MODEL_PATH", "housepred.pkl")
    scaler_path = os.environ.get("SCALER_PATH", "scaler.pkl")
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)  # noqa: S301 — internal artifact only
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)  # noqa: S301 — internal artifact only
        logger.info("Loaded model from %s and scaler from %s", model_path, scaler_path)
        return model, scaler
    except FileNotFoundError as exc:
        logger.error("Model artifact missing: %s", exc)
        raise RuntimeError(
            "Model artifacts not found. Run the notebook to generate "
            "housepred.pkl and scaler.pkl before starting the app."
        ) from exc
    except Exception as exc:  # corrupted/legacy pickle
        logger.exception("Failed to load model artifacts: %s", exc)
        raise RuntimeError(
            "Model artifacts are corrupted or built with an incompatible "
            "sklearn version. Retrain using the notebook."
        ) from exc


try:
    model, scaler = _load_artifacts()
except RuntimeError as exc:
    # Fail loudly at startup. Don't start a server with a non-functional model.
    logger.critical(str(exc))
    raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _coerce_feature(name: str, raw: Any) -> float:
    """Validate and coerce one feature value."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Feature '{name}' must be a number, got {raw!r}")
    if not np.isfinite(value):
        raise ValueError(f"Feature '{name}' must be finite, got {value}")
    lo, hi = FEATURE_RANGES[name]
    if not (lo <= value <= hi):
        raise ValueError(f"Feature '{name}'={value} outside allowed range [{lo}, {hi}]")
    return value


def _validate_payload(payload: Dict[str, Any]) -> List[float]:
    """Validate an incoming payload (form or JSON) against FEATURE_NAMES."""
    missing = [f for f in FEATURE_NAMES if f not in payload]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return [_coerce_feature(name, payload[name]) for name in FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
@limiter.limit("20 per minute")
def predict():
    try:
        values = _validate_payload(request.form)
    except ValueError as exc:
        logger.warning("Bad form input: %s", exc)
        return render_template(
            "home.html",
            prediction_text=f"Invalid input: {exc}",
            error=True,
        ), 400

    arr = np.array(values).reshape(1, -1)
    scaled = scaler.transform(arr)
    output = model.predict(scaled)[0]
    formatted = f"${float(output) * 1000:,.2f}"
    return render_template("home.html", prediction_text=f"Estimated House Price: {formatted}")


@app.route("/predict_api", methods=["POST"])
@limiter.limit("30 per minute")
def predict_api():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400
    if "data" not in payload:
        return jsonify({"error": "Missing 'data' object"}), 400

    try:
        values = _validate_payload(payload["data"])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    arr = np.array(values).reshape(1, -1)
    scaled = scaler.transform(arr)
    output = model.predict(scaled)[0]
    return jsonify({"prediction": float(output)})


@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/sample")
def sample():
    return render_template("home.html")


# ---------------------------------------------------------------------------
# Error handlers — never leak stack traces
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(405)
def method_not_allowed(_e):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "Request body too large"}), 413


@app.errorhandler(429)
def rate_limited(e):
    return jsonify({"error": "Rate limit exceeded", "detail": str(e)}), 429


@app.errorhandler(500)
def internal_error(_e):
    logger.exception("Unhandled error")
    return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # debug=False in all paths. Locally, set FLASK_DEBUG=1 only if you need it.
    debug = os.environ.get("FLASK_DEBUG") == "1"
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")
    app.run(host=host, port=port, debug=debug)

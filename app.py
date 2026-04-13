"""
SAP Customer Risk Monitoring System
Flask Application
"""

import os
import json
import uuid
import time
import gzip as _gzip
import requests as http_requests
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.utils import secure_filename
from ml_model import run_customer_risk_analysis
import numpy as np

app = Flask(__name__)
app.secret_key = "sap_crm_secret_2024"

BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER  = os.path.join(BASE_DIR, "uploads")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
ALLOWED_EXTENSIONS = {"csv"}

app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB

os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


# ── Gzip compression ──────────────────────────────────────────────────────────
@app.after_request
def compress_response(response):
    if (response.status_code < 200 or response.status_code >= 300
            or response.direct_passthrough):
        return response
    if "gzip" not in request.headers.get("Accept-Encoding", "").lower():
        return response
    content = response.get_data()
    if len(content) < 1024:
        return response
    compressed = _gzip.compress(content, compresslevel=6)
    if len(compressed) >= len(content):
        return response
    response.set_data(compressed)
    response.headers["Content-Encoding"] = "gzip"
    response.headers["Content-Length"]   = len(compressed)
    response.vary.add("Accept-Encoding")
    return response


# ── Static file caching ───────────────────────────────────────────────────────
@app.after_request
def add_cache_headers(response):
    if request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "public, max-age=604800, immutable"
    return response


# ── Utilities ─────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_numpy(obj):
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    return obj


def cleanup_old_results(max_age_seconds=86400):
    """Delete result JSON files older than 24 hours."""
    try:
        now = time.time()
        for fname in os.listdir(RESULTS_FOLDER):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(RESULTS_FOLDER, fname)
            if now - os.path.getmtime(fpath) > max_age_seconds:
                os.remove(fpath)
    except Exception:
        pass


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    cleanup_old_results()

    # All 6 tables are required
    required_fields = ["bsid_file", "bsad_file", "bkpf_file",
                       "kna1_file", "knb1_file", "knkk_file"]
    saved_paths = {}

    for field in required_fields:
        if field not in request.files:
            return jsonify({"error": f"Missing file: {field}"}), 400
        file = request.files[field]
        if file.filename == "":
            return jsonify({"error": f"No file selected for {field}"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": f"Invalid file type for {field}. Only CSV allowed."}), 400

        uid      = str(uuid.uuid4())[:8]
        filename = uid + "_" + secure_filename(file.filename)
        path     = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)
        saved_paths[field] = path

    try:
        results = run_customer_risk_analysis(
            bsid_path=saved_paths["bsid_file"],
            bsad_path=saved_paths["bsad_file"],
            bkpf_path=saved_paths["bkpf_file"],
            kna1_path=saved_paths["kna1_file"],
            knb1_path=saved_paths["knb1_file"],
            knkk_path=saved_paths["knkk_file"],
        )
        results = json.loads(json.dumps(results, default=convert_numpy))
    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
    finally:
        for p in saved_paths.values():
            try:
                os.remove(p)
            except Exception:
                pass

    result_id   = str(uuid.uuid4())[:12]
    result_file = os.path.join(RESULTS_FOLDER, f"{result_id}.json")
    with open(result_file, "w") as f:
        json.dump(results, f)

    session["result_id"] = result_id
    return jsonify({"status": "ok", "result_id": result_id})


@app.route("/results")
def results():
    result_id = session.get("result_id")
    if not result_id:
        return redirect(url_for("index", msg="no_session"))
    result_file = os.path.join(RESULTS_FOLDER, f"{result_id}.json")
    if not os.path.exists(result_file):
        session.pop("result_id", None)
        return redirect(url_for("index", msg="expired"))
    with open(result_file) as f:
        data = json.load(f)
    return render_template("results.html", data=json.dumps(data))


@app.route("/customers")
def customers():
    result_id = session.get("result_id")
    if not result_id:
        return redirect(url_for("index", msg="no_session"))
    result_file = os.path.join(RESULTS_FOLDER, f"{result_id}.json")
    if not os.path.exists(result_file):
        session.pop("result_id", None)
        return redirect(url_for("index", msg="expired"))
    with open(result_file) as f:
        data = json.load(f)
    return render_template("customers.html", data=json.dumps(data))


@app.route("/intelligence")
def intelligence():
    result_id = session.get("result_id")
    if not result_id:
        return redirect(url_for("index", msg="no_session"))
    result_file = os.path.join(RESULTS_FOLDER, f"{result_id}.json")
    if not os.path.exists(result_file):
        session.pop("result_id", None)
        return redirect(url_for("index", msg="expired"))
    with open(result_file) as f:
        data = json.load(f)
    return render_template("intelligence.html", data=json.dumps(data))


@app.route("/api/data")
def api_data():
    """Serve result JSON async — avoids embedding huge payloads in HTML."""
    result_id = session.get("result_id")
    if not result_id:
        return jsonify({"error": "no_session"}), 401
    result_file = os.path.join(RESULTS_FOLDER, f"{result_id}.json")
    if not os.path.exists(result_file):
        return jsonify({"error": "expired"}), 404
    with open(result_file) as f:
        data = json.load(f)
    resp = jsonify(data)
    resp.headers["Cache-Control"] = "private, max-age=300"
    return resp


@app.route("/api/claude", methods=["POST"])
def claude_proxy():
    """
    OpenRouter proxy for AI intelligence page.
    Primary  : arcee-ai/trinity-large-preview:free  (128K context)
    Fallback : qwen/qwen3-235b-a22b:free             (1M context)
    Returns Anthropic-style response for frontend compatibility.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return jsonify({"error": "OPENROUTER_API_KEY not set on server"}), 500

    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    system_prompt = payload.get("system", "")
    raw_messages  = payload.get("messages", [])
    max_tokens    = min(int(payload.get("max_tokens", 2048)), 2048)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for msg in raw_messages:
        messages.append({"role": msg["role"], "content": msg["content"]})

    MODELS = [
        "arcee-ai/trinity-large-preview:free",
        "qwen/qwen3-235b-a22b:free",
    ]

    def _call(model):
        try:
            resp = http_requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "HTTP-Referer":  "https://sap-crm.app",
                    "X-Title":       "SAP Customer Risk Monitor",
                },
                json={"model": model, "max_tokens": max_tokens, "messages": messages},
                timeout=120,
            )
            try:
                data = resp.json()
            except Exception:
                return None, f"Non-JSON response (HTTP {resp.status_code})"

            if resp.status_code == 200:
                choices = data.get("choices") or []
                if choices:
                    text = choices[0].get("message", {}).get("content", "")
                    if text and text.strip():
                        return text, None
                return None, f"Empty response: {str(data)[:200]}"

            err = data.get("error", {}).get("message", str(data)) if isinstance(data, dict) else str(data)
            return None, f"HTTP {resp.status_code}: {err}"

        except http_requests.exceptions.Timeout:
            return None, "Timed out after 120s"
        except Exception as e:
            return None, str(e)

    last_err = ""
    for model in MODELS:
        text, err = _call(model)
        if text:
            return jsonify({"content": [{"type": "text", "text": text}]}), 200
        last_err = err

    return jsonify({"error": f"All models failed. Last error: {last_err}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
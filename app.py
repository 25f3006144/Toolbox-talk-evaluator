from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"
PIPELINE_INPUT = INPUT_DIR / "con.mp4"

PARAMETERS_FILE = OUTPUT_DIR / "parameter_extraction.json"
SCORING_FILE = OUTPUT_DIR / "scoring_results.json"
REPORT_FILE = OUTPUT_DIR / "final_evaluation_report.md"

ALLOWED_EXTENSIONS = {"mp4", "mov", "mkv"}
MAX_UPLOAD_MB = 500
SCRIPT_TIMEOUT_SECONDS = 1800

STAGES = [
    ("main.py", "Video, audio, speech and computer-vision analysis"),
    ("run_scoring_engine.py", "Scoring engine"),
    ("generate_report.py", "Final evaluation report"),
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# The pipeline uses one fixed input filename, so only one job may run at once.
LOCK = threading.Lock()
JOB = {
    "state": "idle",  # idle | running | done | error
    "stage": -1,
    "current_stage": "",
    "error": None,
    "logs": "",
    "started": None,
}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_for_json(value):
    """Convert NaN/Infinity to null so browser JSON parsing remains valid."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    return value


def read_json_file(path):
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return clean_for_json(data) if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def first_number(data, keys):
    if not isinstance(data, dict):
        return None
    for key in keys:
        value = data.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def get_summary():
    """Read the actual output keys produced by this Toolbox Talk project."""
    parameters = read_json_file(PARAMETERS_FILE)
    scoring = read_json_file(SCORING_FILE)

    coverage = first_number(
        scoring,
        [
            "topic_detection_coverage_percentage",
            "topic_detection_coverage",
            "topic_coverage_percentage",
        ],
    )
    coverage_label = f"{coverage:.2f}%" if coverage is not None else "Not available"

    topic_count = scoring.get("topic_detection_count", {})
    if not isinstance(topic_count, dict):
        topic_count = {}
    detected = topic_count.get("detected")
    total = topic_count.get("total_expected")

    if isinstance(detected, (int, float)) and isinstance(total, (int, float)):
        topics_label = f"{detected:g} of {total:g}"
    else:
        old_topics = scoring.get("topics_detected")
        if isinstance(old_topics, list):
            topics_label = str(len(old_topics))
        elif isinstance(old_topics, (int, float)) and not isinstance(old_topics, bool):
            topics_label = str(old_topics)
        else:
            topics_label = "Not available"

    # Keep this unvalidated until the project's scoring rules are validated.
    overall = scoring.get("overall_score")
    score_label = "Not validated" if overall is None else str(overall)

    duration = first_number(parameters, ["video_duration_seconds", "video_duration"])
    if duration is None:
        duration_label = "See report"
    elif duration >= 60:
        duration_label = f"{int(duration // 60)} min {int(duration % 60)} sec"
    else:
        duration_label = f"{duration:.1f} sec"

    return {
        "coverage": coverage_label,
        "coverage_value": coverage,
        "topics": topics_label,
        "score": score_label,
        "duration": duration_label,
        "parameters_available": PARAMETERS_FILE.exists(),
        "scoring_available": SCORING_FILE.exists(),
        "report_available": REPORT_FILE.exists(),
    }


def run_script(script_name):
    script_path = BASE_DIR / script_name
    if not script_path.is_file():
        raise FileNotFoundError(f"Pipeline script was not found next to app.py: {script_name}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=SCRIPT_TIMEOUT_SECONDS,
        check=False,
    )
    output = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()

    if completed.returncode != 0:
        raise RuntimeError(
            f"{script_name} failed (exit code {completed.returncode}).\n\n{output[-8000:]}"
        )
    return output or f"{script_name} finished without console output."


def run_pipeline(backup_dir, had_original):
    backup_path = Path(backup_dir) / "original_con.mp4"
    logs = []

    try:
        for index, (script, label) in enumerate(STAGES):
            with LOCK:
                JOB.update(stage=index, current_stage=label)
            output = run_script(script)
            logs.append(f"===== {script} =====\n{output}")
            with LOCK:
                JOB["logs"] = "\n\n".join(logs)[-16000:]

        with LOCK:
            JOB.update(
                state="done",
                stage=len(STAGES),
                current_stage="Analysis complete",
                error=None,
                logs="\n\n".join(logs)[-16000:],
            )

    except subprocess.TimeoutExpired as exc:
        logs.append(f"ERROR: timed out while running {exc.cmd}")
        with LOCK:
            JOB.update(
                state="error",
                error="A pipeline step took longer than 30 minutes and was stopped.",
                logs="\n\n".join(logs)[-16000:],
            )
    except Exception as exc:
        logs.append(f"ERROR: {exc}")
        with LOCK:
            JOB.update(
                state="error",
                error=str(exc),
                logs="\n\n".join(logs)[-16000:],
            )
        app.logger.exception("Toolbox Talk analysis failed")
    finally:
        try:
            if had_original and backup_path.exists():
                shutil.copy2(backup_path, PIPELINE_INPUT)
            elif not had_original and PIPELINE_INPUT.exists():
                PIPELINE_INPUT.unlink()
        except OSError:
            app.logger.exception("Could not restore the original input video")
        shutil.rmtree(backup_dir, ignore_errors=True)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/report", methods=["GET"])
def report_page():
    # A separate report.html is optional; don't crash if it hasn't been created.
    if (BASE_DIR / "templates" / "report.html").exists():
        return render_template("report.html")
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    uploaded = request.files.get("video")
    if uploaded is None or not uploaded.filename:
        return jsonify(error="Please choose a video file first."), 400

    safe_name = secure_filename(uploaded.filename)
    if not safe_name or not allowed_file(safe_name):
        return jsonify(error="Choose an MP4, MOV, or MKV video file."), 400

    missing = [name for name, _ in STAGES if not (BASE_DIR / name).is_file()]
    if missing:
        return jsonify(error="Pipeline files not found next to app.py: " + ", ".join(missing)), 500

    with LOCK:
        if JOB["state"] == "running":
            return jsonify(error="An analysis is already running. Please wait for it to finish."), 409
        JOB.update(
            state="running",
            stage=0,
            current_stage=STAGES[0][1],
            error=None,
            logs="",
            started=time.time(),
        )

    backup_dir = tempfile.mkdtemp(prefix="toolbox_backup_")
    had_original = PIPELINE_INPUT.exists()
    backup_path = Path(backup_dir) / "original_con.mp4"

    try:
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Remove results from any previous analysis.
        for result_file in (
            PARAMETERS_FILE,
            SCORING_FILE,
            REPORT_FILE,
        ):
            if result_file.exists():
                result_file.unlink()

        if had_original:
            shutil.copy2(PIPELINE_INPUT, backup_path)

        # Existing scripts expect data/input/con.mp4.
        uploaded.save(PIPELINE_INPUT)
    except Exception as exc:
        try:
            if had_original and backup_path.exists():
                shutil.copy2(backup_path, PIPELINE_INPUT)
            elif not had_original and PIPELINE_INPUT.exists():
                PIPELINE_INPUT.unlink()
        except OSError:
            app.logger.exception("Could not restore input after upload failure")
        shutil.rmtree(backup_dir, ignore_errors=True)
        with LOCK:
            JOB.update(state="error", error=f"Could not save uploaded video: {exc}")
        return jsonify(error=f"Could not save uploaded video: {exc}"), 500

    threading.Thread(
        target=run_pipeline,
        args=(backup_dir, had_original),
        daemon=True,
        name="toolbox-analysis",
    ).start()
    return jsonify(ok=True, message="Upload received. Analysis has started.")


@app.route("/api/status", methods=["GET"])
def api_status():
    with LOCK:
        started = JOB["started"]
        payload = {
            "state": JOB["state"],
            "stage": JOB["stage"],
            "current_stage": JOB["current_stage"],
            "stages": [label for _, label in STAGES],
            "error": JOB["error"],
            "elapsed": int(time.time() - started) if started else 0,
            "logs": JOB["logs"],
        }
    payload["has_previous"] = REPORT_FILE.exists()
    return jsonify(payload)


@app.route("/api/result", methods=["GET"])
def api_result():
    with LOCK:
        if JOB["state"] != "done":
            return jsonify(
                summary={},
                report="",
                logs=JOB["logs"],
                has_report=False,
                scoring={},
                parameters={},
            )

        logs = JOB["logs"]

    try:
        report = (
            REPORT_FILE.read_text(
                encoding="utf-8",
                errors="replace"
            )
            if REPORT_FILE.exists()
            else ""
        )
    except OSError:
        report = ""

    return jsonify(
        summary=get_summary(),
        report=report,
        logs=logs,
        has_report=bool(report),
        scoring=read_json_file(SCORING_FILE),
        parameters=read_json_file(PARAMETERS_FILE),
    )


@app.route("/download-report", methods=["GET"])
def download_report():
    if not REPORT_FILE.exists():
        return "No evaluation report is available yet. Run an analysis first.", 404
    return send_from_directory(
        OUTPUT_DIR,
        REPORT_FILE.name,
        as_attachment=True,
        download_name=REPORT_FILE.name,
    )


@app.errorhandler(413)
def file_too_large(_error):
    message = f"This video exceeds the {MAX_UPLOAD_MB} MB upload limit."
    if request.path == "/analyze" or request.path.startswith("/api/"):
        return jsonify(error=message), 413
    return message, 413


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    if isinstance(error, HTTPException):
        if request.path == "/analyze" or request.path.startswith("/api/"):
            return jsonify(error=f"{error.code} {error.name}: {error.description}"), error.code
        return error

    app.logger.exception("Unhandled website error")
    message = "An unexpected server error occurred. Check the VS Code terminal for details."
    if request.path == "/analyze" or request.path.startswith("/api/"):
        return jsonify(error=message), 500
    return message, 500


if __name__ == "__main__":
    print("Starting Toolbox Talk Evaluator website...")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)

"""
Neural ODE Training Dashboard — Flask backend.

Runs make targets as subprocesses, streams stdout/stderr via SSE,
parses training metrics from log lines, serves result images.
"""

import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

app = Flask(__name__, static_folder=None)

# ── Job registry ──────────────────────────────────────────────────────────────

jobs = {}  # job_id -> {target, status, start, end, proc, queue}
jobs_lock = threading.Lock()

MAKE_BIN = "make"
PYTHON = str(ROOT / ".venv" / "bin" / "python")

TARGETS = [
    {"id": "smoke",       "label": "smoke",       "desc": "Sanity check imports"},
    {"id": "toy-spiral",  "label": "toy-spiral",  "desc": "2D spiral fit (~30s)"},
    {"id": "toy-mnist",   "label": "toy-mnist",   "desc": "MNIST ODE-Net vs ResNet vs RK-Net"},
    {"id": "realworld",   "label": "realworld",   "desc": "Latent ODE on ETT time-series (50 epochs)"},
    {"id": "test-solver", "label": "test-solver", "desc": "Unit tests: ODE solver"},
    {"id": "test-adjoint","label": "test-adjoint","desc": "Unit tests: adjoint method"},
    {"id": "test-all",    "label": "test-all",    "desc": "All unit tests"},
    {"id": "clean",       "label": "clean",       "desc": "Remove results + caches"},
]


# ── Metric parsers ────────────────────────────────────────────────────────────

SPIRAL_RE  = re.compile(
    r"iter\s+(\d+)/(\d+)\s+loss=([\d.eE+\-]+)\s+nfe=(\d+)")
MNIST_RE   = re.compile(
    r"epoch\s+(\d+)/(\d+)\s+loss=([\d.eE+\-]+)\s+train=([\d.]+)%\s+test=([\d.]+)%\s+nfe=(-?\d+)\s+t=([\d.]+)s")
RW_RE      = re.compile(
    r"epoch\s+(\d+)/(\d+)\s+β=([\d.]+)\s+"
    r"train_loss=([\d.eE+\-]+)\s+recon=([\d.eE+\-]+)\s+kl=([\d.eE+\-]+)\s+"
    r"val_loss=([\d.eE+\-]+)\s+val_recon=([\d.eE+\-]+)\s+nfe=(\d+)\s+t=([\d.]+)s")
DEVICE_RE  = re.compile(r"Device:\s+(\S+)")
MODEL_RE   = re.compile(r"Training\s+(ODE-Net|RK-Net|ResNet)\s+\|")


def parse_metric(line: str):
    m = SPIRAL_RE.search(line)
    if m:
        return {"type": "spiral", "iter": int(m[1]), "total": int(m[2]),
                "loss": float(m[3]), "nfe": int(m[4])}
    m = MNIST_RE.search(line)
    if m:
        return {"type": "mnist", "epoch": int(m[1]), "total": int(m[2]),
                "loss": float(m[3]), "train_acc": float(m[4]),
                "test_acc": float(m[5]), "nfe": int(m[6]), "time": float(m[7])}
    m = RW_RE.search(line)
    if m:
        return {"type": "realworld", "epoch": int(m[1]), "total": int(m[2]),
                "beta": float(m[3]),
                "train_loss": float(m[4]), "recon": float(m[5]), "kl": float(m[6]),
                "val_loss": float(m[7]), "val_recon": float(m[8]),
                "nfe": int(m[9]), "time": float(m[10])}
    m = DEVICE_RE.search(line)
    if m:
        return {"type": "device", "device": m[1]}
    m = MODEL_RE.search(line)
    if m:
        return {"type": "model_start", "model": m[1]}
    return None


# ── Job runner ────────────────────────────────────────────────────────────────

def _run_job(job_id: str, target: str):
    q = jobs[job_id]["queue"]
    try:
        proc = subprocess.Popen(
            [MAKE_BIN, "-C", str(ROOT), target],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        with jobs_lock:
            jobs[job_id]["proc"] = proc

        for line in proc.stdout:
            line = line.rstrip("\n")
            metric = parse_metric(line)
            event = {"text": line, "metric": metric, "ts": time.time()}
            q.put(json.dumps(event))

        proc.wait()
        rc = proc.returncode
        with jobs_lock:
            jobs[job_id]["status"] = "done" if rc == 0 else "failed"
            jobs[job_id]["rc"] = rc
            jobs[job_id]["end"] = time.time()
        q.put(json.dumps({"text": f"\n[exit {rc}]", "metric": None,
                          "ts": time.time(), "done": True, "rc": rc}))
    except Exception as exc:
        with jobs_lock:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["end"] = time.time()
        q.put(json.dumps({"text": f"[error] {exc}", "metric": None,
                          "ts": time.time(), "done": True, "rc": -1}))
    finally:
        q.put(None)  # sentinel


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/api/targets")
def api_targets():
    return jsonify(TARGETS)


@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.get_json(force=True)
    target = body.get("target", "")
    valid = {t["id"] for t in TARGETS}
    if target not in valid:
        return jsonify({"error": "unknown target"}), 400

    job_id = str(uuid.uuid4())[:8]
    q: queue.Queue = queue.Queue()
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "target": target,
            "status": "running",
            "start": time.time(),
            "end": None,
            "rc": None,
            "proc": None,
            "queue": q,
        }

    t = threading.Thread(target=_run_job, args=(job_id, target), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/stop/<job_id>", methods=["POST"])
def api_stop(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404
    proc = job.get("proc")
    if proc and proc.poll() is None:
        proc.terminate()
        job["status"] = "stopped"
    return jsonify({"ok": True})


@app.route("/api/stream/<job_id>")
def api_stream(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "not found"}), 404

    q = job["queue"]

    def generate():
        while True:
            try:
                item = q.get(timeout=30)
            except queue.Empty:
                yield "data: {\"keepalive\": true}\n\n"
                continue
            if item is None:
                break
            yield f"data: {item}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


@app.route("/api/jobs")
def api_jobs():
    with jobs_lock:
        out = [{k: v for k, v in j.items() if k not in ("proc", "queue")}
               for j in jobs.values()]
    return jsonify(out)


@app.route("/api/results")
def api_results():
    if not RESULTS_DIR.exists():
        return jsonify([])
    files = sorted(RESULTS_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return jsonify([f.name for f in files])


@app.route("/api/results/<filename>")
def api_result_file(filename):
    return send_from_directory(str(RESULTS_DIR), filename)


@app.route("/")
@app.route("/<path:path>")
def serve_static(path="index.html"):
    return send_from_directory(str(Path(__file__).parent), path)


if __name__ == "__main__":
    import socket
    port = int(os.environ.get("PORT", 5001))
    while port < 5100:
        with socket.socket() as s:
            if s.connect_ex(("localhost", port)) != 0:
                break
        port += 1
    print(f"Dashboard: http://localhost:{port}")
    print(f"Root:      {ROOT}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

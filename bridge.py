import base64
import importlib.util
import io
import json
import queue
import threading
from pathlib import Path
# .venv\Scripts\python bridge.py
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, Response

BASE_DIR = Path(__file__).resolve().parent
QUARKSAT_PATH = BASE_DIR / "BWSI-CubeSat" / "src" / "QuarkSatGround.py"

def load_quarksat_module():
    spec = importlib.util.spec_from_file_location("quarksat_ground", QUARKSAT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load QuarkSatGround from {QUARKSAT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

quarksat_module = load_quarksat_module()
run_quarksat = quarksat_module.main


app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def decode_image(uploaded_file):
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Invalid image upload: {uploaded_file.filename}")
    return image


@app.get("/")
def root():
    return jsonify({"message": "Welcome to the QuarkSat Image Comparison API!"})

@app.get("/index")
def index():
    return app.send_static_file("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/compare", methods=["OPTIONS"])
def compare_options():
    return ("", 204)


@app.post("/compare")
def compare():
    image_a = request.files.get("image_a")
    image_b = request.files.get("image_b")

    if image_a is None or image_b is None:
        return jsonify({"error": "Both image_a and image_b are required."}), 400

    try:
        img1 = decode_image(image_a)
        img2 = decode_image(image_b)

        # Reset loading cache
        with _loading_lock:
            _loading_cache["image1"] = None
            _loading_cache["image2"] = None
            _loading_cache["warped1"] = None
            _loading_cache["warped2"] = None
            _loading_cache["hist1"] = None
            _loading_cache["hist2"] = None
        quarksat_module.set_state("initial")
        quarksat_module.sift_done_event.clear()
        quarksat_module.warp_done_event.clear()
        quarksat_module.histogram_done_event.clear()

        # Start keypoint map caching in background
        t1 = threading.Thread(target=_cache_keypoint_maps, args=(img1.copy(), img2.copy()), daemon=True)
        t2 = threading.Thread(target=_cache_warped_images, daemon=True)
        t3 = threading.Thread(target=_cache_histogram_images, daemon=True)
        t1.start(); t2.start(); t3.start()

        result = run_quarksat(img1, img2)

        # Wait for cache threads so /loading has all data before the response arrives
        t1.join(); t2.join(); t3.join()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Comparison failed: {exc}"}), 500

    success, encoded = cv2.imencode(".png", result)
    if not success:
        return jsonify({"error": "Failed to encode output image."}), 500

    return send_file(
        io.BytesIO(encoded.tobytes()),
        mimetype="image/png",
        as_attachment=False,
        download_name="result.png",
    )

# Cached keypoint images (populated by background thread during /compare)
_loading_cache = {"image1": None, "image2": None, "warped1": None, "warped2": None, "hist1": None, "hist2": None}
_loading_lock = threading.Lock()


def _cache_keypoint_maps(img1, img2):
    """Build keypoint-map PNGs in the background once SIFT finishes."""
    try:
        # Block until SIFT sets the event
        quarksat_module.sift_done_event.wait()
        result = quarksat_module.return_keypoint_map(img1, img2)
        if result is None:
            return
        result1, result2 = result
        ok1, enc1 = cv2.imencode(".png", result1)
        ok2, enc2 = cv2.imencode(".png", result2)
        if ok1 and ok2:
            with _loading_lock:
                _loading_cache["image1"] = base64.b64encode(enc1.tobytes()).decode()
                _loading_cache["image2"] = base64.b64encode(enc2.tobytes()).decode()
            quarksat_module._notify("cache_ready")
    except Exception:
        raise RuntimeError("Failed to cache keypoint maps for loading screen.")


def _cache_warped_images():
    """Encode warped + target images once warp is done."""
    try:
        quarksat_module.warp_done_event.wait()
        w_img = quarksat_module.cached_warped_img
        t_img = quarksat_module.cached_target_img
        if w_img is None or t_img is None:
            return
        ok1, enc1 = cv2.imencode(".png", w_img)
        ok2, enc2 = cv2.imencode(".png", t_img)
        if ok1 and ok2:
            with _loading_lock:
                _loading_cache["warped1"] = base64.b64encode(enc1.tobytes()).decode()
                _loading_cache["warped2"] = base64.b64encode(enc2.tobytes()).decode()
            quarksat_module._notify("cache_ready")
    except Exception:
        raise RuntimeError("Failed to cache warped images for loading screen.")


def _cache_histogram_images():
    """Encode histogram-matched + target images once histogram matching is done."""
    try:
        quarksat_module.histogram_done_event.wait()
        h_img = quarksat_module.cached_histmatched_img
        t_img = quarksat_module.cached_histtarget_img
        if h_img is None or t_img is None:
            return
        ok1, enc1 = cv2.imencode(".png", h_img)
        ok2, enc2 = cv2.imencode(".png", t_img)
        if ok1 and ok2:
            with _loading_lock:
                _loading_cache["hist1"] = base64.b64encode(enc1.tobytes()).decode()
                _loading_cache["hist2"] = base64.b64encode(enc2.tobytes()).decode()
            quarksat_module._notify("cache_ready")
    except Exception:
        raise RuntimeError("Failed to cache histogram images for loading screen.")


def _build_loading_data():
    """Build a snapshot of current state + all available cached data."""
    state = quarksat_module.state_machine
    INITIAL_STATES = {'compute_homography_matrix', 'sift_done_start_homography', 'refining_alignment'}
    REFINED_STATES = {'warping_image', 'matching_histograms', 'histogram_done', 'comparing_images'}

    data = {"state": state}

    if state in INITIAL_STATES:
        h = quarksat_module.initial_homography
        if h is not None:
            data["homography"] = h.tolist()
    elif state in REFINED_STATES:
        h = quarksat_module.refined_homography
        if h is not None:
            data["homography"] = h.tolist()

    with _loading_lock:
        for key in ("image1", "image2", "warped1", "warped2", "hist1", "hist2"):
            val = _loading_cache.get(key)
            if val:
                data[key] = val

    return data


@app.get("/events")
def events():
    """SSE stream that pushes a snapshot on every state / cache change."""
    q = quarksat_module.subscribe_state()

    def generate():
        try:
            while True:
                try:
                    q.get(timeout=30)
                except queue.Empty:
                    yield ": keepalive\n\n"
                    continue
                data = _build_loading_data()
                yield f"data: {json.dumps(data)}\n\n"
        except GeneratorExit:
            pass
        finally:
            quarksat_module.unsubscribe_state(q)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/loading")
def loading():
    state = quarksat_module.state_machine

    if state == "initial":
        return jsonify({"state": state, "message": "Waiting to start…"})
    elif state == "loading_start":
        return jsonify({"state": state, "message": "Detecting features (SIFT)…"})
    elif state == "compute_homography_matrix":
        resp = {"state": state, "message": "Computing homography matrix (RANSAC)…"}
        h_mat = quarksat_module.initial_homography
        if h_mat is not None:
            resp["homography"] = h_mat.tolist()
        return jsonify(resp)
    elif state == "sift_done_start_homography":
        with _loading_lock:
            img1 = _loading_cache.get("image1")
            img2 = _loading_cache.get("image2")
        resp = {"state": state, "message": "Computing homography…"}
        if img1:
            resp["image1"] = img1
        if img2:
            resp["image2"] = img2
        h_mat = quarksat_module.initial_homography
        if h_mat is not None:
            resp["homography"] = h_mat.tolist()
        return jsonify(resp)
    elif state == "refining_alignment":
        with _loading_lock:
            img1 = _loading_cache.get("image1")
            img2 = _loading_cache.get("image2")
        resp = {"state": state, "message": "Refining alignment (ECC)…"}
        if img1:
            resp["image1"] = img1
        if img2:
            resp["image2"] = img2
        h_mat = quarksat_module.initial_homography
        if h_mat is not None:
            resp["homography"] = h_mat.tolist()
        return jsonify(resp)
    elif state == "matching_histograms":
        resp = {"state": state, "message": "Matching histograms (resolving brightness and color)…"}
        h_mat = quarksat_module.refined_homography
        if h_mat is not None:
            resp["homography"] = h_mat.tolist()
        with _loading_lock:
            w1 = _loading_cache.get("warped1")
            w2 = _loading_cache.get("warped2")
        if w1:
            resp["warped1"] = w1
        if w2:
            resp["warped2"] = w2
        return jsonify(resp)
    elif state == "warping_image":
        resp = {"state": state, "message": "Warping image…"}
        h_mat = quarksat_module.refined_homography
        if h_mat is not None:
            resp["homography"] = h_mat.tolist()
        with _loading_lock:
            w1 = _loading_cache.get("warped1")
            w2 = _loading_cache.get("warped2")
        if w1:
            resp["warped1"] = w1
        if w2:
            resp["warped2"] = w2
        return jsonify(resp)
    elif state == "histogram_done":
        resp = {"state": state, "message": "Histograms/Colors matched"}
        h_mat = quarksat_module.refined_homography
        if h_mat is not None:
            resp["homography"] = h_mat.tolist()
        with _loading_lock:
            img1 = _loading_cache.get("image1")
            img2 = _loading_cache.get("image2")
            w1 = _loading_cache.get("warped1")
            w2 = _loading_cache.get("warped2")
            hi1 = _loading_cache.get("hist1")
            hi2 = _loading_cache.get("hist2")
        if img1:
            resp["image1"] = img1
        if img2:
            resp["image2"] = img2
        if w1:
            resp["warped1"] = w1
        if w2:
            resp["warped2"] = w2
        if hi1:
            resp["hist1"] = hi1
        if hi2:
            resp["hist2"] = hi2
        return jsonify(resp)
    elif state == "comparing_images":
        resp = {"state": state, "message": "Comparing pixels…"}
        h_mat = quarksat_module.refined_homography
        if h_mat is not None:
            resp["homography"] = h_mat.tolist()
        with _loading_lock:
            img1 = _loading_cache.get("image1")
            img2 = _loading_cache.get("image2")
            w1 = _loading_cache.get("warped1")
            w2 = _loading_cache.get("warped2")
            hi1 = _loading_cache.get("hist1")
            hi2 = _loading_cache.get("hist2")
        if img1:
            resp["image1"] = img1
        if img2:
            resp["image2"] = img2
        if w1:
            resp["warped1"] = w1
        if w2:
            resp["warped2"] = w2
        if hi1:
            resp["hist1"] = hi1
        if hi2:
            resp["hist2"] = hi2
        return jsonify(resp)
    else:
        return jsonify({"state": state, "message": f"Processing ({state})…"})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True, threaded=True)
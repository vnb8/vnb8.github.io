import importlib.util
import io
from pathlib import Path
# updated
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file

BASE_DIR = Path(__file__).resolve().parent
QUARKSAT_PATH = BASE_DIR / "BWSI-CubeSat" / "src" / "QuarkSatGround.py"


def load_quarksat_main():
    spec = importlib.util.spec_from_file_location("quarksat_ground", QUARKSAT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load QuarkSatGround from {QUARKSAT_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


run_quarksat = load_quarksat_main()

app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="")


def decode_image(uploaded_file):
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Invalid image upload: {uploaded_file.filename}")
    return image


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.post("/compare")
def compare():
    image_a = request.files.get("image_a")
    image_b = request.files.get("image_b")

    if image_a is None or image_b is None:
        return jsonify({"error": "Both image_a and image_b are required."}), 400

    try:
        img1 = decode_image(image_a)
        img2 = decode_image(image_b)
        result = run_quarksat(img1, img2)
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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
import os
import uuid
import cv2

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

MODEL_NAME = "yolo11n.pt"
model = YOLO(MODEL_NAME)

TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET", "POST"])
def index():
    error = ""
    result_image = ""
    original_name = ""
    object_counts = {}
    image_width = 0
    image_height = 0
    total_objects = 0

    if request.method == "POST":
        file = request.files.get("image")

        if file is None or file.filename == "":
            error = "Please choose an image."

        elif not allowed_file(file.filename):
            error = "Only JPG, JPEG and PNG images are supported."

        else:
            original_name = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{original_name}"
            input_path = os.path.join(UPLOAD_FOLDER, unique_name)

            try:
                # The folders are created above, so this save is safe.
                file.save(input_path)

                detection_results = model.predict(
                    source=input_path,
                    conf=0.35,
                    verbose=False
                )

                result = detection_results[0]

                # YOLO draws boxes and confidence labels automatically.
                annotated = result.plot()

                output_name = f"result_{uuid.uuid4().hex}.jpg"
                output_path = os.path.join(RESULT_FOLDER, output_name)

                saved = cv2.imwrite(output_path, annotated)

                if not saved:
                    raise RuntimeError("Could not save detection result.")

                image_height, image_width = annotated.shape[:2]

                for box in result.boxes:
                    class_id = int(box.cls[0])

                    if class_id in TARGET_CLASSES:
                        class_name = TARGET_CLASSES[class_id]
                        object_counts[class_name] = (
                            object_counts.get(class_name, 0) + 1
                        )

                total_objects = sum(object_counts.values())
                result_image = output_name

                # Remove uploaded source image after processing.
                try:
                    os.remove(input_path)
                except OSError:
                    pass

            except Exception as exc:
                error = (
                    "Object detection failed. "
                    "Please try another JPG, JPEG or PNG image."
                )
                print("Detection error:", exc)

                try:
                    if os.path.exists(input_path):
                        os.remove(input_path)
                except OSError:
                    pass

    return render_template(
        "index.html",
        error=error,
        result_image=result_image,
        original_name=original_name,
        object_counts=object_counts,
        image_width=image_width,
        image_height=image_height,
        total_objects=total_objects
    )


@app.route("/health")
def health():
    return "OK"


if __name__ == "__main__":
    app.run(debug=True)

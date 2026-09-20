# Object Detection and Tracking

CodeAlpha AI Internship - Task 4.

Upload a JPG, JPEG, or PNG image and use YOLO to detect supported objects. The result is displayed in a simple web UI with bounding boxes, confidence labels, object counts, and image details.

## Supported objects

- Person
- Bicycle
- Car
- Motorcycle
- Bus
- Truck

## Run locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

The YOLO model downloads automatically on the first run.

## Render

Build command:

```text
pip install -r requirements.txt
```

Start command:

```text
gunicorn app:app
```

The application processes uploaded images, so it can run as a web service without accessing a server webcam.

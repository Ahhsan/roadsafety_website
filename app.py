# app.py
from flask import Flask, render_template, request, jsonify, Response
from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import numpy as np

# For Accident Detection
from detection import AccidentDetectionModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'static/results'

# ---------------------------
# Ambulance Detection Model (YOLO)
# ---------------------------
ambulance_model = YOLO("best.pt")  # Your custom ambulance detection model

# ---------------------------
# Traffic Detection (YOLOv4 via OpenCV DNN)
# ---------------------------
YOLO_CFG = "yolov4.cfg"
YOLO_WEIGHTS = "yolov4.weights"
COCO_NAMES = "coco.names"

if not (os.path.exists(YOLO_CFG) and os.path.exists(YOLO_WEIGHTS) and os.path.exists(COCO_NAMES)):
    raise FileNotFoundError("One or more YOLOv4 files are missing.")

traffic_net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
with open(COCO_NAMES, "r") as f:
    classes = f.read().strip().split("\n")

VEHICLE_CLASSES = {"car", "bus", "truck", "motorbike"}
PEDESTRIAN_CLASSES = {"person"}

# ---------------------------
# Accident Detection Model (Keras)
# ---------------------------
accident_model = AccidentDetectionModel("model.json", "model_weights.h5")

# Ensure necessary directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# ---------------------------
# Routes for the pages
# ---------------------------
@app.route('/')
def home():
    return render_template('base.html')

@app.route('/ambulance')
def ambulance_page():
    return render_template('ambulance.html')

@app.route('/traffic')
def traffic_page():
    return render_template('traffic.html')

@app.route('/accident')
def accident_page():
    return render_template('accident.html')

# ---------------------------
# Live Processing for Ambulance Detection
# ---------------------------
@app.route('/upload_live_video', methods=['POST'])
def upload_live_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400
    video = request.files['video']
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{video.filename}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)
    return jsonify({"video_filename": filename})

@app.route('/detect_live', methods=['GET'])
def detect_live():
    video_filename = request.args.get('video')
    if not video_filename:
        return "No video specified", 400
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    return Response(generate_live_ambulance_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_live_ambulance_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = ambulance_model.predict(frame, conf=0.80)
        annotated_frame = results[0].plot() if results else frame
        ret2, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()
    os.remove(video_path)

# ---------------------------
# Live Processing for Accident Detection
# ---------------------------
@app.route('/upload_live_accident', methods=['POST'])
def upload_live_accident():
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400
    video = request.files['video']
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{video.filename}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)
    return jsonify({"video_filename": filename})

@app.route('/detect_live_accident', methods=['GET'])
def detect_live_accident():
    video_filename = request.args.get('video')
    if not video_filename:
        return "No video specified", 400
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
    return Response(generate_live_accident_frames(video_path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_live_accident_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(rgb_frame, (250, 250))
        pred, prob = accident_model.predict_accident(roi[np.newaxis, :])
        if pred == "Accident":
            prob_val = round(prob[0][0] * 100, 2)
            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"{pred} {prob_val}%", (20, 30), font, 1, (255, 255, 0), 2)
        ret2, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()
    os.remove(video_path)

# ---------------------------
# Traffic Detection Endpoints (Vehicle & Pedestrian)
# (These remain unchanged.)
# ---------------------------
def process_traffic_image(image_path, target_classes):
    image = cv2.imread(image_path)
    if image is None:
        return None, 0
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=(416, 416),
                                  swapRB=True, crop=False)
    traffic_net.setInput(blob)
    output_layers = traffic_net.getUnconnectedOutLayersNames()
    detections = traffic_net.forward(output_layers)
    boxes = []
    confidences = []
    class_ids = []
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                label = classes[class_id]
                if label in target_classes:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    count = 0
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            count += 1
            color = (255, 0, 0) if label in VEHICLE_CLASSES else (0, 255, 0)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, f"{label}: {confidence:.2f}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image, count

@app.route('/detect_vehicle', methods=['POST'])
def detect_vehicle():
    if 'vehicle' not in request.files:
        return jsonify({"error": "No vehicle image uploaded"}), 400
    try:
        vehicle_file = request.files['vehicle']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], vehicle_file.filename)
        vehicle_file.save(image_path)
        annotated_image, vehicle_count = process_traffic_image(image_path, VEHICLE_CLASSES)
        if annotated_image is None:
            return jsonify({"error": "Invalid image file"}), 400
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_filename = f"annotated_vehicle_{timestamp}.jpg"
        output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
        cv2.imwrite(output_path, annotated_image)
        os.remove(image_path)
        return jsonify({
            "vehicle_count": vehicle_count,
            "annotated_image": f"/static/results/{output_filename}"
        })
    except Exception as e:
        print(f"Error processing vehicle image: {str(e)}")
        return jsonify({"error": "Error processing vehicle image"}), 500

@app.route('/detect_pedestrian', methods=['POST'])
def detect_pedestrian():
    if 'pedestrian' not in request.files:
        return jsonify({"error": "No pedestrian image uploaded"}), 400
    try:
        pedestrian_file = request.files['pedestrian']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], pedestrian_file.filename)
        pedestrian_file.save(image_path)
        annotated_image, pedestrian_count = process_traffic_image(image_path, PEDESTRIAN_CLASSES)
        if annotated_image is None:
            return jsonify({"error": "Invalid image file"}), 400
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_filename = f"annotated_pedestrian_{timestamp}.jpg"
        output_path = os.path.join(app.config['RESULTS_FOLDER'], output_filename)
        cv2.imwrite(output_path, annotated_image)
        os.remove(image_path)
        return jsonify({
            "pedestrian_count": pedestrian_count,
            "annotated_image": f"/static/results/{output_filename}"
        })
    except Exception as e:
        print(f"Error processing pedestrian image: {str(e)}")
        return jsonify({"error": "Error processing pedestrian image"}), 500

if __name__ == '__main__':
    app.run(debug=True)

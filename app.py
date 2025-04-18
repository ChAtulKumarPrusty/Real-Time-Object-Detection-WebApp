from flask import Flask, render_template, Response, request, jsonify
import cv2 as cv
import numpy as np

app = Flask(__name__)

# Global variables for object detection
cap_index = 0  # Default camera index
cap = cv.VideoCapture(cap_index)
confThreshold = 0.5  # Confidence threshold (default)
whT = 320  # Width and height for YOLO model input
nmsThreshold = 0.3  # Non-Maximum Suppression threshold

# Load YOLO model
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Load class names
with open('coco.names', 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Function to process and detect objects
def findObjects(outputs, img, confThreshold):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:  # Apply confidence threshold
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    if len(indices) > 0:
        for i in indices.flatten():
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                       (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

# Generate frames from the video capture
def generate_frames():
    global cap, confThreshold  # Use the dynamic confThreshold
    while True:
        success, img = cap.read()
        if not success:
            break

        # Preprocess frame for object detection
        blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)

        # Perform object detection with the current confThreshold
        findObjects(outputs, img, confThreshold)

        # Encode frame to JPEG format for the video stream
        ret, buffer = cv.imencode('.jpg', img)
        frame = buffer.tobytes()

        # Yield the frame to the browser
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to stream video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to update the confidence threshold
@app.route('/set_confidence', methods=['POST'])
def set_confidence():
    global confThreshold
    # Get confidence value from the frontend (slider)
    try:
        confThreshold = float(request.form['confidence'])
        print(f"Updated confidence threshold: {confThreshold}")
        return jsonify(success=True)
    except ValueError:
        return jsonify(success=False)

# Home route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

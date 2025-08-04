import cv2
import numpy as np
import streamlit as st
import tempfile
import time
import os
import urllib.request

# ===== CONFIG =====
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.3

# ===== YOLO FILES =====
YOLO_CFG = "yolov3-tiny.cfg"
YOLO_WEIGHTS = "yolov3-tiny.weights"
COCO_NAMES = "coco.names"

# ===== DOWNLOAD FILES IF MISSING =====
def download_file(url, filename):
    if not os.path.exists(filename):
        st.write(f"📥 Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        st.write(f"✅ {filename} downloaded!")

download_file("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg", YOLO_CFG)
download_file("https://pjreddie.com/media/files/yolov3-tiny.weights", YOLO_WEIGHTS)
download_file("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", COCO_NAMES)

# ===== LOAD LABELS =====
labels = open(COCO_NAMES).read().strip().split("\n")

# ===== LOAD YOLO MODEL =====
net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
layer_names = net.getLayerNames()
layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# ===== DETECTION FUNCTION =====
def detect_objects(image):
    if image is None:
        st.error("❌ No image found for detection.")
        return None

    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layer_outputs = net.forward(layer_names)
    end = time.time()

    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(image, text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

# ===== STREAMLIT UI =====
st.title("🔍 YOLOv3-Tiny Object Detection")

option = st.radio("Choose Input Method", ("Upload Image", "Webcam"))

# ===== UPLOAD IMAGE =====
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()

        image = cv2.imread(temp_file.name)

        if image is None:
            st.error("❌ Could not read the uploaded file. Please upload a valid JPG/PNG.")
            st.stop()

        result = detect_objects(image)
        if result is not None:
            st.image(result, channels="BGR")

# ===== WEBCAM =====
elif option == "Webcam":
    run = st.checkbox("Run Webcam")
    cap = cv2.VideoCapture(0)
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to access webcam.")
            break

        result = detect_objects(frame)
        if result is not None:
            st.image(result, channels="BGR")

    cap.release()

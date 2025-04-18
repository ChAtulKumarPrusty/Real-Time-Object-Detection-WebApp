# 🔍 Real-Time Object Detection Web App using YOLOv3 and Flask

This project is a real-time object detection system built with **Flask**, **OpenCV**, and **YOLOv3**. The application streams video from your webcam, detects objects using the YOLOv3 model, and displays the detections in the browser. It also provides a dynamic slider to adjust the confidence threshold for object detection.

---

## 🚀 Features

- Real-time object detection using YOLOv3
- Live video feed in the browser
- Adjustable confidence threshold via slider
- Easy-to-use web interface
- Built with Flask and OpenCV

---

## 📸 Demo

> *You can add a GIF or a video demo here.*

---

## 🧠 Tech Stack

- **Python**
- **Flask** – for the web server
- **OpenCV** – for video processing and object detection
- **YOLOv3** – for deep learning-based object detection

---

## 📁 Project Structure

📂 project-root ├── yolov3.cfg ├── yolov3.weights ├── coco.names ├── app.py ├── templates/ │ └── index.html ├── static/ │ └── (optional assets like CSS or JS) └── README.md

---

## ⚙️ Setup Instructions

### 1. Clone the repository

git clone https://github.com/ChAtulKumarPrusty/Real-Time-Object-Detection-WebApp.git
cd your-repo-name

Install dependencies
pip install -r requirements.txt
If you don't have a requirements.txt, install directly:
pip install flask opencv-python numpy


## 📦 Download YOLOv3 Files

Download the following files and place them in the **root directory** of this project:
- **YOLOv3 Config (`yolov3.cfg`)**  
  🔗 [Download yolov3.cfg](https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg)
- **YOLOv3 Weights (`yolov3.weights`)**  
  🔗 [Download yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
- **COCO Class Names (`coco.names`)**  
  🔗 [Download coco.names](https://github.com/pjreddie/darknet/raw/master/data/coco.names)


▶️ Run the App
python app.py
Open your browser and go to: http://127.0.0.1:5000

🧩 Customization
Adjust the default confidence threshold in app.py:
confThreshold = 0.5
Change the YOLO input size:
whT = 320  # Can be 320, 416, or 608 for YOLO

🙌 Credits
YOLOv3 by Joseph Redmon
OpenCV DNN module
Flask Web Framework

🤝 Connect
Feel free to connect or contribute to the project!
Made with ❤️ by Ch Atul Kumar Prusty


---

Let me know if you'd like me to add:

- A `requirements.txt` file
- A demo GIF/YouTube video badge
- Deployment on Heroku/Render tutorial steps

Happy uploading! 🚀

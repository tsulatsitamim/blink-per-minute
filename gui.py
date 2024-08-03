import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QSlider
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import os
import time
import shutil
import subprocess
import csv
from collections import deque

from moviepy.editor import VideoFileClip

def get_video_length(file_path):
    clip = VideoFileClip(file_path)
    return clip.duration

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
else:
    bundle_dir = os.path.abspath(os.path.dirname(__file__))

blinks_dir = os.path.join(bundle_dir, 'blinks')
os.makedirs(blinks_dir, exist_ok=True)

class BlinkDetectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        self.video_path = ""
        self.video_duration = 0
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
        self.detector = dlib.get_frontal_face_detector()
        dat_file_path = os.path.join(bundle_dir, 'shape_predictor_68_face_landmarks.dat')
        self.predictor = dlib.shape_predictor(dat_file_path)
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        
        self.EYE_AR_THRESH = 0.15
        self.EYE_AR_CONSEC_FRAMES = 3
        self.COUNTER = 0
        self.TOTAL = 0
        self.total_frames = 0

        self.blink_times = deque()
        self.csv_file = None
        self.csv_writer = None

    def delete_existing_output(self):
        if os.path.exists(blinks_dir):
            for file in os.listdir(blinks_dir):
                file_path = os.path.join(blinks_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

    def browse_output(self):
        if os.path.exists(blinks_dir):
            if sys.platform == "win32":
                os.startfile(blinks_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", blinks_dir])
            else:
                subprocess.Popen(["xdg-open", blinks_dir])
        else:
            print("Output directory does not exist.")

    def initUI(self):
        self.setWindowTitle('Blink Detector')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # Upload button
        self.upload_btn = QPushButton('Upload Video')
        self.upload_btn.clicked.connect(self.upload_video)
        layout.addWidget(self.upload_btn)

        # Threshold slider
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel('Threshold:'))
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(1, 50)
        self.threshold_slider.setValue(15)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.threshold_slider)
        self.threshold_label = QLabel('0.15')
        threshold_layout.addWidget(self.threshold_label)
        layout.addLayout(threshold_layout)

        # Start button
        self.start_btn = QPushButton('Start')
        self.start_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.start_btn)

        # Video display
        self.video_label = QLabel()
        layout.addWidget(self.video_label)

        # add progress
        self.frame_progress_label = QLabel('Frame: 0 / 0')
        layout.addWidget(self.frame_progress_label)

        # Browse output button
        self.browse_output_btn = QPushButton('Browse Output')
        self.browse_output_btn.clicked.connect(self.browse_output)
        layout.addWidget(self.browse_output_btn)

        self.setLayout(layout)

    def upload_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi)")
        if self.video_path:
            self.upload_btn.setText(f"Uploaded: {os.path.basename(self.video_path)}")

    def update_threshold(self):
        self.EYE_AR_THRESH = self.threshold_slider.value() / 100
        self.threshold_label.setText(f'{self.EYE_AR_THRESH:.2f}')

    def start_processing(self):
        if not self.video_path:
            return
        
        self.delete_existing_output()
        self.cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_duration = get_video_length(self.video_path)

        # Initialize CSV file
        csv_path = os.path.join(blinks_dir, 'blink_data.csv')
        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Time', 'Total Blinks', 'Blinks in Last 60s'])

        # Update every 30 ms
        self.timer.start(30) 

    def calculate_blinks_last_60s(self, current_time):
        while self.blink_times and current_time - self.blink_times[0] > 60:
            self.blink_times.popleft()
        return len(self.blink_times)
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            if self.csv_file:
                self.csv_file.close()
            return
        
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.frame_progress_label.setText(f'Frame: {current_frame} / {self.total_frames}')

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        rects = self.detector(gray, 0)

        for rect in rects:
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0


            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            current_time = round((current_frame / self.total_frames) * self.video_duration)
            minutes, seconds = divmod(current_time, 60)
            time_str = f"{minutes:02d}:{seconds:02d}"

            blink_detected = False
            if ear < self.EYE_AR_THRESH:
                self.COUNTER += 1
            else:
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    self.TOTAL += 1
                    self.blink_times.append(current_time)
                    blink_detected = True
                self.COUNTER = 0

            blinks_last_60s = self.calculate_blinks_last_60s(current_time)

            cv2.putText(frame, f"Time: {time_str}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Blinks: {self.TOTAL}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Blinks (60s): {blinks_last_60s}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if blink_detected:
                self.csv_writer.writerow([time_str, self.TOTAL, blinks_last_60s])
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                img_filename = os.path.join(blinks_dir, f"blink_{timestamp}.png")
                cv2.imwrite(img_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if ear < self.EYE_AR_THRESH:
                self.COUNTER += 1
            else:
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    self.TOTAL += 1
                    # Save the frame as an image when blink is detected
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    img_filename = os.path.join(blinks_dir, f"blink_{timestamp}.png")
                    cv2.imwrite(img_filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                self.COUNTER = 0

        # Resize the frame to width 450 while maintaining aspect ratio
        height, width, _ = frame.shape
        aspect_ratio = height / width
        new_width = 900
        new_height = int(new_width * aspect_ratio)
        resized_frame = cv2.resize(frame, (new_width, new_height))



        h, w, ch = resized_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(resized_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_image))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = BlinkDetectorApp()
    ex.show()
    sys.exit(app.exec_())

#!/usr/bin/env python3
"""
Simple Human Detection using YOLOv5n + rpicam-still
+ Firebase Integration
Optimized for Raspberry Pi Zero 2W (2025)
"""

import cv2
import time
import subprocess
import os
from ultralytics import YOLO
from datetime import datetime

# 🔥 NEW: Firebase imports
import firebase_admin
from firebase_admin import credentials, db

class RPiCamStillHumanDetector:
    def __init__(self):
        print("🚀 Initializing Human Detector using rpicam-still...")

        # Load YOLO model
        print("📥 Loading YOLOv5n model...")
        self.model = YOLO("yolov5n.pt")
        self.human_class_id = 0  # COCO 'person' class
        self.confidence_threshold = 0.5
        self.log_file = "human_detections.txt"
        self.libcamera_cmd = "rpicam-still"
        print("✅ Detector initialized with rpicam-still")

        # 🔥 NEW: Initialize Firebase
        print("🌐 Connecting to Firebase...")
        cred = credentials.Certificate("/home/beris/work/FRED/serviceAccountKey.json")
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://fred-90e16-default-rtdb.asia-southeast1.firebasedatabase.app/'
        })
        self.firebase_ref = db.reference('human_detections')
        print("✅ Connected to Firebase")

    def capture_frame(self):
        """Capture a frame using rpicam-still"""
        filename = f"temp_{int(time.time()*1000)}.jpg"
        try:
            subprocess.run(
                [self.libcamera_cmd, "-o", filename, "--timeout", "0.5"],
                capture_output=True,
                timeout=2
            )
            if os.path.exists(filename):
                frame = cv2.imread(filename)
                os.remove(filename)
                return frame
        except Exception as e:
            print(f"⚠️ Capture error: {e}")
        return None

    def detect_humans(self, frame):
        """Detect humans in a frame"""
        if frame is None:
            return 0, []

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            results = self.model(frame_rgb, verbose=False)
            human_count = 0
            human_boxes = []

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    if int(box.cls[0]) == self.human_class_id and float(box.conf[0]) >= self.confidence_threshold:
                        human_count += 1
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0])
                        human_boxes.append((x1, y1, x2, y2, confidence))

            return human_count, human_boxes

        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            return 0, []

    def log_detection(self, count):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp}: {count} humans detected\n"
        with open(self.log_file, "a") as f:
            f.write(log_entry)
        print(f"📝 {log_entry.strip()}")

        # 🔥 NEW: Upload to Firebase
        try:
            self.firebase_ref.push({
                'timestamp': timestamp,
                'count': count
            })
            print("📤 Sent to Firebase")
        except Exception as e:
            print(f"⚠️ Firebase error: {e}")

    def run_detection_loop(self, duration=300):
        print(f"🔍 Starting detection for {duration} seconds (rpicam-still)...")
        start_time = time.time()
        frame_count = 0
        total_detections = 0

        try:
            while (time.time() - start_time) < duration:
                frame = self.capture_frame()
                if frame is not None:
                    frame_count += 1
                    human_count, boxes = self.detect_humans(frame)

                    if human_count > 0:
                        total_detections += human_count
                        self.log_detection(human_count)

                time.sleep(0.2)

        except KeyboardInterrupt:
            print("\n👋 Detection stopped by user")

        finally:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"\n📊 Summary:")
            print(f"Duration: {elapsed:.1f}s | Frames: {frame_count} | Detections: {total_detections} | Avg FPS: {fps:.1f}")

if __name__ == "__main__":
    detector = RPiCamStillHumanDetector()
    detector.run_detection_loop(300)

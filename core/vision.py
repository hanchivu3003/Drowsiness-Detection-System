import cv2
import threading
from ultralytics import YOLO


class VisionSystem:
    def __init__(self, model_path, src=0, conf=0.5):

        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()

        self.running = False
        self.lock = threading.Lock()

        self.model = YOLO(model_path)
        self.conf = conf

    def start(self):
        if self.running:
            return self

        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        return results

    def stop(self):
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
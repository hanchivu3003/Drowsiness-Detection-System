from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QWidget,
    QVBoxLayout, QHBoxLayout, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QTabWidget, QSlider, QCheckBox, QListWidget
import cv2

from core.vision import VisionSystem
from core.drowsiness import DrowsinessDetector

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class StatisticsTab(QWidget):
    def __init__(self, detector, parent=None):
        super().__init__(parent)
        self.detector = detector
        layout = QVBoxLayout()
        self.setLayout(layout)

        # --- 1. Biểu đồ Real-time (Trạng thái bất thường theo thời gian) ---
        layout.addWidget(QLabel("Real-time Anomalies (Live):"))
        self.line_fig = Figure(figsize=(5, 4), dpi=70)
        self.line_canvas = FigureCanvas(self.line_fig)
        self.ax_line = self.line_fig.add_subplot(111)
        self.ax_line.set_ylim(-0.5, 2.5)
        self.ax_line.set_yticks([0, 1, 2])
        self.ax_line.set_yticklabels(['Safe', 'Warning', 'Danger'])

        # Dữ liệu cho biểu đồ đường
        self.x_data = list(range(50))
        self.y_data = [0] * 50
        self.line_plot, = self.ax_line.plot(self.x_data, self.y_data, 'r-')
        layout.addWidget(self.line_canvas)

        # --- 2. Biểu đồ tròn (Tỉ lệ vi phạm trong phiên làm việc) ---
        layout.addWidget(QLabel("Violation Distribution (%):"))
        self.pie_fig = Figure(figsize=(5, 4), dpi=70)
        self.pie_canvas = FigureCanvas(self.pie_fig)
        self.ax_pie = self.pie_fig.add_subplot(111)
        layout.addWidget(self.pie_canvas)

        # Bộ đếm thống kê cho biểu đồ tròn
        self.stats_counts = {"DANGER": 0, "WARNING": 0, "SAFE": 0}

    def update_charts(self, current_state):
        # Cập nhật dữ liệu biểu đồ đường
        state_map = {"SAFE": 0, "WARNING": 1, "DANGER": 2}
        new_val = state_map.get(current_state, 0)

        self.y_data.pop(0)
        self.y_data.append(new_val)
        self.line_plot.set_ydata(self.y_data)
        self.ax_line.relim()
        self.line_canvas.draw()

        # Cập nhật dữ liệu biểu đồ tròn
        self.stats_counts[current_state] += 1
        self.draw_pie_chart()

    def draw_pie_chart(self):
        self.ax_pie.clear()
        labels = list(self.stats_counts.keys())
        sizes = list(self.stats_counts.values())
        colors = ['#ff4d4d', '#ffa64d', '#4dff4d']  # Red, Orange, Green

        if sum(sizes) > 0:
            self.ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
            self.ax_pie.axis('equal')
        self.pie_canvas.draw()


class SettingsTab(QWidget):
    def __init__(self, detector, parent=None):
        super().__init__(parent)
        self.detector = detector  # Truyền instance của DrowsinessDetector vào để chỉnh sửa trực tiếp

        layout = QVBoxLayout()
        self.setLayout(layout)

        # 1. Điều chỉnh thời gian nhắm mắt (Eye Closed Threshold)
        # Giới hạn nhỏ nhất là 1s
        layout.addWidget(QLabel("Eye Closed Threshold (1.0s - 3.0s):"))
        self.eye_slider = QSlider(Qt.Horizontal)
        self.eye_slider.setMinimum(10)  # Tương ứng 1.0s
        self.eye_slider.setMaximum(30)  # Tương ứng 3.0s
        self.eye_slider.setValue(int(self.detector.EYE_CLOSED_THRESHOLD * 10))
        self.eye_label = QLabel(f"{self.detector.EYE_CLOSED_THRESHOLD}s")

        eye_layout = QHBoxLayout()
        eye_layout.addWidget(self.eye_slider)
        eye_layout.addWidget(self.eye_label)
        layout.addLayout(eye_layout)
        self.eye_slider.valueChanged.connect(self.update_eye_threshold)

        # 2. Điều chỉnh thời gian Delay giữa các trạng thái
        layout.addWidget(QLabel("Danger State Delay (1.0s - 5.0s):"))
        self.delay_slider = QSlider(Qt.Horizontal)
        self.delay_slider.setMinimum(10)
        self.delay_slider.setMaximum(50)
        self.delay_slider.setValue(int(self.detector.DANGER_DELAY * 10))
        self.delay_label = QLabel(f"{self.detector.DANGER_DELAY}s")

        delay_layout = QHBoxLayout()
        delay_layout.addWidget(self.delay_slider)
        delay_layout.addWidget(self.delay_label)
        layout.addLayout(delay_layout)
        self.delay_slider.valueChanged.connect(self.update_delay_threshold)

        # 3. Âm thanh cảnh báo
        self.sound_checkbox = QCheckBox("Enable Alarm Sound")
        self.sound_checkbox.setChecked(True)  # Mặc định bật
        layout.addWidget(self.sound_checkbox)

        layout.addWidget(QLabel("Alarm Volume:"))
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setValue(70)
        layout.addWidget(self.vol_slider)

        # 4. Chế độ tối (Dark Mode)
        self.dark_mode_checkbox = QCheckBox("Dark Mode")
        layout.addWidget(self.dark_mode_checkbox)
        self.dark_mode_checkbox.stateChanged.connect(self.toggle_dark_mode)

        layout.addStretch()

    def update_eye_threshold(self, value):
        val = value / 10.0
        self.detector.EYE_CLOSED_THRESHOLD = val
        self.eye_label.setText(f"{val}s")

    def update_delay_threshold(self, value):
        val = value / 10.0
        self.detector.DANGER_DELAY = val
        self.delay_label.setText(f"{val}s")

    def toggle_dark_mode(self, state):
        if state == Qt.Checked:
            # Bạn có thể gọi một hàm ở MainWindow để thay đổi StyleSheet của toàn App
            self.window().setStyleSheet("background-color: #2b2b2b; color: white;")
        else:
            self.window().setStyleSheet("")  # Reset về mặc định

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.vision = VisionSystem("E:\\Drowsiness-Detection\\models\\best1.pt").start()
        self.drowsiness = DrowsinessDetector()

        self.setWindowTitle("Drowsiness Detection")
        self.setGeometry(100, 100, 1200, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        self.status_label = QLabel("AWAKE")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: green;
            padding: 10px;
            border: 2px solid black;
        """)
        main_layout.addWidget(self.status_label)


        body_layout = QHBoxLayout()
        main_layout.addLayout(body_layout)


        self.camera_label = QLabel()
        self.camera_label.setFixedSize(800, 500)
        self.camera_label.setStyleSheet("background-color: black;")
        self.camera_label.setAlignment(Qt.AlignCenter)

        body_layout.addWidget(self.camera_label)

        right_panel = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tabs.setFixedSize(350, 450)

        # Tạo các Widget cho từng Tab
        self.tab_settings = QWidget()
        self.tab_settings = SettingsTab(self.drowsiness)  # Truyền detector vào
        self.tabs.addTab(self.tab_settings, "Settings")
        # self.tab_stats = QWidget()
        self.tab_stats = StatisticsTab(self.drowsiness)
        self.tab_logs = QListWidget()

        self.tabs.addTab(self.tab_logs, "Logs")
        self.tabs.addTab(self.tab_stats, "Stats")
        self.tabs.addTab(self.tab_settings, "Settings")


        # Thêm tabs vào right_panel thay cho chart_label cũ
        right_panel.addWidget(self.tabs)

        # ==== FPS ====
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("font-size: 16px;")
        right_panel.addWidget(self.fps_label)

        # ==== BUTTONS ====
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")

        self.start_btn.setStyleSheet("padding: 10px;")
        self.stop_btn.setStyleSheet("padding: 10px;")

        right_panel.addWidget(self.start_btn)
        right_panel.addWidget(self.stop_btn)

        right_panel.addStretch()

        body_layout.addLayout(right_panel)

        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)

    def start_camera(self):
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        self.vision.stop()

    def get_color(cls_id):
        import random
        random.seed(cls_id)
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

    def update_frame(self):
        import time

        ret, frame = self.vision.read()
        if not ret:
            return

        if not hasattr(self, "prev_time"):
            self.prev_time = time.time()

        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        self.fps_label.setText(f"FPS: {int(fps)}")

        results = self.vision.detect(frame)

        if len(results) > 0:
            boxes = results[0].boxes

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    names = results[0].names
                    label = f"{names[cls]} {conf:.2f}"

                    if cls in [1, 8]:
                        color = (0, 0, 255)  # Red (BGR)
                    elif cls in [3, 5, 6, 7]:
                        color = (0, 255, 255)  # Yellow (BGR)
                    else:
                        color = (0, 255, 0)  # Green (BGR)

                    label = f"{names[cls]} {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2)

        state, message = self.drowsiness.update(results)
        self.set_status(state, message)
        self.tab_stats.update_charts(state)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w

        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        self.camera_label.setPixmap(pixmap)

    def set_status(self, state, message):
        # Thiết lập màu sắc và nội dung dựa trên state
        if state == "DANGER":
            color = "red"
            display_text = f"STATUS: {message}"
        elif state == "WARNING":
            color = "orange"
            display_text = f"STATUS: {message}"
        else:  # Trạng thái SAFE
            color = "green"
            display_text = f"STATUS: {message}"

        # Cập nhật label duy nhất một lần để tránh lag giao diện
        self.status_label.setText(display_text)
        self.status_label.setStyleSheet(f"""
            font-size: 28px;
            font-weight: bold;
            color: {color};
            padding: 10px;
            border: 2px solid {color};
            background-color: #f0f0f0;
        """)
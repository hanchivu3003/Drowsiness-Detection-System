from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QWidget,
    QVBoxLayout, QHBoxLayout, QFrame
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QTabWidget, QSlider, QCheckBox, QListWidget
from datetime import datetime
import cv2

from core.vision import VisionSystem
from core.drowsiness import DrowsinessDetector

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import time


class StatisticsTab(QWidget):
    def __init__(self, detector, parent=None):
        super().__init__(parent)
        self.detector = detector
        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel("Real-time Anomalies (Live):"))
        self.line_fig = Figure(figsize=(5, 4), dpi=70)
        self.line_canvas = FigureCanvas(self.line_fig)
        self.ax_line = self.line_fig.add_subplot(111)
        self.ax_line.set_ylim(2.5, -0.5)
        self.ax_line.set_yticks([0, 1, 2])
        self.ax_line.set_yticklabels(['Safe', 'Warning', 'Danger'])

        # line chart
        self.times = []  # lưu timestamp
        self.states = []  # lưu state (0,1,2)
        # line nối
        self.line_plot, = self.ax_line.plot([], [], linewidth=1)
        # scatter (các chấm)
        self.scatter = self.ax_line.scatter([], [], s=50)

        # time chart
        self.start_time = time.time()
        self.ax_line.set_xlim(0, 60)  # 60 giây gần nhất
        self.ax_line.set_xlabel("Time (s)")


        layout.addWidget(self.line_canvas)

        layout.addWidget(QLabel("Violation Distribution (%):"))
        self.pie_fig = Figure(figsize=(5, 4), dpi=70)
        self.pie_canvas = FigureCanvas(self.pie_fig)
        self.ax_pie = self.pie_fig.add_subplot(111)
        layout.addWidget(self.pie_canvas)

        self.stats_counts = {"DANGER": 0, "WARNING": 0, "SAFE": 0}

    def update_charts(self, current_state):

        state_map = {"SAFE": 0, "WARNING": 1, "DANGER": 2}
        color_map = {0: "green", 1: "orange", 2: "red"}

        now = time.time() - self.start_time
        new_val = state_map.get(current_state, 0)

        # ===== Logic 1 phút =====
        if len(self.states) > 0:
            last_time = self.times[-1]
            last_state = self.states[-1]

            # nếu cùng trạng thái và chưa quá 60s → update chấm cuối
            if new_val == last_state and (now - last_time) < 60:
                self.times[-1] = now
            else:
                self.times.append(now)
                self.states.append(new_val)
        else:
            self.times.append(now)
            self.states.append(new_val)

        # ===== Giữ dữ liệu trong 60s =====
        while len(self.times) > 0 and (now - self.times[0]) > 60:
            self.times.pop(0)
            self.states.pop(0)

        # update line
        self.line_plot.set_data(self.times, self.states)

        colors = [color_map[s] for s in self.states]
        self.scatter.set_offsets(np.c_[self.times, self.states])
        self.scatter.set_color(colors)
        # x
        self.ax_line.set_xlim(max(0, now - 60), now)
        self.ax_line.set_ylim(-0.5, 2.5)

        self.line_canvas.draw_idle()

        # ===== Pie chart =====
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
        self.pie_canvas.draw_idle()


class SettingsTab(QWidget):
    def __init__(self, detector, parent=None):
        super().__init__(parent)
        self.detector = detector  # Truyền instance của DrowsinessDetector vào để chỉnh sửa trực tiếp

        layout = QVBoxLayout()
        self.setLayout(layout)

        # 1. Điều chỉnh thời gian nhắm mắt (Eye Closed Threshold)
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
        self.vol_slider.valueChanged.connect(self.update_volume)
        layout.addWidget(self.vol_slider)
        self.update_volume(self.vol_slider.value())

        self.sound_checkbox.stateChanged.connect(self.on_sound_toggled)

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

    def update_volume(self, value):

        volume_level = value / 100.0

        if hasattr(self.detector, 'sound_danger'):
            self.detector.sound_danger.set_volume(volume_level)

        if hasattr(self.detector, 'sound_warning'):
            self.detector.sound_warning.set_volume(volume_level)

    def on_sound_toggled(self, state):
        self.detector.enable_alarm_sound = state == Qt.Checked
        if not self.detector.enable_alarm_sound:
            self.detector.stop_alarm()

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

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # ===== LEFT PANEL (Camera + Status) =====
        left_panel = QVBoxLayout()

        # STATUS
        self.status_label = QLabel("AWAKE")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedHeight(100)
        self.status_label.setStyleSheet("""
            font-size: 30px;
            font-weight: bold;
            background-color: green;
            color: white;
        """)
        left_panel.addWidget(self.status_label)

        # CAMERA
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 400)
        self.camera_label.setStyleSheet("background-color: black;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.camera_label)
        #fps
        self.fps_label = QLabel(self.camera_label)
        self.fps_label.setStyleSheet("""
            color: white;
            font-size: 14px;
            background-color: rgba(0,0,0,120);
            padding: 3px;
        """)
        self.fps_label.move(self.camera_label.width() - 80, 5)
        self.fps_label.resize(70, 20)
        self.fps_label.setText("FPS: 0")
        self.fps_label.raise_()

        main_layout.addLayout(left_panel, 3)  # tỷ lệ 3

        # ===== RIGHT PANEL =====
        right_panel = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(300)

        self.tab_settings = SettingsTab(self.drowsiness)
        self.tab_stats = StatisticsTab(self.drowsiness)
        self.tab_logs = QListWidget()
        self._log_max_lines = 500

        self.tabs.addTab(self.tab_settings, "Settings")
        self.tabs.addTab(self.tab_logs, "Logs")
        self.tabs.addTab(self.tab_stats, "Stats")

        right_panel.addWidget(self.tabs)

        # FPS
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("font-size: 16px;")
        # right_panel.addWidget(self.fps_label)

        # BUTTONS
        self.start_btn = QPushButton("START")
        self.start_btn.setFixedHeight(50)
        self.start_btn.setFixedWidth(200)
        self.start_btn.setStyleSheet("""
            background-color: #28a745;
            color: white;
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px;
        """)

        self.stop_btn = QPushButton("EXIT")
        self.stop_btn.setFixedHeight(40)
        self.stop_btn.setFixedWidth(60)
        self.stop_btn.setStyleSheet("""
            background-color: #dc3545;
            color: white;
            font-size: 16px;
            font-weight: bold;
            border-radius: 8px;
        """)

        # right_panel.addWidget(self.start_btn)
        # right_panel.addWidget(self.stop_btn)
        # right_panel.addStretch()
        start_layout = QHBoxLayout()
        start_layout.addStretch()
        start_layout.addWidget(self.start_btn)
        start_layout.addStretch()
        right_panel.addLayout(start_layout)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.stop_btn)
        right_panel.addLayout(btn_layout)

        main_layout.addLayout(right_panel, 1)  # tỷ lệ 1

        self.cap = cv2.VideoCapture(0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)

    def start_camera(self):
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        self.drowsiness.stop_alarm()
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
        self.drowsiness.play_alarm(state)
        self.set_status(state, message)
        self.tab_stats.update_charts(state)
        self.append_status_log(state, message, int(fps))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w

        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        self.camera_label.setPixmap(pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio
        ))

    def append_status_log(self, state, message, fps):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {state} | {message} | FPS {fps}"
        self.tab_logs.addItem(line)
        while self.tab_logs.count() > self._log_max_lines:
            self.tab_logs.takeItem(0)
        last = self.tab_logs.item(self.tab_logs.count() - 1)
        if last is not None:
            self.tab_logs.scrollToItem(last)

    def set_status(self, state, message):
        if state == "DANGER":
            bg_color = "red"
            text_color = "white"
        elif state == "WARNING":
            bg_color = "yellow"
            text_color = "black"
        else:
            bg_color = "green"
            text_color = "white"

        self.status_label.setText(f"{message}")

        self.status_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: bold;
            color: {text_color};
            background-color: {bg_color};
        """)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fps_label.move(self.camera_label.width() - 80, 5)
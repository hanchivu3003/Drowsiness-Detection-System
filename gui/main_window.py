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
from matplotlib.lines import Line2D
import numpy as np
import time


class DoubleClickButton(QPushButton):
    def __init__(self, *args, on_double_click=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_double_click = on_double_click

    def mouseDoubleClickEvent(self, event):
        if callable(self._on_double_click):
            self._on_double_click()
        event.accept()


class LiveCameraWindow(QWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self._main = main_window

        self.setWindowTitle("Live Camera")
        self.setMinimumSize(640, 400)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.image_label = QLabel("No camera")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2b2b2b; color: white;")
        layout.addWidget(self.image_label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(30)

    def showEvent(self, event):
        # If this window is reused (shown again after being closed),
        # make sure its refresh timer is running.
        if not self.timer.isActive():
            self.timer.start(30)
        super().showEvent(event)

    def _tick(self):
        # Read directly from camera stream; do NOT run inference/model here.
        ret, frame_bgr = self._main.vision.read()
        self._main._set_camera_connected(bool(ret))

        if not ret or frame_bgr is None:
            self.image_label.setText("No camera connected")
            self.image_label.setStyleSheet("background-color: #b00020; color: white; font-size: 18px;")
            return

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        # Resume main processing loop when live window closes
        if hasattr(self._main, "_on_live_window_closed"):
            self._main._on_live_window_closed()
        super().closeEvent(event)


class StatisticsTab(QWidget):
    def __init__(self, detector, parent=None):
        super().__init__(parent)
        self.detector = detector
        layout = QVBoxLayout()
        self.setLayout(layout)

        layout.addWidget(QLabel("Duration and Time"))
        self.line_fig = Figure(figsize=(5, 4), dpi=70)
        self.line_canvas = FigureCanvas(self.line_fig)
        self.ax_line = self.line_fig.add_subplot(111)
        self.ax_line.set_ylim(0, 10.5)
        self.ax_line.set_yticks([0, 1, 5, 10])
        self.ax_line.set_ylabel("Duration (s)")

        # Scatter events: each state change -> new dot
        self._x_step = 0.8  # requested "0.8cm" spacing (uniform step on x-axis)
        self._max_points = 12
        self._event_states = []          # "WARNING" | "DANGER"
        self._event_durations = []       # seconds (finalized on state change)
        self._event_timestamps = []      # datetime string when state started (for x labels)

        # Track current segment (we only draw previous segment when state changes)
        self._segment_state = None
        self._segment_start_time = None
        self._segment_start_label = None

        self.ax_line.set_xlabel("System time")
        self.ax_line.set_xticks([])
        self.ax_line.grid(True, axis="y", alpha=0.25)


        layout.addWidget(self.line_canvas)

        layout.addWidget(QLabel("Violation Distribution (%):"))
        self.pie_fig = Figure(figsize=(5, 4), dpi=70)
        self.pie_canvas = FigureCanvas(self.pie_fig)
        self.ax_pie = self.pie_fig.add_subplot(111)
        layout.addWidget(self.pie_canvas)

        self.stats_counts = {"DANGER": 0, "WARNING": 0, "SAFE": 0}
        self._state_durations = {"DANGER": 0.0, "WARNING": 0.0, "SAFE": 0.0}
        # Render default pie (SAFE 100%) on startup
        self.draw_pie_chart()

    def update_charts(self, current_state):
        now = time.time()
        state = current_state if current_state in {"SAFE", "WARNING", "DANGER"} else "SAFE"

        # initialize first segment
        if self._segment_state is None:
            self._segment_state = state
            self._segment_start_time = now
            self._segment_start_label = datetime.now().strftime("%H:%M")
        else:
            # If state changed: finalize previous segment and (maybe) append a dot
            if state != self._segment_state:
                prev_state = self._segment_state
                prev_start = self._segment_start_time
                prev_label = self._segment_start_label
                prev_duration = max(0.0, now - prev_start)

                # --- Pie chart accounting (3 states, time-weighted) ---
                if prev_state in self._state_durations:
                    self._state_durations[prev_state] += prev_duration

                if prev_state in {"WARNING", "DANGER"}:
                    self._event_states.append(prev_state)
                    self._event_durations.append(prev_duration)
                    self._event_timestamps.append(prev_label)

                    # keep last N dots
                    if len(self._event_states) > self._max_points:
                        extra = len(self._event_states) - self._max_points
                        self._event_states = self._event_states[extra:]
                        self._event_durations = self._event_durations[extra:]
                        self._event_timestamps = self._event_timestamps[extra:]

                    # redraw ONLY when we add a new dot (dots stay fixed; no moving)
                    self.ax_line.cla()
                    self.ax_line.set_ylim(0, 10.5)
                    self.ax_line.set_yticks([0, 1, 5, 10])
                    self.ax_line.set_ylabel("Duration (s)")
                    self.ax_line.set_xlabel("System time")
                    self.ax_line.grid(True, axis="y", alpha=0.25)

                    xs = [i * self._x_step for i in range(len(self._event_states))]
                    ys = [min(10.0, float(d)) for d in self._event_durations]

                    def _color_for(s):
                        return "red" if s == "DANGER" else "orange"

                    colors = [_color_for(s) for s in self._event_states]

                    # line connecting dots
                    if len(xs) >= 2:
                        self.ax_line.plot(xs, ys, linewidth=1.2, color="#666666", alpha=0.9)

                    self.ax_line.scatter(xs, ys, s=70, c=colors, zorder=3)

                    # x tick labels under each dot = system time when state started
                    self.ax_line.set_xticks(xs)
                    self.ax_line.set_xticklabels(self._event_timestamps, rotation=45, ha="right", fontsize=8)

                    if len(xs) > 0:
                        self.ax_line.set_xlim(-0.5 * self._x_step, xs[-1] + 0.5 * self._x_step)

                    # Legend (top-right): yellow=WARNING, red=DANGER
                    legend_handles = [
                        Line2D([0], [0], marker='o', color='none', markerfacecolor='orange',
                               markeredgecolor='orange', markersize=8, label='WARNING'),
                        Line2D([0], [0], marker='o', color='none', markerfacecolor='red',
                               markeredgecolor='red', markersize=8, label='DANGER'),
                    ]
                    self.ax_line.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=8)

                    self.line_canvas.draw_idle()

                # Update pie chart ONLY on state change (after accounting)
                self.draw_pie_chart()

                # start new segment
                self._segment_state = state
                self._segment_start_time = now
                self._segment_start_label = datetime.now().strftime("%H:%M")

        # Pie chart is updated only when state changes

    def draw_pie_chart(self):
        self.ax_pie.clear()
        labels = list(self._state_durations.keys())
        sizes = list(self._state_durations.values())
        colors = ['#ff4d4d', '#ffa64d', '#4dff4d']  # Red, Orange, Green

        # Default at app start: SAFE = 100% until durations exist
        if sum(sizes) <= 0:
            # Order matches insertion order of dict: DANGER, WARNING, SAFE
            sizes = [0.0, 0.0, 1.0]
            labels = ["", "", "SAFE"]

        def _autopct(pct):
            # Hide tiny/zero slices (and hide labels when sizes are 0)
            return f"{pct:.1f}%" if pct >= 0.5 else ""

        self.ax_pie.pie(sizes, labels=labels, autopct=_autopct, colors=colors, startangle=140)
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

        # 2. Thời gian còi Danger kêu thêm (sound hold)
        layout.addWidget(QLabel("Danger Alarm Sound Duration (1.0s - 5.0s):"))
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
        self._camera_connected = False
        self._last_frame_bgr = None
        self._live_window = None
        self._live_mode = False
        self._was_timer_active_before_live = False

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
        self.fps_overlay_label = QLabel(self.camera_label)
        self.fps_overlay_label.setStyleSheet("""
            color: white;
            font-size: 14px;
            background-color: rgba(0,0,0,120);
            padding: 3px;
        """)
        self.fps_overlay_label.move(5, 5)
        self.fps_overlay_label.resize(90, 22)
        self.fps_overlay_label.setText("FPS: 0")
        self.fps_overlay_label.raise_()

        main_layout.addLayout(left_panel, 3)  # tỷ lệ 3

        # ===== RIGHT PANEL =====
        right_panel = QVBoxLayout()

        # Top-right live camera button
        live_btn_row = QHBoxLayout()
        live_btn_row.addStretch()
        self.live_camera_btn = DoubleClickButton("Live Camera", on_double_click=self.open_live_camera_window)
        self.live_camera_btn.setFixedHeight(34)
        self.live_camera_btn.setFixedWidth(130)
        live_btn_row.addWidget(self.live_camera_btn)
        right_panel.addLayout(live_btn_row)
        self._update_live_camera_btn_style()

        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(300)

        self.tab_settings = SettingsTab(self.drowsiness)
        self.tab_stats = StatisticsTab(self.drowsiness)
        self.tab_logs = QListWidget()
        self._log_max_lines = 500

        self.tabs.addTab(self.tab_settings, "Settings")
        self.tabs.addTab(self.tab_logs, "Logs")
        self.tabs.addTab(self.tab_stats, "Stats")
        self.tabs.setCurrentWidget(self.tab_stats)

        right_panel.addWidget(self.tabs)

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

    def _update_live_camera_btn_style(self):
        if getattr(self, "_camera_connected", False):
            self.live_camera_btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    font-size: 14px;
                    font-weight: bold;
                    border-radius: 8px;
                }
                QPushButton:hover { background-color: #23923d; }
            """)
        else:
            self.live_camera_btn.setStyleSheet("""
                QPushButton {
                    background-color: #dc3545;
                    color: white;
                    font-size: 14px;
                    font-weight: bold;
                    border-radius: 8px;
                }
                QPushButton:hover { background-color: #c82333; }
            """)

    def _set_camera_connected(self, connected: bool):
        connected = bool(connected)
        if getattr(self, "_camera_connected", None) == connected:
            return
        self._camera_connected = connected
        self._update_live_camera_btn_style()

    def open_live_camera_window(self):
        if self._live_window is None:
            self._live_window = LiveCameraWindow(self)
        # Pause main processing (model/detect) while live window is open
        self._was_timer_active_before_live = self.timer.isActive()
        self._live_mode = True
        if self.timer.isActive():
            self.timer.stop()
        self._live_window.show()
        self._live_window.raise_()
        self._live_window.activateWindow()

    def _on_live_window_closed(self):
        self._live_mode = False
        if self._was_timer_active_before_live:
            self.timer.start(30)
        self._was_timer_active_before_live = False

    def start_camera(self):
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        self.drowsiness.stop_alarm()
        self.vision.stop()
        self._set_camera_connected(False)

    def get_color(cls_id):
        import random
        random.seed(cls_id)
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

    @staticmethod
    def _display_class_name(raw_name: str) -> str:
        # Camera view is from the observer perspective; swap left/right for user-facing meaning.
        if raw_name == "HeadLeft":
            return "HeadRight"
        if raw_name == "HeadRight":
            return "HeadLeft"
        return raw_name

    def update_frame(self):
        import time

        ret, frame = self.vision.read()
        if not ret:
            self._set_camera_connected(False)
            return
        self._set_camera_connected(True)

        # If user is in Live Camera mode, we pause processing/detect.
        if getattr(self, "_live_mode", False):
            return

        if not hasattr(self, "prev_time"):
            self.prev_time = time.time()

        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        self.fps_overlay_label.setText(f"FPS: {int(fps)}")

        results = self.vision.detect(frame)

        if len(results) > 0:
            boxes = results[0].boxes

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    names = results[0].names
                    shown_name = self._display_class_name(names[cls])
                    label = f"{shown_name} {conf:.2f}"

                    if cls in [1, 8]:
                        color = (0, 0, 255)  # Red (BGR)
                    elif cls in [3, 5, 6, 7]:
                        color = (0, 255, 255)  # Yellow (BGR)
                    else:
                        color = (0, 255, 0)  # Green (BGR)

                    shown_name = self._display_class_name(names[cls])
                    label = f"{shown_name} {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2)

        state, message = self.drowsiness.update(results)
        self.drowsiness.play_alarm(state)
        self.set_status(state, message)
        self.tab_stats.update_charts(state)
        self.append_status_log(state, message, int(fps))

        self._last_frame_bgr = frame.copy()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w

        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        self.camera_label.setPixmap(pixmap.scaled(
            self.camera_label.size(),
            Qt.KeepAspectRatio
        ))
        self.fps_overlay_label.raise_()

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
        self.fps_overlay_label.move(5, 5)
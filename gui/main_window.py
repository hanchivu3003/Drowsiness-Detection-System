from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QWidget,
    QVBoxLayout, QHBoxLayout, QFrame, QSpinBox
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
        layout.setSpacing(10)
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

        display_frame = cv2.flip(frame_bgr, 1) if getattr(self._main, "_mirror_display", False) else frame_bgr
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
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

        duration_title = QLabel("Duration and Time")
        duration_title.setAlignment(Qt.AlignCenter)
        duration_title.setStyleSheet("font-weight: bold;")
        layout.addWidget(duration_title)
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

        pie_title = QLabel("Violation Distribution (%):")
        pie_title.setAlignment(Qt.AlignCenter)
        pie_title.setStyleSheet("font-weight: bold;")
        layout.addWidget(pie_title)
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
        layout.setSpacing(14)
        layout.setContentsMargins(12, 14, 12, 12)
        self.setLayout(layout)
        layout.addSpacing(8)

        def _make_card():
            card = QFrame()
            card_layout = QVBoxLayout()
            card_layout.setContentsMargins(10, 8, 10, 8)
            card_layout.setSpacing(6)
            card.setLayout(card_layout)
            return card, card_layout

        # 1. Điều chỉnh thời gian nhắm mắt (Eye Closed Threshold)
        eye_card, eye_card_layout = _make_card()
        eye_card_layout.addWidget(QLabel("Eye Closed Threshold (1.0s - 10.0s):"))
        self.eye_slider = QSlider(Qt.Horizontal)
        self.eye_slider.setMinimum(10)  # Tương ứng 1.0s
        self.eye_slider.setMaximum(100)  # Tương ứng 10.0s
        self.eye_slider.setValue(int(self.detector.EYE_CLOSED_THRESHOLD * 10))
        self.eye_label = QLabel(f"{self.detector.EYE_CLOSED_THRESHOLD}s")

        eye_layout = QHBoxLayout()
        eye_layout.addWidget(self.eye_slider)
        eye_layout.addWidget(self.eye_label)
        eye_card_layout.addLayout(eye_layout)
        layout.addWidget(eye_card)
        self.eye_slider.valueChanged.connect(self._update_eye_label)

        # 2. Ngưỡng cảnh báo mất tập trung (HeadLeft/HeadRight)
        distraction_card, distraction_card_layout = _make_card()
        distraction_card_layout.addWidget(QLabel("Head Distraction Warning Time (1.0s - 15.0s):"))
        self.distraction_slider = QSlider(Qt.Horizontal)
        self.distraction_slider.setMinimum(10)
        self.distraction_slider.setMaximum(150)
        self.distraction_slider.setValue(int(self.detector.DISTRACTION_THRESHOLD * 10))
        self.distraction_label = QLabel(f"{self.detector.DISTRACTION_THRESHOLD}s")

        distraction_layout = QHBoxLayout()
        distraction_layout.addWidget(self.distraction_slider)
        distraction_layout.addWidget(self.distraction_label)
        distraction_card_layout.addLayout(distraction_layout)
        layout.addWidget(distraction_card)
        self.distraction_slider.valueChanged.connect(self._update_distraction_label)

        # 3-4. Yawn settings on one row (small selectors)
        yawn_card, yawn_card_layout = _make_card()
        yawn_card_layout.addWidget(QLabel("Yawn Warning Settings:"))
        yawn_row_layout = QHBoxLayout()
        yawn_row_layout.setSpacing(8)
        yawn_row_layout.addWidget(QLabel("Window (min):"))
        self.yawn_window_spin = QSpinBox()
        self.yawn_window_spin.setMinimum(1)
        self.yawn_window_spin.setMaximum(10)
        self.yawn_window_spin.setValue(max(1, int(self.detector.YAWN_WARNING_WINDOW_SECONDS // 60)))
        self.yawn_window_spin.setFixedWidth(64)
        yawn_row_layout.addWidget(self.yawn_window_spin)

        yawn_row_layout.addSpacing(12)
        yawn_row_layout.addWidget(QLabel("Count:"))
        self.yawn_count_spin = QSpinBox()
        self.yawn_count_spin.setMinimum(2)
        self.yawn_count_spin.setMaximum(10)
        self.yawn_count_spin.setValue(int(self.detector.YAWN_WARNING_COUNT))
        self.yawn_count_spin.setFixedWidth(64)
        yawn_row_layout.addWidget(self.yawn_count_spin)
        yawn_row_layout.addStretch()
        yawn_card_layout.addLayout(yawn_row_layout)
        layout.addWidget(yawn_card)

        # 5. Thời gian còi Danger kêu thêm (sound hold)
        danger_card, danger_card_layout = _make_card()
        danger_card_layout.addWidget(QLabel("Danger Alarm Sound Duration (1.0s - 15.0s):"))
        self.delay_slider = QSlider(Qt.Horizontal)
        self.delay_slider.setMinimum(10)
        self.delay_slider.setMaximum(150)
        self.delay_slider.setValue(int(self.detector.DANGER_DELAY * 10))
        self.delay_label = QLabel(f"{self.detector.DANGER_DELAY}s")

        delay_layout = QHBoxLayout()
        delay_layout.addWidget(self.delay_slider)
        delay_layout.addWidget(self.delay_label)
        danger_card_layout.addLayout(delay_layout)
        layout.addWidget(danger_card)
        self.delay_slider.valueChanged.connect(self._update_delay_label)

        # 6. Thời gian còi Warning kêu (sound hold)
        warning_card, warning_card_layout = _make_card()
        warning_card_layout.addWidget(QLabel("Warning Alarm Sound Duration (1.0s - 15.0s):"))
        self.warning_delay_slider = QSlider(Qt.Horizontal)
        self.warning_delay_slider.setMinimum(10)
        self.warning_delay_slider.setMaximum(150)
        self.warning_delay_slider.setValue(int(self.detector.WARNING_DELAY * 10))
        self.warning_delay_label = QLabel(f"{self.detector.WARNING_DELAY}s")

        warning_delay_layout = QHBoxLayout()
        warning_delay_layout.addWidget(self.warning_delay_slider)
        warning_delay_layout.addWidget(self.warning_delay_label)
        warning_card_layout.addLayout(warning_delay_layout)
        layout.addWidget(warning_card)
        self.warning_delay_slider.valueChanged.connect(self._update_warning_delay_label)

        # 7. Âm thanh cảnh báo
        sound_card, sound_card_layout = _make_card()
        self.sound_checkbox = QCheckBox("Enable Alarm Sound")
        self.sound_checkbox.setChecked(bool(self.detector.enable_alarm_sound))
        sound_card_layout.addWidget(self.sound_checkbox)

        sound_card_layout.addWidget(QLabel("Alarm Volume:"))
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setMinimum(0)
        self.vol_slider.setMaximum(100)
        initial_vol = 100
        if hasattr(self.detector, "sound_warning"):
            initial_vol = int(self.detector.sound_warning.get_volume() * 100)
        self.vol_slider.setValue(initial_vol)
        sound_card_layout.addWidget(self.vol_slider)
        layout.addWidget(sound_card)

        # 8. Chế độ tối (Dark Mode)
        theme_card, theme_card_layout = _make_card()
        self.dark_mode_checkbox = QCheckBox("Dark Mode")
        theme_card_layout.addWidget(self.dark_mode_checkbox)
        self.dark_mode_checkbox.stateChanged.connect(self.toggle_dark_mode)
        layout.addWidget(theme_card)

        save_row = QHBoxLayout()
        save_row.addStretch()
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.setFixedWidth(150)
        self.save_btn.clicked.connect(self.save_settings)
        save_row.addWidget(self.save_btn)
        layout.addLayout(save_row)

        self.save_status_label = QLabel("")
        self.save_status_label.setStyleSheet("color: #2f7d32;")
        self.save_status_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.save_status_label)

        layout.addStretch()
        self.toggle_dark_mode(self.dark_mode_checkbox.checkState())
        self._bind_unsaved_markers()

    def _update_eye_label(self, value):
        self.eye_label.setText(f"{value / 10.0}s")

    def _update_distraction_label(self, value):
        self.distraction_label.setText(f"{value / 10.0}s")

    def _update_delay_label(self, value):
        self.delay_label.setText(f"{value / 10.0}s")

    def _update_warning_delay_label(self, value):
        self.warning_delay_label.setText(f"{value / 10.0}s")

    def _bind_unsaved_markers(self):
        self.eye_slider.valueChanged.connect(self._mark_unsaved)
        self.distraction_slider.valueChanged.connect(self._mark_unsaved)
        self.yawn_window_spin.valueChanged.connect(self._mark_unsaved)
        self.yawn_count_spin.valueChanged.connect(self._mark_unsaved)
        self.delay_slider.valueChanged.connect(self._mark_unsaved)
        self.warning_delay_slider.valueChanged.connect(self._mark_unsaved)
        self.vol_slider.valueChanged.connect(self._mark_unsaved)
        self.sound_checkbox.stateChanged.connect(self._mark_unsaved)

    def _mark_unsaved(self, *_):
        self.save_status_label.setStyleSheet("color: #d97706;")
        self.save_status_label.setText("Unsaved changes")

    def save_settings(self):
        self.update_eye_threshold(self.eye_slider.value())
        self.update_distraction_threshold(self.distraction_slider.value())
        self.update_yawn_warning_window(self.yawn_window_spin.value())
        self.update_yawn_warning_count(self.yawn_count_spin.value())
        self.update_delay_threshold(self.delay_slider.value())
        self.update_warning_delay_threshold(self.warning_delay_slider.value())
        self.update_volume(self.vol_slider.value())
        self.on_sound_toggled(self.sound_checkbox.checkState())
        self.save_status_label.setStyleSheet("color: #2f7d32;")
        self.save_status_label.setText("Settings saved for current session")

    def update_eye_threshold(self, value):
        val = value / 10.0
        self.detector.EYE_CLOSED_THRESHOLD = val
        self.eye_label.setText(f"{val}s")

    def update_delay_threshold(self, value):
        val = value / 10.0
        self.detector.DANGER_DELAY = val
        self.delay_label.setText(f"{val}s")

    def update_warning_delay_threshold(self, value):
        val = value / 10.0
        self.detector.WARNING_DELAY = val
        self.warning_delay_label.setText(f"{val}s")

    def update_distraction_threshold(self, value):
        val = value / 10.0
        self.detector.DISTRACTION_THRESHOLD = val
        self.distraction_label.setText(f"{val}s")

    def update_yawn_warning_count(self, value):
        self.detector.YAWN_WARNING_COUNT = int(value)

    def update_yawn_warning_window(self, value):
        minutes = int(value)
        self.detector.YAWN_WARNING_WINDOW_SECONDS = float(minutes * 60)

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
            self.window().setStyleSheet("""
                QWidget {
                    background-color: #1f232a;
                    color: #e8edf2;
                }
                QLabel {
                    color: #e8edf2;
                }
                QFrame {
                    background-color: #2a3038;
                    border: 1px solid #3a434f;
                    border-radius: 8px;
                }
                QAbstractSpinBox, QSlider, QCheckBox, QListWidget {
                    background-color: #2a3038;
                    color: #e8edf2;
                }
                QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
                    background-color: #3a434f;
                    border: none;
                    width: 14px;
                }
                QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {
                    width: 8px;
                    height: 8px;
                }
                QPushButton {
                    background-color: #2f7ae5;
                    color: #ffffff;
                    border: none;
                    border-radius: 6px;
                    padding: 6px 10px;
                }
                QPushButton:hover {
                    background-color: #3d87ef;
                }
                QTabWidget::pane {
                    border: 1px solid #3d3d3d;
                    background: #1f232a;
                    top: -1px;
                }
                QTabWidget::tab-bar {
                    alignment: center;
                    left: 0px;
                }
                QTabBar::tab {
                    background: #3a3a3a;
                    color: #f0f0f0;
                    padding: 6px 14px;
                    margin: 0 3px;
                    border: 1px solid #3d3d3d;
                    border-bottom: none;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    min-width: 86px;
                    text-align: center;
                }
                QTabBar::tab:selected {
                    background: #505050;
                    color: #ffffff;
                    font-weight: bold;
                }
            """)
        else:
            self.window().setStyleSheet("""
                QWidget {
                    background-color: #f5f7fa;
                    color: #1b1f24;
                }
                QLabel {
                    color: #1b1f24;
                }
                QFrame {
                    background-color: #f3f6fa;
                    border: 1px solid #e2e8f0;
                    border-radius: 8px;
                }
                QAbstractSpinBox, QSlider, QCheckBox, QListWidget {
                    background-color: #ffffff;
                    color: #1b1f24;
                }
                QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
                    background-color: #e7ecf2;
                    border: none;
                    width: 14px;
                }
                QAbstractSpinBox::up-arrow, QAbstractSpinBox::down-arrow {
                    width: 8px;
                    height: 8px;
                }
                QPushButton {
                    background-color: #2f7ae5;
                    color: #ffffff;
                    border: none;
                    border-radius: 6px;
                    padding: 6px 10px;
                }
                QPushButton:hover {
                    background-color: #4e8ff3;
                }
                QTabWidget::pane {
                    border: 1px solid #c9d3df;
                    background: #f5f7fa;
                    top: -1px;
                }
                QTabWidget::tab-bar {
                    alignment: center;
                    left: 0px;
                }
                QTabBar::tab {
                    background: #e7ecf2;
                    color: #1b1f24;
                    padding: 6px 14px;
                    margin: 0 3px;
                    border: 1px solid #c9d3df;
                    border-bottom: none;
                    border-top-left-radius: 6px;
                    border-top-right-radius: 6px;
                    min-width: 86px;
                    text-align: center;
                }
                QTabBar::tab:selected {
                    background: #ffffff;
                    color: #111111;
                    font-weight: bold;
                }
            """)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.vision = VisionSystem("D:\\Drowsiness-Detection-System\\models\\best1.pt").start()
        self.drowsiness = DrowsinessDetector()
        self._camera_connected = False
        self._last_frame_bgr = None
        self._live_window = None
        self._live_mode = False
        self._was_timer_active_before_live = False
        self._mirror_display = True

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
        self.tabs.tabBar().setExpanding(False)

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
        self.stop_btn.clicked.connect(self.close)
        self.tab_settings.toggle_dark_mode(self.tab_settings.dark_mode_checkbox.checkState())

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

    def closeEvent(self, event):
        self.stop_camera()
        if self._live_window is not None:
            self._live_window.close()
        event.accept()

    def get_color(cls_id):
        import random
        random.seed(cls_id)
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )

    @staticmethod
    def _display_class_name(class_id: int, names: dict) -> str:
        # Keep detection logic untouched; only swap text shown on screen.
        raw_name = names.get(class_id, str(class_id))

        # Primary swap by known class ids in this project.
        if class_id == 5:
            return "HeadRight"
        if class_id == 6:
            return "HeadLeft"

        # Safety fallback if model/export mapping changes but class names stay the same.
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
        display_frame = cv2.flip(frame, 1) if self._mirror_display else frame.copy()
        frame_width = frame.shape[1]

        if len(results) > 0:
            boxes = results[0].boxes

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    names = results[0].names
                    shown_name = self._display_class_name(cls, names)
                    label = f"{shown_name} {conf:.2f}"

                    if cls in [1, 8]:
                        color = (0, 0, 255)  # Red (BGR)
                    elif cls in [3, 5, 6, 7]:
                        color = (0, 255, 255)  # Yellow (BGR)
                    else:
                        color = (0, 255, 0)  # Green (BGR)

                    draw_x1, draw_x2 = x1, x2
                    if self._mirror_display:
                        draw_x1 = frame_width - x2
                        draw_x2 = frame_width - x1

                    cv2.rectangle(display_frame, (draw_x1, y1), (draw_x2, y2), color, 2)

                    cv2.putText(display_frame, label, (draw_x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 2)

        state, message = self.drowsiness.update(results)
        self.drowsiness.play_alarm(state)
        self.set_status(state, message)
        self.tab_stats.update_charts(state)
        self.append_status_log(state, message, int(fps))

        self._last_frame_bgr = frame.copy()

        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w

        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
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
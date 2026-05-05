import time
from pathlib import Path

import pygame

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DrowsinessDetector:
    def __init__(self):
        # Khởi tạo các biến đếm thời gian
        self.eye_closed_start_time = None
        self.distraction_start_time = None
        self.yawn_count = 0
        self.last_yawn_time = 0

        self.danger_end_time = 0
        self.DANGER_DELAY = 3.0
        # Âm thanh (đường dẫn tuyệt đối để chạy được dù cwd khác)
        pygame.mixer.init()
        self.sound_warning = pygame.mixer.Sound(
            str(_PROJECT_ROOT / "assets" / "sounds" / "warning.wav")
        )
        self.sound_danger = pygame.mixer.Sound(
            str(_PROJECT_ROOT / "assets" / "sounds" / "danger.wav")
        )

        self.enable_alarm_sound = True
        self._prev_alarm_state = "SAFE"

        # Các ngưỡng cấu hình
        self.EYE_CLOSED_THRESHOLD = 1.0  # Nhắm mắt quá 1 giây -> Danger
        self.DISTRACTION_THRESHOLD = 3.0  # Ngoảnh mặt quá 3 giây -> Warning
        self.YAWN_COOLDOWN = 5.0  # Khoảng cách giữa 2 lần ngáp

        # Trạng thái hiện tại
        self.current_state = "SAFE"
        self.detail_message = "Focused"

        # Map index class từ model
        self.CLASS_NAMES = {
            0: 'OpenEye', 1: 'CloseEye', 2: 'NoYawn', 3: 'Yawn',
            4: 'Focused', 5: 'HeadLeft', 6: 'HeadRight', 7: 'HeadBack', 8: 'HeadDown'
        }

    def stop_alarm(self):
        pygame.mixer.Channel(0).stop()

    def play_alarm(self, state):
        ch = pygame.mixer.Channel(0)

        if not self.enable_alarm_sound:
            ch.stop()
            self._prev_alarm_state = state
            return

        prev = self._prev_alarm_state

        if state == "SAFE":
            ch.stop()
            self._prev_alarm_state = state
            return

        if state == "DANGER":
            if not ch.get_busy() or ch.get_sound() != self.sound_danger:
                ch.play(self.sound_danger, loops=-1)
            self._prev_alarm_state = state
            return

        if state == "WARNING":
            if prev != "WARNING":
                ch.stop()
                ch.play(self.sound_warning)
            self._prev_alarm_state = state
            return

        self._prev_alarm_state = state

    def update(self, results):
        current_time = time.time()

        # 1. TRƯỜNG HỢP KHÔNG PHÁT HIỆN ĐỐI TƯỢNG
        if not results or len(results[0].boxes) == 0:
            # Nếu đang trong thời gian delay của Danger thì vẫn giữ Danger
            if current_time < self.danger_end_time:
                return "DANGER", "CRITICAL: DROWSINESS DETECTED!"
            return "SAFE", "DRIVER NOT FOUND"

        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        is_danger_now = False

        # Nhắm mắt (CloseEye)
        if 1 in classes:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = current_time
            if current_time - self.eye_closed_start_time >= self.EYE_CLOSED_THRESHOLD:
                is_danger_now = True
        else:
            self.eye_closed_start_time = None

        # Gục đầu (HeadDown)
        if 8 in classes:
            is_danger_now = True

        # Nếu phát hiện Danger ở frame này, cập nhật mốc kết thúc delay
        if is_danger_now:
            self.danger_end_time = current_time + self.DANGER_DELAY
            self.current_state = "DANGER"
            self.detail_message = "CRITICAL: DROWSINESS DETECTED!"
            return self.current_state, self.detail_message

        if current_time < self.danger_end_time:
            return "DANGER", "CRITICAL: DROWSINESS DETECTED!"

        # Mất tập trung (HeadLeft, HeadRight, HeadBack)
        distracted_classes = {5, 6, 7}
        if any(c in distracted_classes for c in classes) and 4 not in classes:
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
            if current_time - self.distraction_start_time >= self.DISTRACTION_THRESHOLD:
                self.current_state = "WARNING"
                self.detail_message = "WARNING: DISTRACTION DETECTED!"
                return self.current_state, self.detail_message
        else:
            self.distraction_start_time = None

        # Ngáp (Yawn)
        if 3 in classes:
            if current_time - self.last_yawn_time > self.YAWN_COOLDOWN:
                self.yawn_count += 1
                self.last_yawn_time = current_time
            self.current_state = "WARNING"
            self.detail_message = f"WARNING: FATIGUE (Yawn count: {self.yawn_count})"
            return self.current_state, self.detail_message

        # --- TRẠNG THÁI AN TOÀN ---
        self.current_state = "SAFE"
        self.detail_message = "Focused"
        return self.current_state, self.detail_message
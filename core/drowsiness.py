import time
from pathlib import Path

import pygame
from collections import deque

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DrowsinessDetector:
    def __init__(self):
        # Khởi tạo các biến đếm thời gian
        self.eye_closed_start_time = None
        self.distraction_start_time = None
        self.yawn_count = 0
        self.last_yawn_time = 0
        self._yawn_times_60s = deque()
        self._yawn_burst_triggered_in_window = False

        # Warning burst when yawns exceed threshold in 60s
        self._warning_burst_remaining = 0
        self._warning_burst_next_at = 0.0
        self.WARNING_BURST_COUNT = 3
        self.WARNING_BURST_GAP_S = 0.3
        self.YAWN_WARNING_COUNT = 4  # old behavior equivalent: yawns > 3 in 60s
        self.YAWN_WARNING_WINDOW_SECONDS = 60.0
        self._suppress_next_warning_sound = False
        self._yawn_warning_latched = False

        # Alarm hold time (sound only): after leaving DANGER, keep sounding for a bit
        self.danger_alarm_until = 0.0
        self.DANGER_DELAY = 3.0  # seconds
        self.warning_alarm_until = 0.0
        self.WARNING_DELAY = 3.0  # seconds
        # Âm thanh (đường dẫn tuyệt đối để chạy được dù cwd khác)
        pygame.mixer.init()
        self.sound_warning = pygame.mixer.Sound(
            str(_PROJECT_ROOT / "assets" / "sounds" / "warning1.mp3")
        )
        self.sound_danger = pygame.mixer.Sound(
            str(_PROJECT_ROOT / "assets" / "sounds" / "danger.wav")
        )

        self.enable_alarm_sound = True
        self._prev_alarm_state = "SAFE"

        # Các ngưỡng cấu hình
        self.EYE_CLOSED_THRESHOLD = 2.0  # Nhắm mắt quá ngưỡng -> Danger
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
        self.danger_alarm_until = 0.0
        self.warning_alarm_until = 0.0

    def play_alarm(self, state):
        ch = pygame.mixer.Channel(0)

        if not self.enable_alarm_sound:
            ch.stop()
            self.danger_alarm_until = 0.0
            self.warning_alarm_until = 0.0
            self._prev_alarm_state = state
            return

        now = time.time()

        # If a yawn-triggered warning burst is pending, play it (unless danger is active).
        # This is independent from the UI/chart state timing.
        if self._warning_burst_remaining > 0:
            danger_active = (now < self.danger_alarm_until) or (state == "DANGER")
            if not danger_active and now >= self._warning_burst_next_at:
                # Don't interrupt a currently playing warning; just wait for the next gap.
                if (not ch.get_busy()) or (ch.get_sound() != self.sound_warning):
                    # If something else is playing (but not danger), stop it and play warning.
                    if ch.get_busy() and ch.get_sound() != self.sound_danger:
                        ch.stop()
                    ch.play(self.sound_warning)
                    self._warning_burst_remaining -= 1
                    self._warning_burst_next_at = now + float(self.WARNING_BURST_GAP_S)

        # Refresh hold windows while we are in each non-safe state.
        # This guarantees configured duration even when source audio file is short.
        if state == "DANGER":
            self.danger_alarm_until = now + float(self.DANGER_DELAY)
        elif state == "WARNING":
            self.warning_alarm_until = now + float(self.WARNING_DELAY)

        danger_should_sound = now < self.danger_alarm_until
        warning_should_sound = now < self.warning_alarm_until

        # In SAFE state, continue sounding only while hold windows are still active.
        if state == "SAFE":
            if danger_should_sound:
                if (not ch.get_busy()) or (ch.get_sound() != self.sound_danger):
                    if ch.get_busy() and ch.get_sound() != self.sound_danger:
                        ch.stop()
                    ch.play(self.sound_danger, loops=-1)
            elif warning_should_sound:
                if (not ch.get_busy()) or (ch.get_sound() != self.sound_warning):
                    if ch.get_busy() and ch.get_sound() != self.sound_warning:
                        ch.stop()
                    ch.play(self.sound_warning, loops=-1)
            elif ch.get_busy():
                ch.stop()
            self._prev_alarm_state = state
            return

        # Danger has highest priority.
        if state == "DANGER" or danger_should_sound:
            if (not ch.get_busy()) or (ch.get_sound() != self.sound_danger):
                if ch.get_busy() and ch.get_sound() != self.sound_danger:
                    ch.stop()
                ch.play(self.sound_danger, loops=-1)
            self._prev_alarm_state = "DANGER"
            return

        # WARNING should keep sounding (loop) until SAFE + hold window expires.
        if state == "WARNING" or warning_should_sound:
            if (not ch.get_busy()) or (ch.get_sound() != self.sound_warning):
                if ch.get_busy() and ch.get_sound() != self.sound_warning:
                    ch.stop()
                ch.play(self.sound_warning, loops=-1)
            self._prev_alarm_state = "WARNING"
            return

        self._prev_alarm_state = state

    def update(self, results):
        current_time = time.time()

        # 1. TRƯỜNG HỢP KHÔNG PHÁT HIỆN ĐỐI TƯỢNG
        if not results or len(results[0].boxes) == 0:
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

        # Gục đầu (HeadDown) hoặc ngửa đầu ra sau (HeadBack) -> Danger ngay lập tức
        if 8 in classes or 7 in classes:
            is_danger_now = True

        # Nếu phát hiện Danger ở frame này, cập nhật mốc kết thúc delay
        if is_danger_now:
            # Hold the alarm sound a bit longer, but do not hold the UI state.
            self.danger_alarm_until = current_time + self.DANGER_DELAY
            self.current_state = "DANGER"
            self.detail_message = "CRITICAL: DROWSINESS DETECTED!"
            return self.current_state, self.detail_message

        # Mất tập trung (HeadLeft, HeadRight)
        distracted_classes = {5, 6}
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

                # Track yawns in a configurable sliding window
                self._yawn_times_60s.append(current_time)
                while (
                    len(self._yawn_times_60s) > 0
                    and (current_time - self._yawn_times_60s[0]) > float(self.YAWN_WARNING_WINDOW_SECONDS)
                ):
                    self._yawn_times_60s.popleft()

                # Trigger burst once threshold is reached in the rolling 60s window.
                if len(self._yawn_times_60s) >= int(self.YAWN_WARNING_COUNT) and not self._yawn_burst_triggered_in_window:
                    self._yawn_burst_triggered_in_window = True
                    self._warning_burst_remaining = int(self.WARNING_BURST_COUNT)
                    self._warning_burst_next_at = current_time  # start immediately
                    self._suppress_next_warning_sound = True
                    self._yawn_warning_latched = True

            # Only raise WARNING state once when the threshold is exceeded.
            if self._yawn_warning_latched:
                self._yawn_warning_latched = False
                self.current_state = "WARNING"
                self.detail_message = f"WARNING: FATIGUE (Yawn count: {self.yawn_count})"
                return self.current_state, self.detail_message

            # Otherwise: do not force WARNING just because we see a yawn.
            # Continue evaluating other conditions / fall through to SAFE if none.

        # Reset trigger once we fall back below threshold in the rolling window
        if len(self._yawn_times_60s) < int(self.YAWN_WARNING_COUNT):
            self._yawn_burst_triggered_in_window = False

        # --- TRẠNG THÁI AN TOÀN ---
        self.current_state = "SAFE"
        self.detail_message = "FOCUSED"
        return self.current_state, self.detail_message
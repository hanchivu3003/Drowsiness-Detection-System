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

        # Warning burst when yawns > 3 in 60s
        self._warning_burst_remaining = 0
        self._warning_burst_next_at = 0.0
        self.WARNING_BURST_COUNT = 3
        self.WARNING_BURST_GAP_S = 0.3
        self._suppress_next_warning_sound = False
        self._yawn_warning_latched = False

        # Alarm hold time (sound only): after leaving DANGER, keep sounding for a bit
        self.danger_alarm_until = 0.0
        self.DANGER_DELAY = 3.0  # seconds
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
            # Swap left/right for user-facing meaning in camera view
            4: 'Focused', 5: 'HeadRight', 6: 'HeadLeft', 7: 'HeadBack', 8: 'HeadDown'
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

        # Keep danger alarm active for a short hold, but don't force UI/state to remain DANGER.
        # Requirement: when leaving DANGER, danger sound should continue for at most DANGER_DELAY seconds then stop.
        danger_should_sound = (state == "DANGER") or (now < self.danger_alarm_until)

        # If SAFE: do NOT stop warning sound; let wav finish naturally.
        # But if danger hold has ended, we should stop danger sound.
        if state == "SAFE":
            if now >= self.danger_alarm_until and ch.get_busy() and ch.get_sound() == self.sound_danger:
                ch.stop()
            self._prev_alarm_state = state
            return

        # DANGER (or danger-hold): play danger continuously, but only within the hold window.
        # When the hold window ends and state is no longer DANGER, stop the danger sound.
        if danger_should_sound:
            if state == "DANGER":
                # refresh the hold window each frame we remain in DANGER
                self.danger_alarm_until = now + float(self.DANGER_DELAY)

            if (not ch.get_busy()) or (ch.get_sound() != self.sound_danger):
                # If something else is playing, replace with danger
                if ch.get_busy() and ch.get_sound() != self.sound_danger:
                    ch.stop()
                ch.play(self.sound_danger, loops=-1)

            # If we've left DANGER and the hold window is over, stop.
            if state != "DANGER" and now >= self.danger_alarm_until and ch.get_busy() and ch.get_sound() == self.sound_danger:
                ch.stop()

            self._prev_alarm_state = "DANGER"
            return

        if state == "WARNING":
            # WARNING: play once on transition into WARNING.
            # If state changes to SAFE later, we let the wav finish (no stop()).
            if prev != "WARNING":
                if self._suppress_next_warning_sound:
                    self._suppress_next_warning_sound = False
                    self._prev_alarm_state = state
                    return
                # If danger sound is playing, keep danger priority and don't override.
                if ch.get_busy() and ch.get_sound() == self.sound_danger:
                    self._prev_alarm_state = state
                    return
                ch.play(self.sound_warning)
            self._prev_alarm_state = state
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

        # Gục đầu (HeadDown)
        if 8 in classes:
            is_danger_now = True

        # Nếu phát hiện Danger ở frame này, cập nhật mốc kết thúc delay
        if is_danger_now:
            # Hold the alarm sound a bit longer, but do not hold the UI state.
            self.danger_alarm_until = current_time + self.DANGER_DELAY
            self.current_state = "DANGER"
            self.detail_message = "CRITICAL: DROWSINESS DETECTED!"
            return self.current_state, self.detail_message

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

                # Track yawns in a sliding 60s window
                self._yawn_times_60s.append(current_time)
                while len(self._yawn_times_60s) > 0 and (current_time - self._yawn_times_60s[0]) > 60.0:
                    self._yawn_times_60s.popleft()

                # Trigger burst when yawns > 3 in last 60s (i.e. 4th yawn), once per window
                if len(self._yawn_times_60s) > 3 and not self._yawn_burst_triggered_in_window:
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

        # Reset trigger once we fall back to <=3 yawns in the rolling window
        if len(self._yawn_times_60s) <= 3:
            self._yawn_burst_triggered_in_window = False

        # --- TRẠNG THÁI AN TOÀN ---
        self.current_state = "SAFE"
        self.detail_message = "FOCUSED"
        return self.current_state, self.detail_message
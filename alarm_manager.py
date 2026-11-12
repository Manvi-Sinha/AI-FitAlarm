"""alarm_manager.py

Non-blocking alarm playback and alarm-schedule helpers.

Ported from the original main.py (create_alarm_melody, AlarmPlayer,
parse_alarm_time, next_alarm_datetime, format_remaining). The melody, the
winsound.SND_ASYNC | SND_LOOP playback and the stop behaviour are unchanged.
New: playback errors are stored in `last_error` so the UI can show them
instead of only printing to the console.
"""

import math
import os
import struct
import threading
import wave
from datetime import datetime, timedelta

# winsound only exists on Windows. If it is missing the app still runs, silently.
try:
    import winsound
except ImportError:
    winsound = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALARM_WAV_PATH = os.path.join(BASE_DIR, "ai_alarm_melody.wav")


# ------------------------- ALARM MELODY -------------------------
def create_alarm_melody(path=ALARM_WAV_PATH):
    """Generate a short looping melody WAV using Python's standard library."""
    sample_rate = 22050
    notes = [659.25, 783.99, 880.00, 783.99, 659.25, 587.33, 659.25, 783.99]
    note_seconds = 0.34
    gap_seconds = 0.045
    amplitude = 0.42

    frames = bytearray()
    for frequency in notes:
        count = int(sample_rate * note_seconds)
        for i in range(count):
            t = i / sample_rate
            fade_in = min(1.0, t / 0.025)
            fade_out = min(1.0, (note_seconds - t) / 0.045)
            envelope = max(0.0, min(fade_in, fade_out))
            tone = math.sin(2 * math.pi * frequency * t)
            overtone = 0.22 * math.sin(2 * math.pi * frequency * 2 * t)
            sample = int(
                32767 * amplitude * envelope * (tone + overtone) / 1.22
            )
            frames.extend(struct.pack("<h", sample))
        frames.extend(b"\x00\x00" * int(sample_rate * gap_seconds))

    # Write to a temporary file first so a half-written WAV is never played.
    temp_path = path + ".tmp"
    with wave.open(temp_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(bytes(frames))
    os.replace(temp_path, path)


class AlarmPlayer:
    """Play a looping WAV melody asynchronously without blocking pose detection.

    One instance is used per workout session. After stop() it cannot be
    restarted (same behaviour as the original application).
    """

    def __init__(self):
        self._finished = False
        self._started = False
        self._playing = False
        self._lock = threading.Lock()
        self.last_error = None

    @property
    def finished(self):
        return self._finished

    @property
    def playing(self):
        return self._playing

    def start(self):
        """Start the melody. Returns True when sound is actually playing."""
        with self._lock:
            if self._finished or self._started:
                return self._playing
            self._started = True

        if winsound is None:
            self.last_error = (
                "Alarm sound is unavailable: winsound exists only on Windows."
            )
            print("Warning:", self.last_error)
            return False

        try:
            if not os.path.exists(ALARM_WAV_PATH):
                create_alarm_melody(ALARM_WAV_PATH)
            # SND_ASYNC returns immediately, so no sound thread is needed.
            winsound.PlaySound(
                ALARM_WAV_PATH,
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP,
            )
            with self._lock:
                # stop() may have been called while the file was being prepared.
                if self._finished:
                    self._silence()
                    return False
                self._playing = True
            return True
        except (RuntimeError, OSError, wave.Error) as err:
            self.last_error = f"Could not play the alarm melody ({err})."
            print("Warning:", self.last_error)
            return False

    def _silence(self):
        if winsound is None:
            return
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
            winsound.PlaySound(None, winsound.SND_ASYNC)
        except (RuntimeError, OSError) as err:
            print(f"Warning: could not stop alarm sound cleanly ({err}).")

    def stop(self):
        """Stop the melody. Safe to call several times and from any thread."""
        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._playing = False

        self._silence()
        print("Alarm stopped.")


# ------------------------- SCHEDULING HELPERS -------------------------
def parse_alarm_time(text):
    """Turn '06:30' into (6, 30). Raises ValueError for invalid 24-hour text."""
    parsed = datetime.strptime(text.strip(), "%H:%M")
    return parsed.hour, parsed.minute


def next_alarm_datetime(hour, minute):
    """Next moment (local time) when the clock shows hour:minute.

    If that time has already passed today, tomorrow is used.
    """
    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def format_remaining(total_seconds):
    """Turn a number of seconds into HH:MM:SS text."""
    total_seconds = max(0, int(total_seconds))
    hours, rest = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

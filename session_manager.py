"""session_manager.py

The single owner of everything that must not be duplicated when Streamlit
reruns its script: the webcam thread, the alarm schedule, the alarm sound and
the Qwen bridge.

Why this module exists
----------------------
Streamlit re-executes app.py on every click and on every auto-refresh. If the
webcam loop lived in app.py, every rerun could open a second camera or start a
second thread. Instead:

* get_controller() returns ONE process-wide WorkoutController.
* The controller runs the camera loop on ONE background thread and keeps only
  the newest JPEG frame and the current counters.
* app.py just reads controller.snapshot() and draws it.

Session phases:
    IDLE -> CHECKING -> SCHEDULED -> ACTIVE -> COMPLETED
                \\-> ACTIVE (start now)  \\-> STOPPING -> IDLE
"""

import atexit
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime

from agent_bridge import AgentBridge
from alarm_manager import AlarmPlayer, next_alarm_datetime
from exercise_engine import (
    EXERCISES,
    MAX_TARGET_REPS,
    MIN_TARGET_REPS,
    MODEL_PATH,
    check_camera,
    run_exercise_session,
)
from local_agent import check_groq_status
from workout_storage import build_record, save_workout_session

PHASE_IDLE = "IDLE"
PHASE_CHECKING = "CHECKING"
PHASE_SCHEDULED = "SCHEDULED"
PHASE_ACTIVE = "ACTIVE"
PHASE_STOPPING = "STOPPING"
PHASE_COMPLETED = "COMPLETED"

MODE_SCHEDULE = "schedule"
MODE_NOW = "now"

AI_STATUS_CACHE_SECONDS = 20.0

# Text the original local_agent / agent_bridge add when the LLM is unavailable.
_FALLBACK_MARKERS = (" (AI unavailable:", " AI Coach error:")


@dataclass(frozen=True)
class Snapshot:
    """An immutable copy of the session state for the UI to draw."""
    phase: str
    mode: str
    exercise: str
    target_reps: int
    count: int
    movement: str
    feedback: str
    detail: str
    ai_message: str
    error: str
    notice: str
    scheduled_for: object          # datetime or None
    seconds_remaining: int
    frame_jpeg: object             # bytes or None
    coach_status: str              # "none" | "pending" | "ready"
    coach_summary: str
    coach_fallback: bool
    saved: bool
    history_error: str
    alarm_state: str               # "idle" | "starting" | "ringing" | "unavailable" | "stopped"
    worker_alive: bool
    session_id: int


def _clean_coach_summary(text):
    """Return (summary_text, is_fallback)."""
    text = str(text).strip()
    fallback = False
    for marker in _FALLBACK_MARKERS:
        if marker in text:
            text = text.split(marker)[0].strip()
            fallback = True
    return text, fallback


class _WorkerSink:
    """Receives updates from the camera thread. Stale sessions are ignored."""

    def __init__(self, controller, gen):
        self._c = controller
        self._gen = gen

    def on_running(self):
        c = self._c
        with c._lock:
            if c._gen == self._gen and c._phase == PHASE_ACTIVE:
                c._feedback = "Alarm ringing. Step back so your full body is visible."

    def on_frame(self, jpeg):
        c = self._c
        with c._lock:
            if c._gen == self._gen:
                c._frame = jpeg

    def on_status(self, count, movement, feedback, detail):
        c = self._c
        with c._lock:
            if c._gen == self._gen and c._phase == PHASE_ACTIVE:
                c._count = int(count)
                c._movement = movement
                c._feedback = feedback
                c._detail = detail

    def on_alarm_error(self, message):
        c = self._c
        with c._lock:
            if c._gen == self._gen:
                c._notice = message

    def on_completed(self, count):
        self._c._handle_completed(self._gen, count)

    def on_error(self, message):
        c = self._c
        with c._lock:
            if c._gen == self._gen and not c._error:
                c._error = message


class _BridgeSink:
    """Looks like a queue to AgentBridge (it only calls put) but updates state."""

    def __init__(self, controller, gen):
        self._c = controller
        self._gen = gen

    def put(self, item):
        kind, payload = item
        self._c._handle_bridge_message(self._gen, kind, payload)


class WorkoutController:
    def __init__(self):
        self._lock = threading.RLock()
        self._gen = 0                       # increments whenever a session is replaced/cancelled
        self._phase = PHASE_IDLE
        self._mode = MODE_NOW
        self._exercise = EXERCISES[0]
        self._target = 10
        self._scheduled_for = None

        self._count = 0
        self._movement = "Get Ready"
        self._feedback = ""
        self._detail = ""
        self._ai_message = ""
        self._error = ""
        self._notice = ""
        self._frame = None

        self._coach_status = "none"
        self._coach_summary = ""
        self._coach_fallback = False
        self._saved = False
        self._history_error = ""

        self._stop_event = None
        self._alarm = None
        self._bridge = None
        self._worker = None

        self._ai_status = None
        self._ai_checked_at = 0.0

    # ------------------------------------------------------------------ queries
    def snapshot(self):
        with self._lock:
            remaining = 0
            if self._phase == PHASE_SCHEDULED and self._scheduled_for is not None:
                remaining = max(
                    0, math.ceil((self._scheduled_for - datetime.now()).total_seconds())
                )
            return Snapshot(
                phase=self._phase,
                mode=self._mode,
                exercise=self._exercise,
                target_reps=self._target,
                count=self._count,
                movement=self._movement,
                feedback=self._feedback,
                detail=self._detail,
                ai_message=self._ai_message,
                error=self._error,
                notice=self._notice,
                scheduled_for=self._scheduled_for,
                seconds_remaining=remaining,
                frame_jpeg=self._frame,
                coach_status=self._coach_status,
                coach_summary=self._coach_summary,
                coach_fallback=self._coach_fallback,
                saved=self._saved,
                history_error=self._history_error,
                alarm_state=self._alarm_state_locked(),
                worker_alive=self._worker_alive_locked(),
                session_id=self._gen,
            )

    def ai_status(self, force=False):
        """Cached Groq API availability: {"state", "detail"}."""
        with self._lock:
            cached = self._ai_status
            fresh = (time.time() - self._ai_checked_at) < AI_STATUS_CACHE_SECONDS
        if cached is not None and fresh and not force:
            return cached

        status = check_groq_status(timeout=1.5)   # outside the lock: may take a moment
        with self._lock:
            self._ai_status = status
            self._ai_checked_at = time.time()
        return status

    def _worker_alive_locked(self):
        return self._worker is not None and self._worker.is_alive()

    def _alarm_state_locked(self):
        alarm = self._alarm
        if alarm is None:
            return "idle"
        if alarm.playing:
            return "ringing"
        if alarm.finished:
            return "stopped"
        if alarm.last_error:
            return "unavailable"
        return "starting"

    # ------------------------------------------------------------------ commands
    def start(self, exercise, target_reps, mode, alarm_time=None):
        """Begin a session. Returns (ok, message)."""
        try:
            target_reps = int(target_reps)
        except (TypeError, ValueError):
            return False, f"Enter a whole number from {MIN_TARGET_REPS} to {MAX_TARGET_REPS}."
        if not MIN_TARGET_REPS <= target_reps <= MAX_TARGET_REPS:
            return False, f"Target must be between {MIN_TARGET_REPS} and {MAX_TARGET_REPS}."
        if exercise not in EXERCISES:
            return False, "Choose Squats, Jumping Jacks or High Knees."
        if mode not in (MODE_SCHEDULE, MODE_NOW):
            return False, "Unknown start mode."
        if mode == MODE_SCHEDULE and alarm_time is None:
            return False, "Choose an alarm time."

        with self._lock:
            if self._phase in (PHASE_CHECKING, PHASE_SCHEDULED, PHASE_ACTIVE, PHASE_STOPPING):
                return False, "A session is already in progress."
            if self._worker_alive_locked():
                return False, "The previous session is still releasing the camera. Try again in a moment."

            self._clear_session_locked()
            self._gen += 1
            gen = self._gen
            self._phase = PHASE_CHECKING
            self._mode = mode
            self._exercise = exercise
            self._target = target_reps
            self._scheduled_for = (
                next_alarm_datetime(alarm_time.hour, alarm_time.minute)
                if mode == MODE_SCHEDULE else None
            )
            self._feedback = "Checking camera and model…"

        threading.Thread(
            target=self._preflight, args=(gen,), daemon=True, name="fitalarm-preflight"
        ).start()
        return True, ""

    def stop(self):
        """Stop the workout / cancel the scheduled alarm. Safe to call repeatedly."""
        with self._lock:
            if self._phase in (PHASE_CHECKING, PHASE_SCHEDULED):
                was_scheduled = self._phase == PHASE_SCHEDULED
                self._gen += 1                      # makes preflight/scheduler threads exit
                self._phase = PHASE_IDLE
                self._scheduled_for = None
                self._feedback = ""
                self._notice = "Alarm cancelled." if was_scheduled else "Start cancelled."
                return True

            if self._phase == PHASE_ACTIVE:
                self._phase = PHASE_STOPPING
                self._feedback = "Stopping alarm and camera…"
                if self._stop_event is not None:
                    self._stop_event.set()
                if self._alarm is not None:
                    self._alarm.stop()              # silence immediately
                return True
        return False

    def reset(self):
        """Clear results and return to IDLE. Ignored while a session is running."""
        with self._lock:
            if self._phase in (PHASE_CHECKING, PHASE_ACTIVE, PHASE_STOPPING):
                return False
            if self._phase == PHASE_SCHEDULED:
                self._gen += 1
            elif self._worker_alive_locked():
                # COMPLETED but the camera is still being released: finish it, ignore its output.
                self._gen += 1
                if self._stop_event is not None:
                    self._stop_event.set()
            self._clear_session_locked()
            self._phase = PHASE_IDLE
            return True

    def shutdown(self):
        """Stop everything and release the camera. Called at interpreter exit."""
        with self._lock:
            self._gen += 1
            if self._stop_event is not None:
                self._stop_event.set()
            if self._alarm is not None:
                self._alarm.stop()
            self._close_bridge_locked()
            worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=3.0)

    # ------------------------------------------------------------------ internals
    def _clear_session_locked(self):
        self._count = 0
        self._movement = "Get Ready"
        self._feedback = ""
        self._detail = ""
        self._ai_message = ""
        self._error = ""
        self._notice = ""
        self._frame = None
        self._scheduled_for = None
        self._coach_status = "none"
        self._coach_summary = ""
        self._coach_fallback = False
        self._saved = False
        self._history_error = ""
        if self._alarm is not None:
            self._alarm.stop()
        self._alarm = None
        self._close_bridge_locked()

    def _close_bridge_locked(self):
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None

    def _fail_locked(self, gen, message):
        if gen == self._gen:
            self._error = message
            self._phase = PHASE_IDLE
            self._feedback = ""
            self._scheduled_for = None

    def _preflight(self, gen):
        """Runs on a helper thread: model file, camera and Groq API."""
        with self._lock:
            if gen != self._gen:
                return
            exercise, mode = self._exercise, self._mode

        error = ""
        notice = ""
        if not os.path.exists(MODEL_PATH):
            error = f"Model file not found: {MODEL_PATH}"
        elif mode == MODE_SCHEDULE and not check_camera():
            # For "start now" the worker opens the camera itself, so the camera
            # is not opened twice in a row.
            error = (
                "Could not open the webcam. Close other apps that use the "
                "camera and try again."
            )

        if not error and exercise == "Squats":
            status = self.ai_status(force=True)
            if status["state"] != "online":
                notice = (
                    "Groq AI is not available, and squats require AI validation. "
"Squat repetitions will not count until the connection is restored. "
+ status["detail"]
                )

        with self._lock:
            if gen != self._gen or self._phase != PHASE_CHECKING:
                return
            if error:
                self._fail_locked(gen, error)
                return
            self._notice = notice
            if mode == MODE_NOW:
                self._launch_locked(gen)
            else:
                self._phase = PHASE_SCHEDULED
                self._feedback = ""
                threading.Thread(
                    target=self._scheduler_loop, args=(gen,),
                    daemon=True, name="fitalarm-scheduler",
                ).start()

    def _scheduler_loop(self, gen):
        """Waits for the alarm time. Exits as soon as the session is cancelled."""
        while True:
            with self._lock:
                if gen != self._gen or self._phase != PHASE_SCHEDULED:
                    return
                if datetime.now() >= self._scheduled_for:
                    self._launch_locked(gen)
                    return
            time.sleep(0.2)

    def _launch_locked(self, gen):
        """Start the one camera worker. Caller holds the lock."""
        if self._worker_alive_locked():
            self._fail_locked(gen, "The previous session is still releasing the camera.")
            return

        self._phase = PHASE_ACTIVE
        self._count = 0
        self._movement = "Get Ready"
        self._feedback = "Alarm ringing. Starting camera…"
        self._frame = None
        self._stop_event = threading.Event()
        self._alarm = AlarmPlayer()
        self._close_bridge_locked()
        self._bridge = AgentBridge(_BridgeSink(self, gen))

        self._worker = threading.Thread(
            target=self._worker_main,
            args=(gen, self._exercise, self._target, self._stop_event,
                  self._alarm, self._bridge),
            daemon=True,
            name="fitalarm-camera",
        )
        self._worker.start()

    def _worker_main(self, gen, exercise, target, stop_event, alarm, bridge):
        sink = _WorkerSink(self, gen)
        try:
            run_exercise_session(exercise, target, stop_event, sink, bridge, alarm)
        except Exception as err:        # run_exercise_session already guards itself
            sink.on_error(f"Unexpected error: {err}")
        finally:
            alarm.stop()
            self._on_worker_exit(gen)

    def _on_worker_exit(self, gen):
        with self._lock:
            if gen != self._gen:
                return
            if self._phase == PHASE_ACTIVE:
                if not self._error:
                    self._error = "The session ended unexpectedly."
                self._phase = PHASE_IDLE
                self._feedback = ""
            elif self._phase == PHASE_STOPPING:
                self._phase = PHASE_IDLE
                self._feedback = ""
                if not self._error and not self._notice:
                    self._notice = "Workout stopped."
            if self._phase != PHASE_COMPLETED:
                self._close_bridge_locked()

    def _handle_completed(self, gen, count):
        with self._lock:
            if gen != self._gen or self._phase != PHASE_ACTIVE:
                return
            self._phase = PHASE_COMPLETED
            self._count = int(count)
            self._movement = "Completed"
            self._feedback = "Target reached. Alarm stopped."
            self._detail = ""
            self._coach_status = "pending"
            exercise, target = self._exercise, self._target
            bridge = self._bridge

        # Ask the AI coach first (non-blocking), then save to history.
        if bridge is not None:
            bridge.request_summary({
                "exercise": exercise,
                "completed_reps": int(count),
                "target_reps": target,
                "completed": True,
            })

        record = build_record(exercise, count, target)
        ok, error = save_workout_session(record, session_id=gen)
        with self._lock:
            if gen == self._gen:
                self._saved = ok
                self._history_error = error or ""

    def _handle_bridge_message(self, gen, kind, payload):
        with self._lock:
            if gen != self._gen:
                return
            if kind == "agent_feedback":
                message = ""
                if isinstance(payload, dict):
                    message = str(payload.get("message", ""))
                if self._phase == PHASE_ACTIVE and message:
                    self._ai_message = message
            elif kind == "agent_error":
                print("Agent error:", payload)
                if self._phase == PHASE_ACTIVE:
                    self._ai_message = "AI validation failed. Try another controlled rep."
            elif kind == "coach_summary":
                if self._phase == PHASE_COMPLETED:
                    summary, fallback = _clean_coach_summary(payload)
                    self._coach_summary = summary
                    self._coach_fallback = fallback
                    self._coach_status = "ready"
                    self._close_bridge_locked()


# ------------------------------------------------------------------ singleton
_controller = None
_controller_lock = threading.Lock()


def get_controller():
    """Return the single process-wide controller (created on first use)."""
    global _controller
    with _controller_lock:
        if _controller is None:
            _controller = WorkoutController()
            atexit.register(_controller.shutdown)
        return _controller

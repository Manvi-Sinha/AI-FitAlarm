"""selftest.py  -  offline checks for AI FitAlarm (no webcam, Ollama server or browser).

Run:  python selftest.py

What it verifies
* The three exercise trackers count correctly on synthetic pose data.
* The real frame loop (run_exercise_session) works with a fake camera and a fake
  landmarker: completes, stops the alarm, encodes JPEG frames, releases resources.
* workout_history.json saving, duplicate protection and corrupt-file backup.
* The WorkoutController lifecycle: start now, schedule, cancel, stop, errors,
  reset, completion + coach summary + exactly one history entry.

What it does NOT verify: real webcam, real MediaPipe model, a live Ollama/Qwen
server, winsound audio, or the Streamlit rendering. Test those by hand (README).
"""

import json
import math
import os
import queue
import sys
import tempfile
import threading
import time
import types
from datetime import datetime, timedelta

import numpy as np

# Make `import ollama` work even if it is not installed / the server is off.

import exercise_engine as engine
import local_agent
import session_manager as sm
import workout_storage as storage
from exercise_engine import (
    HighKneesTracker,
    JumpingJackTracker,
    SquatTracker,
)

PASSED = []


def check(name, condition, detail=""):
    if not condition:
        raise AssertionError(f"FAILED: {name} {detail}")
    PASSED.append(name)
    print(f"  ok  {name}")


def wait_for(predicate, timeout=6.0, step=0.02):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(step)
    return False


# ----------------------------------------------------------------------------
# synthetic landmarks
# ----------------------------------------------------------------------------
class LM:
    def __init__(self, x=0.5, y=0.5, visibility=0.99, presence=0.99):
        self.x, self.y = x, y
        self.visibility, self.presence = visibility, presence


W, H = 640, 480


def blank_pose():
    return [LM() for _ in range(33)]


def squat_pose(knee_angle):
    pose = blank_pose()
    length = 96.0  # pixels
    for hip_i, knee_i, ankle_i, x in ((23, 25, 27, 0.45), (24, 26, 28, 0.55)):
        pose[hip_i] = LM(x, 0.30)
        pose[knee_i] = LM(x, 0.50)
        theta = math.radians(knee_angle)
        pose[ankle_i] = LM(
            x + length * math.sin(theta) / W,
            0.50 + (-length * math.cos(theta)) / H,
        )
    return pose


def jj_pose(open_pose):
    pose = blank_pose()
    pose[11], pose[12] = LM(0.40, 0.30), LM(0.60, 0.30)          # shoulders
    pose[23], pose[24] = LM(0.45, 0.55), LM(0.55, 0.55)          # hips (width 0.10)
    wrist_y = 0.10 if open_pose else 0.50
    pose[15], pose[16] = LM(0.30, wrist_y), LM(0.70, wrist_y)    # wrists
    if open_pose:
        pose[27], pose[28] = LM(0.30, 0.90), LM(0.70, 0.90)      # ankles wide
    else:
        pose[27], pose[28] = LM(0.46, 0.90), LM(0.54, 0.90)      # ankles together
    return pose


def knee_pose(left_up, right_up):
    pose = blank_pose()
    pose[23], pose[24] = LM(0.45, 0.50), LM(0.55, 0.50)
    pose[25] = LM(0.45, 0.50 if left_up else 0.70)
    pose[26] = LM(0.55, 0.50 if right_up else 0.70)
    pose[27], pose[28] = LM(0.45, 0.90), LM(0.55, 0.90)
    return pose


# ----------------------------------------------------------------------------
# 1. trackers
# ----------------------------------------------------------------------------
class FakeBridge:
    def __init__(self, verdict=True):
        self.verdict = verdict
        self.submitted = []
        self.results = queue.Queue()

    def submit(self, session):
        self.submitted.append(session)
        self.results.put({
            "candidate_id": session["candidate_id"],
            "is_valid_squat": self.verdict,
            "reason": "synthetic test",
        })
        return True

    def get_result_nowait(self):
        return self.results.get_nowait()


def run_squat_cycle(tracker, t):
    for angle, frames in ((170, 10), (80, 10), (170, 12)):
        for _ in range(frames):
            tracker.update(squat_pose(angle), W, H, t)
            t += 0.1
    tracker.update(squat_pose(170), W, H, t)   # frame that collects the AI verdict
    return t + 0.1


def test_trackers():
    print("trackers")
    bridge = FakeBridge(verdict=True)
    squat = SquatTracker(target_reps=3, agent_bridge=bridge)
    t = run_squat_cycle(squat, 100.0)
    check("squat: one candidate sent to Qwen", len(bridge.submitted) == 1)
    check("squat: AI-approved rep counted", squat.count == 1, f"count={squat.count}")
    check("squat: snapshot has knee angles",
          min(bridge.submitted[0]["knee_angles"]) < 100
          and max(bridge.submitted[0]["knee_angles"]) > 160)
    run_squat_cycle(squat, t)
    check("squat: second rep counted once", squat.count == 2, f"count={squat.count}")

    rejecting = SquatTracker(target_reps=3, agent_bridge=FakeBridge(verdict=False))
    run_squat_cycle(rejecting, 100.0)
    check("squat: AI-rejected rep NOT counted", rejecting.count == 0)

    offline = SquatTracker(target_reps=3, agent_bridge=None)
    run_squat_cycle(offline, 100.0)
    check("squat: no bridge means no count (original behaviour)", offline.count == 0)

    fast = SquatTracker(target_reps=3, agent_bridge=FakeBridge())
    tt = 100.0
    for angle, frames in ((170, 10), (80, 6), (170, 12)):
        for _ in range(frames):
            fast.update(squat_pose(angle), W, H, tt)
            tt += 0.01        # far too fast
    fast.update(squat_pose(170), W, H, tt)
    check("squat: too-fast movement rejected", fast.count == 0 and not fast.agent_bridge.submitted)

    label = squat.update(squat_pose(170), W, H, t + 5).movement
    check("squat: standing label", label == "Standing", label)

    jj = JumpingJackTracker(target_reps=2)
    labels = set()
    for open_pose in [False] * 5 + [True] * 5 + [False] * 5:
        labels.add(jj.update(jj_pose(open_pose), W, H).movement)
    check("jumping jacks: one rep counted", jj.count == 1, f"count={jj.count}")
    check("jumping jacks: Arms Open / Arms Closed labels", {"Arms Open", "Arms Closed"} <= labels, str(labels))
    for open_pose in [True] * 5 + [False] * 5:
        jj.update(jj_pose(open_pose), W, H)
    check("jumping jacks: second rep counted", jj.count == 2)

    hk = HighKneesTracker(target_reps=2)
    labels = set()
    for pose_args in ([(True, False)] * 4 + [(False, False)] * 4
                      + [(False, True)] * 4 + [(False, False)] * 4):
        labels.add(hk.update(knee_pose(*pose_args), W, H).movement)
    check("high knees: left+right raise = 1 rep", hk.count == 1, f"count={hk.count}")
    check("high knees: knee labels", {"Left Knee Raised", "Right Knee Raised"} <= labels, str(labels))

    hk2 = HighKneesTracker(target_reps=2)
    for _ in range(40):
        hk2.update(knee_pose(True, False), W, H)     # left knee held up forever
    check("high knees: one leg alone never counts", hk2.count == 0)

    lost = SquatTracker(target_reps=1, agent_bridge=FakeBridge())
    res = lost.update(None, W, H, 1.0)
    check("squat: no pose gives visibility feedback", "visible" in res.feedback)


# ----------------------------------------------------------------------------
# 2. the real frame loop with fake camera + landmarker
# ----------------------------------------------------------------------------
class FakeCap:
    opened = True
    instances = []

    def __init__(self, index=0):
        self.released = False
        FakeCap.instances.append(self)

    def isOpened(self):
        return FakeCap.opened

    def read(self):
        time.sleep(0.005)
        return True, np.zeros((H, W, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeLandmarker:
    instance = None

    def __init__(self):
        self.closed = False
        self.calls = 0
        FakeLandmarker.instance = self

    def detect_for_video(self, image, ts):
        self.calls += 1
        sequence = [False] * 5 + [True] * 5 + [False] * 5
        open_pose = sequence[min(self.calls, len(sequence)) - 1] if self.calls <= len(sequence) else False
        return types.SimpleNamespace(pose_landmarks=[jj_pose(open_pose)])

    def close(self):
        self.closed = True


class RecordingSink:
    def __init__(self):
        self.events = []
        self.frames = 0
        self.first_jpeg = None

    def on_running(self):
        self.events.append("running")

    def on_frame(self, jpeg):
        self.frames += 1
        self.first_jpeg = self.first_jpeg or jpeg

    def on_status(self, count, movement, feedback, detail):
        self.events.append(("status", count, movement))

    def on_alarm_error(self, message):
        self.events.append(("alarm_error", message))

    def on_completed(self, count):
        self.events.append(("completed", count))

    def on_error(self, message):
        self.events.append(("error", message))


def test_engine_loop():
    print("frame loop (fake camera + fake landmarker)")
    original = (engine.cv2.VideoCapture, engine.model_file_exists,
                engine.vision.PoseLandmarker.create_from_options)
    engine.cv2.VideoCapture = FakeCap
    engine.model_file_exists = lambda: True
    engine.vision.PoseLandmarker.create_from_options = staticmethod(lambda options: FakeLandmarker())
    try:
        from alarm_manager import AlarmPlayer

        sink = RecordingSink()
        alarm = AlarmPlayer()
        FakeCap.opened = True
        FakeCap.instances.clear()
        stop = threading.Event()
        engine.run_exercise_session("Jumping Jacks", 1, stop, sink, None, alarm)

        completed = [e for e in sink.events if isinstance(e, tuple) and e[0] == "completed"]
        check("loop: completion reported exactly once", len(completed) == 1, str(completed))
        check("loop: completed count equals target", completed[0][1] == 1)
        check("loop: alarm stopped", alarm.finished)
        check("loop: camera released", FakeCap.instances[0].released)
        check("loop: landmarker closed", FakeLandmarker.instance.closed)
        check("loop: JPEG frames produced", sink.frames > 0 and sink.first_jpeg[:2] == b"\xff\xd8")
        check("loop: no errors", not any(e[0] == "error" for e in sink.events if isinstance(e, tuple)))

        sink2 = RecordingSink()
        alarm2 = AlarmPlayer()
        FakeCap.opened = False
        FakeCap.instances.clear()
        engine.run_exercise_session("High Knees", 5, threading.Event(), sink2, None, alarm2)
        errors = [e for e in sink2.events if isinstance(e, tuple) and e[0] == "error"]
        check("loop: camera-not-found reported", len(errors) == 1 and "webcam" in errors[0][1])
        check("loop: alarm never rings without camera", not alarm2.playing)
        FakeCap.opened = True

        sink3 = RecordingSink()
        stop3 = threading.Event()
        stop3.set()
        FakeCap.instances.clear()
        engine.run_exercise_session("High Knees", 5, stop3, sink3, None, AlarmPlayer())
        check("loop: stop flag releases camera", FakeCap.instances[0].released)

        engine.model_file_exists = lambda: False
        sink4 = RecordingSink()
        engine.run_exercise_session("Squats", 5, threading.Event(), sink4, None, AlarmPlayer())
        check("loop: missing model file reported",
              any(isinstance(e, tuple) and e[0] == "error" and "Model file" in e[1] for e in sink4.events))
    finally:
        (engine.cv2.VideoCapture, engine.model_file_exists,
         engine.vision.PoseLandmarker.create_from_options) = original


# ----------------------------------------------------------------------------
# 3. storage
# ----------------------------------------------------------------------------
def test_storage():
    print("storage")
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "workout_history.json")
        record = storage.build_record("Squats", 10, 10)
        check("storage: original record keys",
              list(record) == ["date", "time", "exercise", "completed_reps", "target_reps", "completed"])
        ok, error = storage.save_workout_session(record, session_id="t-1", path=path)
        check("storage: saved", ok and error is None)
        storage.save_workout_session(record, session_id="t-1", path=path)
        check("storage: same session saved once", len(storage.load_history(path)) == 1)
        storage.save_workout_session(storage.build_record("High Knees", 5, 5), session_id="t-2", path=path)
        check("storage: second session appended", len(storage.load_history(path)) == 2)

        with open(path, "w", encoding="utf-8") as file:
            file.write("{ not valid json")
        storage.save_workout_session(record, session_id="t-3", path=path)
        backups = [name for name in os.listdir(folder) if ".corrupt-" in name]
        check("storage: corrupt file backed up, not lost", len(backups) == 1)
        check("storage: new history valid after corruption", len(storage.load_history(path)) == 1)


# ----------------------------------------------------------------------------
# 4. controller lifecycle with a fake engine
# ----------------------------------------------------------------------------
FAKE_MODE = {"mode": "complete"}


def fake_session(exercise, target, stop_event, sink, bridge, alarm):
    if FAKE_MODE["mode"] == "camera_error":
        sink.on_error("Could not open the webcam. Close other apps that use the camera and try again.")
        return
    alarm.start()
    sink.on_running()
    count = 0
    while not stop_event.is_set():
        time.sleep(0.03)
        sink.on_frame(b"\xff\xd8fakejpeg")
        if FAKE_MODE["mode"] == "complete":
            count += 1
            sink.on_status(count, "Squatting", f"rep {count}", "")
            if count >= target:
                alarm.stop()
                sink.on_completed(count)
                time.sleep(0.1)
                return


def test_controller():
    print("controller lifecycle")
    with tempfile.TemporaryDirectory() as folder:
        history_path = os.path.join(folder, "workout_history.json")
        saved = (sm.run_exercise_session, sm.check_camera, sm.MODEL_PATH,
         sm.next_alarm_datetime, storage.HISTORY_PATH,
         local_agent.check_groq_status,
         local_agent.generate_workout_summary)
        sm.run_exercise_session = fake_session
        sm.check_camera = lambda: True
        sm.MODEL_PATH = __file__
        storage.HISTORY_PATH = history_path
        sm.check_groq_status = lambda: {"state": "online", "detail": "Groq connected"}
        local_agent.generate_workout_summary = lambda *args, **kwargs: "Nice work. Tip. Keep going."

        try:
            ctl = sm.WorkoutController()

            ok, _ = ctl.start("Jumping Jacks", 3, sm.MODE_NOW)
            check("controller: start accepted", ok)
            ok2, msg = ctl.start("Squats", 5, sm.MODE_NOW)
            check("controller: second start rejected while busy", not ok2, msg)
            check("controller: reaches COMPLETED",
                  wait_for(lambda: ctl.snapshot().phase == sm.PHASE_COMPLETED))
            check("controller: coach summary arrives",
                  wait_for(lambda: ctl.snapshot().coach_status == "ready"))
            snap = ctl.snapshot()
            check("controller: coach text", snap.coach_summary.startswith("Nice work") and not snap.coach_fallback)
            check("controller: count equals target", snap.count == 3)
            check("controller: history saved once", wait_for(lambda: ctl.snapshot().saved)
                  and len(storage.load_history(history_path)) == 1)
            check("controller: worker exits", wait_for(lambda: not ctl.snapshot().worker_alive))
            check("controller: alarm stopped on completion", ctl.snapshot().alarm_state == "stopped")

                        # Coach fallback when Groq is unavailable
            def broken_summary(*args, **kwargs):
                raise ConnectionError("Groq is unavailable")

            local_agent.generate_workout_summary = broken_summary
            ctl.reset()
            ctl.start("High Knees", 2, sm.MODE_NOW)

            check("controller: second session completes",
                  wait_for(lambda: ctl.snapshot().coach_status == "ready"))

            snap = ctl.snapshot()
            check("controller: fallback detected and error text hidden",
                  snap.coach_fallback
                  and "Groq is unavailable" not in snap.coach_summary
                  and bool(snap.coach_summary))

            check("controller: two history entries (no duplicates)",
                  wait_for(lambda: len(storage.load_history(history_path)) == 2))

            local_agent.generate_workout_summary = (
                lambda *args, **kwargs: "Nice work. Tip. Keep going."
            )

            # Stop while running
            ctl.reset()
            wait_for(lambda: not ctl.snapshot().worker_alive)
            FAKE_MODE["mode"] = "hang"
            ctl.start("Jumping Jacks", 5, sm.MODE_NOW)
            check("controller: becomes ACTIVE", wait_for(lambda: ctl.snapshot().phase == sm.PHASE_ACTIVE))
            check("controller: frames flow", wait_for(lambda: ctl.snapshot().frame_jpeg is not None))
            check("controller: reset ignored while running", ctl.reset() is False)
            ctl.stop()
            check("controller: stop -> IDLE", wait_for(lambda: ctl.snapshot().phase == sm.PHASE_IDLE))
            check("controller: camera thread gone after stop", wait_for(lambda: not ctl.snapshot().worker_alive))
            check("controller: stopped workout not saved to history", len(storage.load_history(history_path)) == 2)

            # Camera error
            FAKE_MODE["mode"] = "camera_error"
            ctl.reset()
            ctl.start("Jumping Jacks", 5, sm.MODE_NOW)
            check("controller: camera error surfaces",
                  wait_for(lambda: ctl.snapshot().phase == sm.PHASE_IDLE and "webcam" in ctl.snapshot().error))
            check("controller: no thread left after camera error",
                  wait_for(lambda: not ctl.snapshot().worker_alive))

            # Scheduled alarm
            FAKE_MODE["mode"] = "hang"
            ctl.reset()
            sm.next_alarm_datetime = lambda h, m: datetime.now() + timedelta(seconds=1.2)
            ctl.start("High Knees", 5, sm.MODE_SCHEDULE, datetime.now().time())
            check("controller: SCHEDULED", wait_for(lambda: ctl.snapshot().phase == sm.PHASE_SCHEDULED))
            check("controller: countdown positive", ctl.snapshot().seconds_remaining >= 1)
            check("controller: camera NOT held while waiting", not ctl.snapshot().worker_alive)
            check("controller: alarm fires at the scheduled time",
                  wait_for(lambda: ctl.snapshot().phase == sm.PHASE_ACTIVE, timeout=5))
            ctl.stop()
            wait_for(lambda: ctl.snapshot().phase == sm.PHASE_IDLE and not ctl.snapshot().worker_alive)

            # Cancel scheduled alarm
            sm.next_alarm_datetime = lambda h, m: datetime.now() + timedelta(seconds=2)
            ctl.reset()
            ctl.start("High Knees", 5, sm.MODE_SCHEDULE, datetime.now().time())
            wait_for(lambda: ctl.snapshot().phase == sm.PHASE_SCHEDULED)
            ctl.stop()
            check("controller: cancel -> IDLE", ctl.snapshot().phase == sm.PHASE_IDLE)
            time.sleep(2.6)
            check("controller: cancelled alarm never fires",
                  ctl.snapshot().phase == sm.PHASE_IDLE and not ctl.snapshot().worker_alive)

            # Validation
            check("controller: invalid target rejected", ctl.start("Squats", 0, sm.MODE_NOW)[0] is False)
            check("controller: unknown exercise rejected", ctl.start("Burpees", 5, sm.MODE_NOW)[0] is False)

            # Missing model file
            sm.MODEL_PATH = os.path.join(folder, "missing.task")
            ctl.reset()
            ctl.start("Jumping Jacks", 5, sm.MODE_NOW)
            check("controller: missing model reported",
                  wait_for(lambda: "Model file not found" in ctl.snapshot().error))

            # Squats while Ollama is down -> warning, session still allowed
            sm.MODEL_PATH = __file__
            local_agent.ollama.list = lambda: (_ for _ in ()).throw(ConnectionError("refused"))
            FAKE_MODE["mode"] = "hang"
            ctl.reset()
            ctl.start("Squats", 5, sm.MODE_NOW)
            check("controller: Ollama-down warning for squats",
                  wait_for(lambda: ctl.snapshot().phase == sm.PHASE_ACTIVE and "not ready" in ctl.snapshot().notice))
            ctl.stop()
            wait_for(lambda: not ctl.snapshot().worker_alive)

            ctl.shutdown()
            check("controller: shutdown leaves no worker", not ctl.snapshot().worker_alive)
        finally:
            (sm.run_exercise_session, sm.check_camera, sm.MODEL_PATH,
             sm.next_alarm_datetime, storage.HISTORY_PATH,
             local_agent.ollama.chat, local_agent.ollama.list) = saved


def test_ai_status():
    print("Groq status helper")
    original = local_agent.get_client

    try:
        class FakeModels:
            def list(self):
                return types.SimpleNamespace(data=[{"id": "openai/gpt-oss-20b"}])

        class FakeClient:
            models = FakeModels()

        local_agent.get_client = lambda: FakeClient()
        check("Groq status: online",
              local_agent.check_groq_status()["state"] == "online")

        def refuse():
            raise ConnectionError("Groq unavailable")

        local_agent.get_client = refuse
        check("Groq status: offline",
              local_agent.check_groq_status()["state"] == "offline")

    finally:
        local_agent.get_client = original


def main():
    started = time.time()
    test_trackers()
    test_engine_loop()
    test_storage()
    test_ai_status()
    test_controller()
    leftovers = [t.name for t in threading.enumerate()
                 if t.name.startswith("fitalarm") and t.is_alive()]
    check("no orphan fitalarm threads", not leftovers, str(leftovers))
    print(f"\nAll {len(PASSED)} checks passed in {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()

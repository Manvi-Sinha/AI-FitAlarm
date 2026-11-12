"""exercise_engine.py

MediaPipe pose detection and the three exercise-counting systems.

The counting logic is ported 1:1 from the original main.py functions
run_squat_session, run_jumping_jacks_session and run_high_knees_session. The
thresholds below are the original values. What changed is only the structure:

* Each exercise is a tracker class with update(landmarks, width, height, now)
  that processes ONE frame and returns a FrameResult.
* run_exercise_session() owns the webcam, the MediaPipe landmarker and the
  frame loop. It runs on one background thread and reports to a "sink"
  object (see session_manager.py) instead of calling cv2.imshow / Tkinter.

Sink interface used by run_exercise_session (all methods must be thread safe):
    on_running()
    on_frame(jpeg_bytes)
    on_status(count, movement, feedback, detail)
    on_alarm_error(message)
    on_completed(count)
    on_error(message)
"""

import os
import queue
import time
from collections import deque
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ------------------------- CONFIGURATION (original values) -------------------------
EXERCISES = ("Squats", "Jumping Jacks", "High Knees")

DEFAULT_TARGET_REPS = 10
MIN_TARGET_REPS = 1
MAX_TARGET_REPS = 100

STAND_ANGLE = 160       # knee angle above this = standing (legs nearly straight)
SQUAT_ANGLE = 100       # knee angle below this = squatting (near parallel)
MIN_VISIBILITY = 0.5    # ignore landmarks the model is not confident about
CONFIRM_FRAMES = 3      # a position must hold this many frames to count as a state change
SMOOTHING_WINDOW = 5    # average the angle over this many frames to reduce jitter
MAX_LOST_FRAMES = 30    # ~1 second without a valid body -> reset to WAITING
MIN_REP_SECONDS = 0.5   # a squat faster than this is treated as noise

# Original value was 5 s because an OpenCV window stayed open. In Streamlit the
# completion card is shown immediately, so the camera is released after 1 s.
COMPLETION_DISPLAY_SECONDS = 1.0

# Webcam / display settings (new for the Streamlit version)
CAMERA_INDEX = 0
CAMERA_READ_RETRIES = 30     # tolerate ~1 s of failed reads while the camera warms up
MAX_DISPLAY_WIDTH = 800      # frames wider than this are downscaled for the browser
JPEG_QUALITY = 82

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")

# Landmark indices: (hip, knee, ankle) for each leg
LEFT_LEG = (23, 25, 27)
RIGHT_LEG = (24, 26, 28)

# Skeleton lines
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]

# BGR drawing colours matching the UI accent.
SKELETON_LINE_COLOR = (235, 235, 235)
SKELETON_POINT_COLOR = (160, 230, 62)


# ------------------------- SMALL HELPERS -------------------------
def model_file_exists():
    return os.path.exists(MODEL_PATH)


def check_camera():
    """Quick test that the webcam can be opened (it is released again right away)."""
    test_cap = cv2.VideoCapture(CAMERA_INDEX)
    ok = test_cap.isOpened()
    test_cap.release()
    return ok


def calculate_angle(a, b, c):
    """Angle at point b (in degrees) between the lines b->a and b->c."""
    ba = a - b
    bc = c - b
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # guard against rounding errors
    return float(np.degrees(np.arccos(cos_theta)))


def get_knee_angle(landmarks, width, height):
    """Return a confidence-aware knee angle from clearly visible legs.

    If both legs are reliable and agree, average them. If they disagree
    substantially (often due to camera angle or occlusion), use the leg
    with the stronger landmark confidence instead of averaging bad data.
    """
    valid_legs = []

    for hip_i, knee_i, ankle_i in (LEFT_LEG, RIGHT_LEG):
        hip, knee, ankle = landmarks[hip_i], landmarks[knee_i], landmarks[ankle_i]

        visibility = min(
            float(getattr(hip, "visibility", 0.0)),
            float(getattr(knee, "visibility", 0.0)),
            float(getattr(ankle, "visibility", 0.0)),
        )
        presence = min(
            float(getattr(hip, "presence", 1.0)),
            float(getattr(knee, "presence", 1.0)),
            float(getattr(ankle, "presence", 1.0)),
        )

        confidence = min(visibility, presence)
        if confidence < MIN_VISIBILITY:
            continue

        # Pixel coordinates compensate for non-square camera frames.
        h_pt = np.array([hip.x * width, hip.y * height], dtype=float)
        k_pt = np.array([knee.x * width, knee.y * height], dtype=float)
        a_pt = np.array([ankle.x * width, ankle.y * height], dtype=float)

        # Reject degenerate landmark geometry.
        if (np.linalg.norm(h_pt - k_pt) < 1.0 or
                np.linalg.norm(a_pt - k_pt) < 1.0):
            continue

        angle = calculate_angle(h_pt, k_pt, a_pt)
        valid_legs.append((angle, confidence))

    if not valid_legs:
        return None

    if len(valid_legs) == 1:
        return valid_legs[0][0]

    (angle_a, confidence_a), (angle_b, confidence_b) = valid_legs

    # When the two sides disagree, avoid averaging a potentially occluded leg.
    if abs(angle_a - angle_b) > 25.0:
        return angle_a if confidence_a >= confidence_b else angle_b

    # Confidence-weighted average when both legs provide similar measurements.
    total_confidence = confidence_a + confidence_b
    return (
        angle_a * confidence_a + angle_b * confidence_b
    ) / total_confidence


def _landmark_ok(landmark):
    return min(
        float(getattr(landmark, "visibility", 0.0)),
        float(getattr(landmark, "presence", 1.0)),
    ) >= MIN_VISIBILITY


@dataclass
class FrameResult:
    """What a tracker reports after processing one frame."""
    movement: str      # short label for the UI chip, e.g. "Standing"
    feedback: str      # coaching sentence, e.g. "Bend your knees further"
    state: str         # raw internal state, useful for debugging
    detail: str = ""   # optional extra text, e.g. "Knee angle 143"


# ------------------------- SQUATS (Qwen-validated) -------------------------
class SquatTracker:
    """Squat counter. Python detects a candidate squat; Qwen must confirm it.

    The state machine, the thresholds, the Qwen snapshot and the final
    integrity check are the original ones from run_squat_session.
    """

    exercise = "Squats"

    def __init__(self, target_reps, agent_bridge):
        self.target_reps = target_reps
        self.agent_bridge = agent_bridge
        self.count = 0

        self.state = "WAITING"
        self.angle_history = deque(maxlen=SMOOTHING_WINDOW)
        self.up_frames = 0
        self.down_frames = 0
        self.lost_frames = 0
        self.down_start_time = 0.0
        self.smooth_angle = None

        # Keep the full candidate movement, not just the 5-frame smoothing window.
        self.candidate_angles = []
        self.candidate_id = 0
        self.pending_candidate_id = None
        self.pending_candidate_duration = 0.0
        self.pending_candidate_angles = []
        self.pending_candidate_depth = False

        self.feedback_override = ""
        self.feedback_until = 0.0

    def _collect_ai_result(self, now):
        """Collect the AI's previous proposal without blocking webcam processing."""
        if self.agent_bridge is None:
            return

        while True:
            try:
                agent_result = self.agent_bridge.get_result_nowait()
            except queue.Empty:
                break

            result_candidate_id = agent_result.get("candidate_id")
            if (
                self.pending_candidate_id is not None
                and result_candidate_id == self.pending_candidate_id
            ):
                accepted = (
                    agent_result.get("is_valid_squat") is True
                    and self.pending_candidate_depth
                    and self.pending_candidate_duration >= MIN_REP_SECONDS
                    and bool(self.pending_candidate_angles)
                    and min(self.pending_candidate_angles) < SQUAT_ANGLE
                    and max(self.pending_candidate_angles) > STAND_ANGLE
                )
                if accepted:
                    self.count += 1
                    self.feedback_override = (
                        f"AI verified squat! {self.count}/{self.target_reps} completed"
                    )
                else:
                    reason = str(agent_result.get("reason", "")).strip()
                    self.feedback_override = (
                        "AI did not verify this squat — try a controlled rep"
                    )
                    if reason:
                        print("AI squat decision:", reason)
                self.feedback_until = now + 2.0
                self.pending_candidate_id = None
                self.pending_candidate_angles = []
                self.pending_candidate_depth = False
                self.pending_candidate_duration = 0.0
                self.candidate_angles = []
                self.state = "STANDING"
                self.up_frames = 0
                self.down_frames = 0

    def update(self, landmarks, width, height, now=None):
        now = time.time() if now is None else now

        angle = None
        if landmarks is not None:
            angle = get_knee_angle(landmarks, width, height)

        feedback = "Keep going"
        self._collect_ai_result(now)

        if angle is None:
            self.lost_frames += 1
            self.up_frames = 0
            self.down_frames = 0
            if self.lost_frames > MAX_LOST_FRAMES:
                self.state = "WAITING"
                self.angle_history.clear()
                self.smooth_angle = None
                self.candidate_angles = []
            feedback = "Keep hips, knees and ankles visible"
        else:
            self.lost_frames = 0
            self.angle_history.append(angle)
            self.smooth_angle = sum(self.angle_history) / len(self.angle_history)
            smooth = self.smooth_angle

            if self.pending_candidate_id is not None:
                feedback = "AI is checking your squat — hold on"
            elif self.count < self.target_reps:
                if self.state == "WAITING":
                    if smooth > STAND_ANGLE:
                        self.up_frames += 1
                        feedback = "Stand straight to begin"
                        if self.up_frames >= CONFIRM_FRAMES:
                            self.state = "STANDING"
                            self.up_frames = 0
                            feedback = "Ready — lower into a squat"
                    else:
                        self.up_frames = 0
                        feedback = "Stand straight to begin"

                elif self.state == "STANDING":
                    if smooth < SQUAT_ANGLE:
                        self.down_frames += 1
                        feedback = "Good depth — now stand up"
                        if self.down_frames >= CONFIRM_FRAMES:
                            self.state = "DOWN"
                            self.down_start_time = now
                            # Include the approach to depth and the current window.
                            self.candidate_angles = list(self.angle_history)
                            self.candidate_angles.append(smooth)
                            self.down_frames = 0
                    elif smooth < STAND_ANGLE:
                        self.down_frames = 0
                        feedback = "Bend your knees further"
                    else:
                        self.down_frames = 0
                        feedback = "Ready — lower into a squat"

                elif self.state == "DOWN":
                    self.candidate_angles.append(round(float(smooth), 1))
                    # Bound memory in case someone remains in the squat position.
                    self.candidate_angles = self.candidate_angles[-180:]
                    if smooth > STAND_ANGLE:
                        self.up_frames += 1
                        feedback = "Returning to standing"
                        if self.up_frames >= CONFIRM_FRAMES:
                            self._finish_candidate(now)
                            if self.pending_candidate_id is not None:
                                feedback = "Complete — AI is validating this squat"
                    else:
                        self.up_frames = 0
                        if smooth < SQUAT_ANGLE:
                            feedback = "Good depth — now stand up"
                        else:
                            feedback = "Return to standing to finish the squat"

        if now < self.feedback_until:
            feedback = self.feedback_override

        return FrameResult(
            movement=self._movement_label(angle),
            feedback=feedback,
            state=self.state,
            detail=(
                f"Knee angle {int(self.smooth_angle)}°"
                if self.smooth_angle is not None else ""
            ),
        )

    def _finish_candidate(self, now):
        """The user is back at standing: ask Qwen to validate the candidate."""
        elapsed = now - self.down_start_time
        has_depth = bool(self.candidate_angles) and min(self.candidate_angles) < SQUAT_ANGLE

        if elapsed >= MIN_REP_SECONDS and has_depth and self.agent_bridge is not None:
            self.candidate_id += 1
            session_snapshot = {
                "candidate_id": self.candidate_id,
                "completed_reps": self.count,
                "target_reps": self.target_reps,
                "knee_angles": list(self.candidate_angles),
                "movement_state": "COMPLETE_CANDIDATE",
                "movement_feedback": "User lowered into a squat and returned to standing.",
                "candidate_duration_seconds": round(elapsed, 2),
                "reached_depth": has_depth,
                "returned_to_standing": True,
            }
            if self.agent_bridge.submit(session_snapshot):
                self.pending_candidate_id = self.candidate_id
                self.pending_candidate_duration = elapsed
                self.pending_candidate_angles = list(self.candidate_angles)
                self.pending_candidate_depth = has_depth
            else:
                self.feedback_override = "AI busy — repeat the squat once ready"
                self.feedback_until = now + 1.5
        else:
            self.feedback_override = (
                "Movement too fast or depth not reached — repeat slowly"
            )
            self.feedback_until = now + 1.5

        self.state = "STANDING"
        self.up_frames = 0
        self.candidate_angles = []

    def _movement_label(self, angle):
        if self.pending_candidate_id is not None:
            return "AI Checking"
        if angle is None and self.lost_frames > 0:
            return "Not Visible"
        if self.state == "WAITING":
            return "Get Ready"
        if self.state == "STANDING":
            return "Standing"
        return "Squatting"


# ------------------------- JUMPING JACKS (rule-based) -------------------------
class JumpingJackTracker:
    """One rep is a complete closed -> open -> closed movement.

    Uses wrist height and ankle separation (original logic). Qwen validation
    is intentionally not used for this exercise.
    """

    exercise = "Jumping Jacks"
    REQUIRED_LANDMARKS = (11, 12, 15, 16, 23, 24, 27, 28)

    def __init__(self, target_reps, agent_bridge=None):
        self.target_reps = target_reps
        self.count = 0
        self.phase = "WAITING"
        self.open_frames = 0
        self.closed_frames = 0
        self.lost_frames = 0

    def update(self, pose, width, height, now=None):
        feedback = "Stand straight with your arms down"
        movement = "Get Ready"

        if pose is None:
            self.lost_frames += 1
            self.open_frames = self.closed_frames = 0
            feedback = "Keep your full body visible"
            movement = "Not Visible"
            if self.lost_frames > MAX_LOST_FRAMES:
                self.phase = "WAITING"
        else:
            visible = all(_landmark_ok(pose[i]) for i in self.REQUIRED_LANDMARKS)

            if not visible:
                self.lost_frames += 1
                self.open_frames = self.closed_frames = 0
                feedback = "Keep shoulders, wrists, hips and ankles visible"
                movement = "Not Visible"
            else:
                self.lost_frames = 0
                left_shoulder, right_shoulder = pose[11], pose[12]
                left_wrist, right_wrist = pose[15], pose[16]
                left_hip, right_hip = pose[23], pose[24]
                left_ankle, right_ankle = pose[27], pose[28]

                # Normalized coordinates are suitable for relative distances.
                hip_width = abs(left_hip.x - right_hip.x)
                ankle_width = abs(left_ankle.x - right_ankle.x)
                wrists_above = (
                    left_wrist.y < left_shoulder.y
                    and right_wrist.y < right_shoulder.y
                )
                wrists_down = (
                    left_wrist.y > left_shoulder.y
                    and right_wrist.y > right_shoulder.y
                )
                legs_open = hip_width > 0 and ankle_width > hip_width * 1.35
                legs_closed = hip_width > 0 and ankle_width < hip_width * 1.15

                is_open = wrists_above and legs_open
                is_closed = wrists_down and legs_closed

                if is_open:
                    movement = "Arms Open"
                elif is_closed:
                    movement = "Arms Closed"
                else:
                    movement = "Moving"

                if self.phase == "WAITING":
                    if is_closed:
                        self.closed_frames += 1
                        feedback = "Ready — jump and raise both arms"
                        if self.closed_frames >= CONFIRM_FRAMES:
                            self.phase = "CLOSED"
                            self.closed_frames = 0
                    else:
                        self.closed_frames = 0
                        feedback = "Start with feet together and arms down"

                elif self.phase == "CLOSED":
                    if is_open:
                        self.open_frames += 1
                        feedback = "Great! Bring your arms down and feet together"
                        if self.open_frames >= CONFIRM_FRAMES:
                            self.phase = "OPEN"
                            self.open_frames = 0
                    else:
                        self.open_frames = 0
                        feedback = "Jump and raise both arms"

                elif self.phase == "OPEN":
                    if is_closed:
                        self.closed_frames += 1
                        feedback = "Returning to starting position"
                        if self.closed_frames >= CONFIRM_FRAMES:
                            self.count += 1
                            self.phase = "CLOSED"
                            self.closed_frames = 0
                            feedback = f"Good jumping jack! {self.count}/{self.target_reps}"
                    else:
                        self.closed_frames = 0
                        feedback = "Bring your feet together and lower your arms"

        return FrameResult(movement=movement, feedback=feedback, state=self.phase)


# ------------------------- HIGH KNEES (rule-based) -------------------------
class HighKneesTracker:
    """One rep = a left AND a right high-knee raise.

    Each leg must return to its lowered position before it can be counted
    again (original logic). Qwen validation is intentionally not used.
    """

    exercise = "High Knees"
    REQUIRED_LANDMARKS = (23, 24, 25, 26, 27, 28)

    def __init__(self, target_reps, agent_bridge=None):
        self.target_reps = target_reps
        self.count = 0

        # Per-leg cycle: 0 = waiting for raise, 1 = raised, 2 = returned down.
        self.left_stage = 0
        self.right_stage = 0
        self.left_raise_frames = self.right_raise_frames = 0
        self.left_down_frames = self.right_down_frames = 0
        self.lost_frames = 0

    def update(self, pose, width, height, now=None):
        feedback = "Stand straight — raise your knees alternately"
        movement = "Standing"

        if pose is None:
            self.lost_frames += 1
            self.left_raise_frames = self.right_raise_frames = 0
            self.left_down_frames = self.right_down_frames = 0
            feedback = "Keep your full body visible"
            movement = "Not Visible"
        else:
            visible = all(_landmark_ok(pose[i]) for i in self.REQUIRED_LANDMARKS)

            if not visible:
                self.lost_frames += 1
                self.left_raise_frames = self.right_raise_frames = 0
                self.left_down_frames = self.right_down_frames = 0
                feedback = "Keep hips, knees and ankles visible"
                movement = "Not Visible"
            else:
                self.lost_frames = 0
                left_hip, right_hip = pose[23], pose[24]
                left_knee, right_knee = pose[25], pose[26]

                # A knee is considered raised when it reaches approximately
                # hip level. A small tolerance accommodates body proportions.
                left_is_raised = left_knee.y <= left_hip.y + 0.03
                right_is_raised = right_knee.y <= right_hip.y + 0.03

                # The knee must lower clearly before another raise is accepted.
                left_is_down = left_knee.y >= left_hip.y + 0.09
                right_is_down = right_knee.y >= right_hip.y + 0.09

                if left_is_raised and right_is_raised:
                    movement = "Both Knees Raised"
                elif left_is_raised:
                    movement = "Left Knee Raised"
                elif right_is_raised:
                    movement = "Right Knee Raised"
                else:
                    movement = "Standing"

                if self.left_stage == 0:
                    if left_is_raised:
                        self.left_raise_frames += 1
                        feedback = "Good! Now raise your right knee"
                        if self.left_raise_frames >= CONFIRM_FRAMES:
                            self.left_stage = 1
                            self.left_raise_frames = 0
                    else:
                        self.left_raise_frames = 0
                elif self.left_stage == 1:
                    if left_is_down:
                        self.left_down_frames += 1
                        if self.left_down_frames >= CONFIRM_FRAMES:
                            self.left_stage = 2
                            self.left_down_frames = 0
                    else:
                        self.left_down_frames = 0

                if self.right_stage == 0:
                    if right_is_raised:
                        self.right_raise_frames += 1
                        feedback = "Good! Now raise your left knee"
                        if self.right_raise_frames >= CONFIRM_FRAMES:
                            self.right_stage = 1
                            self.right_raise_frames = 0
                    else:
                        self.right_raise_frames = 0
                elif self.right_stage == 1:
                    if right_is_down:
                        self.right_down_frames += 1
                        if self.right_down_frames >= CONFIRM_FRAMES:
                            self.right_stage = 2
                            self.right_down_frames = 0
                    else:
                        self.right_down_frames = 0

                if self.left_stage == 2 and self.right_stage == 2:
                    self.count += 1
                    self.left_stage = self.right_stage = 0
                    self.left_raise_frames = self.right_raise_frames = 0
                    self.left_down_frames = self.right_down_frames = 0
                    feedback = f"Good high knees! {self.count}/{self.target_reps}"
                elif self.left_stage == 2 and self.right_stage < 2:
                    feedback = "Now complete the right knee raise"
                elif self.right_stage == 2 and self.left_stage < 2:
                    feedback = "Now complete the left knee raise"

        if self.lost_frames > MAX_LOST_FRAMES:
            self.left_stage = self.right_stage = 0

        detail = (
            f"Left {'done' if self.left_stage == 2 else 'pending'}, "
            f"right {'done' if self.right_stage == 2 else 'pending'}"
        )
        return FrameResult(
            movement=movement,
            feedback=feedback,
            state=f"L{self.left_stage}R{self.right_stage}",
            detail=detail,
        )


def make_tracker(exercise, target_reps, agent_bridge=None):
    """Create the tracker for the selected exercise."""
    if exercise == "Jumping Jacks":
        return JumpingJackTracker(target_reps, agent_bridge)
    if exercise == "High Knees":
        return HighKneesTracker(target_reps, agent_bridge)
    if exercise == "Squats":
        return SquatTracker(target_reps, agent_bridge)
    raise ValueError(f"Unsupported exercise: {exercise}")


# ------------------------- DRAWING / ENCODING -------------------------
def draw_pose(frame, landmarks, width, height):
    """Draw the skeleton and joints on the frame (in place)."""
    for start_idx, end_idx in POSE_CONNECTIONS:
        s, e = landmarks[start_idx], landmarks[end_idx]
        cv2.line(
            frame,
            (int(s.x * width), int(s.y * height)),
            (int(e.x * width), int(e.y * height)),
            SKELETON_LINE_COLOR,
            2,
        )
    for lm in landmarks:
        cv2.circle(
            frame, (int(lm.x * width), int(lm.y * height)),
            4, SKELETON_POINT_COLOR, -1
        )


def encode_frame(frame):
    """Downscale if needed and encode as JPEG bytes (None if encoding fails)."""
    height, width = frame.shape[:2]
    if width > MAX_DISPLAY_WIDTH:
        scale = MAX_DISPLAY_WIDTH / float(width)
        frame = cv2.resize(
            frame, (MAX_DISPLAY_WIDTH, int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buffer = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    return buffer.tobytes() if ok else None


# ------------------------- THE SESSION LOOP (one background thread) -------------------------
def run_exercise_session(exercise, target_reps, stop_event, sink, agent_bridge, alarm):
    """Own the webcam + MediaPipe for one workout until done, stopped or failed.

    Runs on a worker thread. Every resource it opens (landmarker, camera,
    alarm) is released in the finally block, whatever happens.
    """
    landmarker = None
    cap = None

    try:
        if not model_file_exists():
            sink.on_error(f"Model file not found: {MODEL_PATH}")
            return

        tracker = make_tracker(exercise, target_reps, agent_bridge)

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            sink.on_error(
                "Could not open the webcam. Close other apps that use the "
                "camera and try again."
            )
            return

        start_time = time.time()
        last_ts = -1
        failed_reads = 0
        completed = False
        completed_at = None
        last_status = None

        # Alarm starts only after the camera and the model are ready (original order).
        if not alarm.start() and alarm.last_error:
            sink.on_alarm_error(alarm.last_error)
        sink.on_running()
        sink.on_status(0, "Get Ready", "Starting…", "")

        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                failed_reads += 1
                if failed_reads > CAMERA_READ_RETRIES:
                    sink.on_error("Failed to read a frame from the webcam.")
                    break
                time.sleep(0.03)
                continue
            failed_reads = 0

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            ts = int((time.time() - start_time) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            detection = landmarker.detect_for_video(mp_image, ts)
            pose = detection.pose_landmarks[0] if detection.pose_landmarks else None

            # Once the target is reached the counters are frozen, so a
            # repetition can never be counted twice or exceed the target.
            if not completed:
                result = tracker.update(pose, width, height, time.time())
                status = (tracker.count, result.movement, result.feedback, result.detail)
                if status != last_status:
                    last_status = status
                    sink.on_status(*status)

                if tracker.count >= target_reps:
                    completed = True
                    completed_at = time.time()
                    alarm.stop()
                    sink.on_completed(min(tracker.count, target_reps))

            if pose is not None:
                draw_pose(frame, pose, width, height)

            if completed:
                text = "TASK COMPLETED"
                (text_w, _), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 4
                )
                cv2.putText(
                    frame, text, (max(10, (width - text_w) // 2), height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, SKELETON_POINT_COLOR, 4,
                )

            jpeg = encode_frame(frame)
            if jpeg is not None:
                sink.on_frame(jpeg)

            if completed and time.time() - completed_at >= COMPLETION_DISPLAY_SECONDS:
                break

    except Exception as err:
        sink.on_error(f"Unexpected error in the {exercise} session: {err}")
    finally:
        alarm.stop()
        if cap is not None:
            cap.release()
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception as err:
                print(f"Warning: could not close the pose landmarker ({err}).")

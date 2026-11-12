"""app.py  -  AI FitAlarm: Wake Up. Move. Conquer.

Run with:  streamlit run app.py

This file only draws the interface and forwards button clicks. The webcam,
MediaPipe, the alarm and Qwen are owned by ONE WorkoutController
(session_manager.py) running on background threads, so Streamlit reruns can
never create a second camera capture or a second worker.
"""

import base64
import html
from datetime import datetime, time as dtime

import streamlit as st

st.set_page_config(
    page_title="AI FitAlarm",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    from alarm_manager import format_remaining, next_alarm_datetime
    from exercise_engine import (
        DEFAULT_TARGET_REPS,
        EXERCISES,
        MAX_TARGET_REPS,
        MIN_TARGET_REPS,
    )
    from local_agent import get_exercise_tip
    from session_manager import (
        MODE_NOW,
        MODE_SCHEDULE,
        PHASE_ACTIVE,
        PHASE_CHECKING,
        PHASE_COMPLETED,
        PHASE_IDLE,
        PHASE_SCHEDULED,
        PHASE_STOPPING,
        get_controller,
    )
except ImportError as import_error:
    st.error(
        f"A required package is missing: {import_error}. "
        "Run: pip install -r requirements.txt"
    )
    st.stop()

# True: the frame is embedded in the page as a base64 <img> (smooth, no file
# churn). False: use st.image() instead (use this if your browser shows no video).
USE_DATA_URI_FRAME = True

MODE_LABELS = ("Schedule alarm", "Start now")

EXERCISE_HELP = {
    "Squats": (
        "A rep counts when you drop below parallel and stand back up. "
        "Groq AI proposes a validation decision for every squat. Keep hips, knees "
        "and ankles in frame."
    ),
    "Jumping Jacks": (
        "A rep is closed, open, closed: arms overhead with feet wide, then "
        "back to arms down and feet together. Keep your whole body in frame."
    ),
    "High Knees": (
        "Raise each knee to hip height. One rep is one left and one right "
        "raise. Keep hips, knees and ankles in frame."
    ),
}

STATUS_CHIP = {
    PHASE_IDLE: ("Ready", "ok"),
    PHASE_CHECKING: ("Checking", "warn"),
    PHASE_SCHEDULED: ("Alarm set", "warn"),
    PHASE_ACTIVE: ("Running", "live"),
    PHASE_STOPPING: ("Stopping", "warn"),
    PHASE_COMPLETED: ("Completed", "ok"),
}

ALARM_LABEL = {
    "starting": "Alarm starting",
    "ringing": "Alarm ringing",
    "stopped": "Alarm stopped",
    "unavailable": "Alarm silent",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900&display=swap');

:root {
  --ink: #0c0e0d;
  --panel: #141917;
  --panel-2: #1a211e;
  --line: #263029;
  --text: #edf2ee;
  --muted: #8a968e;
  --accent: #3ee6a0;
  --accent-dim: rgba(62, 230, 160, 0.13);
  --amber: #ffb547;
  --red: #ff6b6b;
  --font: 'Archivo', 'Segoe UI', system-ui, -apple-system, sans-serif;
}

.stApp, .stApp p, .stApp label, .stApp li, .stApp button, .stApp input,
.stApp textarea, .stApp h1, .stApp h2, .stApp h3 {
  font-family: var(--font);
}
.stApp { background: radial-gradient(1200px 520px at 85% -10%, rgba(62,230,160,0.07), transparent 60%), var(--ink); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stStatusWidget"], [data-testid="stDecoration"] { visibility: hidden; height: 0; }
.block-container { max-width: 1280px; padding-top: 1.6rem; padding-bottom: 3rem; }

/* Auto-refreshing fragments must not flicker or dim while they update. */
[data-stale="true"] { opacity: 1 !important; transition: none !important; }

/* ---------- header ---------- */
.fa-header { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 22px; flex-wrap: wrap; }
.fa-brand { display: flex; align-items: center; gap: 14px; }
.fa-name { font-family: var(--font); font-weight: 800; font-size: 25px; letter-spacing: -0.01em; font-stretch: 118%; line-height: 1.05; color: var(--text); }
.fa-tag-line { font-family: var(--font); font-size: 14px; color: var(--muted); margin-top: 3px; }
.fa-chips { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
.fa-chip { display: inline-flex; align-items: center; gap: 8px; padding: 7px 14px; border-radius: 999px; font-family: var(--font); font-size: 13px; font-weight: 600; background: var(--panel); border: 1px solid var(--line); color: var(--text); }
.fa-chip i { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); display: inline-block; }
.fa-chip.ok i { background: var(--accent); }
.fa-chip.warn i { background: var(--amber); }
.fa-chip.bad i { background: var(--red); }
.fa-chip.live { border-color: rgba(62,230,160,0.55); background: var(--accent-dim); }
.fa-chip.live i { background: var(--accent); animation: fa-pulse 1.6s ease-in-out infinite; }
@keyframes fa-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
@media (prefers-reduced-motion: reduce) { .fa-chip.live i, .fa-wait { animation: none !important; } }

/* ---------- setup panel ---------- */
.st-key-setup { background: var(--panel); border: 1px solid var(--line); border-radius: 20px; padding: 24px 22px 22px 22px; }
.fa-h { font-family: var(--font); font-size: 16px; font-weight: 700; color: var(--text); margin: 4px 0 10px 0; }
.fa-sub { font-family: var(--font); font-size: 13px; color: var(--muted); margin: -2px 0 6px 0; line-height: 1.45; }
.fa-gap { height: 14px; }

/* segmented control for the start mode */
.st-key-mode div[role="radiogroup"] { gap: 4px; background: var(--ink); padding: 4px; border-radius: 13px; border: 1px solid var(--line); width: 100%; }
.st-key-mode label { flex: 1; justify-content: center; padding: 9px 10px; border-radius: 10px; margin: 0 !important; cursor: pointer; }
.st-key-mode label > div:first-child { display: none; }
.st-key-mode label:has(input:checked) { background: var(--accent); }
.st-key-mode label:has(input:checked) p { color: #06140d !important; font-weight: 700; }
.st-key-mode label p { font-size: 14px; }

/* exercise rows */
.st-key-exercise div[role="radiogroup"] { flex-direction: column; gap: 8px; width: 100%; }
.st-key-exercise label { width: 100%; background: var(--panel-2); border: 1px solid var(--line); border-radius: 13px; padding: 13px 16px; margin: 0 !important; cursor: pointer; }
.st-key-exercise label > div:first-child { display: none; }
.st-key-exercise label p { font-size: 15px; font-weight: 600; }
.st-key-exercise label:has(input:checked) { border-color: var(--accent); background: var(--accent-dim); }
.st-key-exercise label:has(input:focus-visible), .st-key-mode label:has(input:focus-visible) { outline: 2px solid var(--accent); outline-offset: 2px; }
.st-key-exercise label:has(input:disabled), .st-key-mode label:has(input:disabled) { opacity: 0.55; cursor: not-allowed; }

.fa-help { display: flex; gap: 12px; align-items: flex-start; margin: 12px 0 4px 0; padding: 12px 14px; border-radius: 12px; background: var(--ink); border: 1px solid var(--line); font-family: var(--font); font-size: 13px; line-height: 1.5; color: var(--muted); }
.fa-help svg { flex: none; margin-top: 2px; }

/* buttons */
.stButton > button { width: 100%; border-radius: 13px; font-weight: 700; padding: 0.7rem 1rem; border: 1px solid var(--line); background: var(--panel-2); color: var(--text); transition: border-color 0.15s, background 0.15s; }
.stButton > button:hover:not(:disabled) { border-color: var(--accent); color: var(--text); }
.stButton > button:disabled { opacity: 0.38; }
.st-key-btn_start button, .st-key-btn_new button { background: var(--accent); border-color: var(--accent); color: #06140d; font-size: 16px; padding: 0.85rem 1rem; }
.st-key-btn_start button:hover:not(:disabled), .st-key-btn_new button:hover:not(:disabled) { background: #5cf0b4; border-color: #5cf0b4; color: #06140d; }
.st-key-btn_start button p, .st-key-btn_new button p { color: #06140d; }
.st-key-btn_stop button:hover:not(:disabled) { border-color: var(--red); }

/* ---------- live stage ---------- */
.fa-video { position: relative; border-radius: 20px; overflow: hidden; background: #070908; border: 1px solid var(--line); aspect-ratio: 4 / 3; max-height: 600px; width: 100%; }
.fa-video img { width: 100%; height: 100%; object-fit: contain; display: block; }
.fa-overlay { position: absolute; top: 14px; left: 14px; right: 14px; display: flex; justify-content: space-between; gap: 8px; pointer-events: none; }
.fa-overlay .group { display: flex; gap: 8px; flex-wrap: wrap; }
.fa-pill { padding: 6px 13px; border-radius: 999px; background: rgba(12,14,13,0.74); backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,0.08); font-family: var(--font); font-size: 13px; font-weight: 600; color: var(--text); }
.fa-pill.accent { color: #06140d; background: var(--accent); border-color: var(--accent); }
.fa-wait { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 10px; text-align: center; padding: 24px; font-family: var(--font); }
.fa-wait .big { font-family: var(--font); font-stretch: 125%; font-weight: 800; font-size: 62px; letter-spacing: 0.01em; color: var(--text); line-height: 1; font-variant-numeric: tabular-nums; }
.fa-wait .title { font-size: 20px; font-weight: 700; color: var(--text); }
.fa-wait .note { font-size: 14px; color: var(--muted); max-width: 440px; line-height: 1.5; }
.fa-wait.pulse .title { animation: fa-pulse 1.8s ease-in-out infinite; }

.fa-hud { display: grid; grid-template-columns: auto 1fr; gap: 28px; align-items: center; margin-top: 18px; padding: 20px 24px; background: var(--panel); border: 1px solid var(--line); border-radius: 20px; }
.fa-count { display: flex; align-items: baseline; gap: 10px; }
.fa-count .num { font-family: var(--font); font-stretch: 125%; font-weight: 800; font-size: 78px; line-height: 0.95; color: var(--text); font-variant-numeric: tabular-nums; }
.fa-count .of { font-family: var(--font); font-stretch: 112%; font-weight: 700; font-size: 28px; color: var(--muted); font-variant-numeric: tabular-nums; }
.fa-meta { min-width: 0; }
.fa-bar { height: 12px; background: var(--ink); border: 1px solid var(--line); border-radius: 999px; overflow: hidden; }
.fa-bar i { display: block; height: 100%; background: linear-gradient(90deg, #1fc27f, var(--accent)); border-radius: 999px; transition: width 0.25s ease; }
.fa-meta .row { display: flex; justify-content: space-between; margin-top: 8px; font-family: var(--font); font-size: 13px; color: var(--muted); }
.fa-feedback { margin-top: 12px; font-family: var(--font); font-size: 18px; font-weight: 600; color: var(--text); line-height: 1.35; }
.fa-ai { margin-top: 6px; font-family: var(--font); font-size: 14px; color: var(--muted); line-height: 1.4; }
.fa-ai b { color: var(--accent); font-weight: 700; }

/* ---------- banners ---------- */
.fa-banner { margin-bottom: 14px; padding: 13px 16px; border-radius: 14px; font-family: var(--font); font-size: 14px; line-height: 1.5; border: 1px solid var(--line); background: var(--panel); color: var(--text); }
.fa-banner.err { border-color: rgba(255,107,107,0.5); background: rgba(255,107,107,0.08); }
.fa-banner.warn { border-color: rgba(255,181,71,0.5); background: rgba(255,181,71,0.08); }
.fa-banner b { font-weight: 700; }

/* ---------- completion + coach ---------- */
.fa-done { margin-top: 26px; padding: 30px 32px; border-radius: 22px; border: 1px solid rgba(62,230,160,0.45); background: linear-gradient(135deg, rgba(62,230,160,0.16), rgba(62,230,160,0.025) 60%), var(--panel); }
.fa-done h2 { font-family: var(--font); font-stretch: 118%; font-weight: 800; font-size: 32px; margin: 0 0 6px 0; padding: 0; color: var(--text); letter-spacing: -0.01em; }
.fa-done .lead { font-family: var(--font); font-size: 17px; color: var(--text); margin: 0; }
.fa-done .saved { font-family: var(--font); font-size: 13px; color: var(--muted); margin-top: 6px; }
.fa-coach { margin-top: 22px; padding: 18px 20px; border-radius: 16px; background: rgba(12,14,13,0.55); border-left: 4px solid var(--accent); }
.fa-coach .who { font-family: var(--font); font-weight: 700; font-size: 15px; color: var(--accent); margin-bottom: 8px; }
.fa-coach .text { font-family: var(--font); font-size: 16px; line-height: 1.6; color: var(--text); max-width: 78ch; }
.fa-coach .note { font-family: var(--font); font-size: 13px; color: var(--amber); margin-top: 10px; }
.fa-tip { margin-top: 14px; font-family: var(--font); font-size: 14px; line-height: 1.55; color: var(--muted); max-width: 78ch; }
.fa-tip b { color: var(--text); font-weight: 700; }
.fa-foot { margin-top: 30px; text-align: center; font-family: var(--font); font-size: 12px; color: var(--muted); }
</style>
"""


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _h(markup):
    """Collapse indentation so Streamlit's markdown never treats HTML as code."""
    return " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())


def _html(markup):
    st.markdown(_h(markup), unsafe_allow_html=True)


def _e(text):
    return html.escape(str(text), quote=True)


def _bolt_svg(size=42):
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 40 40" aria-hidden="true">'
        '<rect width="40" height="40" rx="12" fill="#3ee6a0"/>'
        '<path d="M22.8 6.5 10.5 22.2h8.2l-2 11.3 12.8-16.4h-8.4z" fill="#0c0e0d"/>'
        "</svg>"
    )


def _info_svg():
    return (
        '<svg width="18" height="18" viewBox="0 0 20 20" fill="none" aria-hidden="true">'
        '<circle cx="10" cy="10" r="8.5" stroke="#3ee6a0" stroke-width="1.5"/>'
        '<path d="M10 9v5M10 6.2v.1" stroke="#3ee6a0" stroke-width="1.7" stroke-linecap="round"/>'
        "</svg>"
    )


def _signature(snap):
    """Everything that changes the page layout (not the live numbers)."""
    return (
        snap.phase, snap.coach_status, snap.worker_alive,
        snap.error, snap.notice, snap.saved, snap.history_error,
    )


def _poll_interval(snap):
    if snap.phase in (PHASE_ACTIVE, PHASE_STOPPING):
        return 0.12
    if snap.phase in (PHASE_CHECKING, PHASE_SCHEDULED):
        return 0.5
    if snap.phase == PHASE_COMPLETED and (snap.worker_alive or snap.coach_status == "pending"):
        return 0.5
    if snap.worker_alive:
        return 0.3
    return None


# ----------------------------------------------------------------------------
# button callbacks (run before the script reruns, so the next render is fresh)
# ----------------------------------------------------------------------------
def _cb_start():
    ss = st.session_state
    mode = MODE_SCHEDULE if ss.get("mode") == MODE_LABELS[0] else MODE_NOW
    ok, message = get_controller().start(
        ss.get("exercise", EXERCISES[0]),
        ss.get("target_reps", DEFAULT_TARGET_REPS),
        mode,
        ss.get("alarm_time", dtime(6, 30)),
    )
    ss["flash"] = "" if ok else message


def _cb_stop():
    get_controller().stop()


def _cb_reset():
    get_controller().reset()
    st.session_state["flash"] = ""


# ----------------------------------------------------------------------------
# page sections
# ----------------------------------------------------------------------------
def render_header(snap, ai):
    label, tone = STATUS_CHIP.get(snap.phase, ("Ready", "ok"))
    if snap.phase == PHASE_ACTIVE:
        status_class = "live"
    else:
        status_class = tone

    ai_state = ai.get("state")
    if ai_state == "online":
        ai_label, ai_tone = "Groq online", "ok"
    elif ai_state == "model_missing":
        ai_label, ai_tone = "Groq unavailable", "warn"
    else:
        ai_label, ai_tone = "Groq offline", "warn"

    _html(f"""
    <div class="fa-header">
      <div class="fa-brand">
        {_bolt_svg()}
        <div>
          <div class="fa-name">AI FitAlarm</div>
          <div class="fa-tag-line">Wake Up. Move. Conquer.</div>
        </div>
      </div>
      <div class="fa-chips">
        <span class="fa-chip {ai_tone}" title="{_e(ai.get('detail', ''))}"><i></i>{_e(ai_label)}</span>
        <span class="fa-chip {status_class}"><i></i>{_e(label)}</span>
      </div>
    </div>
    """)


def render_setup(snap):
    ss = st.session_state
    busy = snap.phase in (PHASE_CHECKING, PHASE_SCHEDULED, PHASE_ACTIVE, PHASE_STOPPING) \
        or snap.worker_alive
    schedule_mode = ss.get("mode") == MODE_LABELS[0]

    with st.container(key="setup"):
        _html('<div class="fa-h">When should it start?</div>')
        st.radio(
            "When to start", MODE_LABELS, key="mode", horizontal=True,
            disabled=busy, label_visibility="collapsed",
        )

        st.time_input(
            "Alarm time", key="alarm_time", step=60,
            disabled=busy or not schedule_mode,
        )
        if schedule_mode and not busy:
            alarm_time = ss.get("alarm_time", dtime(6, 30))
            target = next_alarm_datetime(alarm_time.hour, alarm_time.minute)
            day = "today" if target.date() == datetime.now().date() else "tomorrow"
            _html(f'<div class="fa-sub">Rings {day} at {target:%H:%M}. Keep this app running.</div>')
        elif not schedule_mode:
            _html('<div class="fa-sub">The alarm and camera start as soon as you press Start Workout.</div>')

        _html('<div class="fa-gap"></div><div class="fa-h">Exercise</div>')
        st.radio(
            "Exercise", EXERCISES, key="exercise",
            disabled=busy, label_visibility="collapsed",
        )
        _html(f"""
        <div class="fa-help">{_info_svg()}<div>{_e(EXERCISE_HELP[ss.get("exercise", EXERCISES[0])])}</div></div>
        <div class="fa-gap"></div>
        """)

        st.number_input(
            "Target repetitions", min_value=MIN_TARGET_REPS, max_value=MAX_TARGET_REPS,
            step=1, key="target_reps", disabled=busy,
        )
        _html('<div class="fa-gap"></div>')

        st.button(
            "Start Alarm" if schedule_mode else "Start Workout",
            key="btn_start", disabled=busy, on_click=_cb_start,
            help="The alarm rings and the camera opens when the workout begins.",
        )
        col_stop, col_reset = st.columns(2)
        with col_stop:
            st.button(
                "Cancel alarm" if snap.phase == PHASE_SCHEDULED else "Stop workout",
                key="btn_stop", on_click=_cb_stop,
                disabled=snap.phase not in (PHASE_CHECKING, PHASE_SCHEDULED, PHASE_ACTIVE),
            )
        with col_reset:
            st.button("Reset", key="btn_reset", on_click=_cb_reset, disabled=busy)


def render_banners(snap):
    flash = st.session_state.get("flash", "")
    if flash:
        _html(f'<div class="fa-banner err"><b>Cannot start.</b> {_e(flash)}</div>')
    if snap.error:
        _html(f'<div class="fa-banner err"><b>Something went wrong.</b> {_e(snap.error)}</div>')
    if snap.notice:
        _html(f'<div class="fa-banner warn">{_e(snap.notice)}</div>')


def _video_html(snap):
    """The big preview area: live frame, countdown or a waiting message."""
    if snap.frame_jpeg and snap.phase in (PHASE_ACTIVE, PHASE_STOPPING, PHASE_COMPLETED):
        encoded = base64.b64encode(snap.frame_jpeg).decode("ascii")
        alarm_text = ALARM_LABEL.get(snap.alarm_state, "")
        alarm_pill = f'<span class="fa-pill">{_e(alarm_text)}</span>' if alarm_text else ""
        movement = (
            f'<span class="fa-pill accent">{_e(snap.movement)}</span>'
            if snap.movement else ""
        )
        return f"""
        <div class="fa-video">
          <img src="data:image/jpeg;base64,{encoded}" alt="Live webcam preview with pose skeleton"/>
          <div class="fa-overlay">
            <div class="group"><span class="fa-pill">{_e(snap.exercise)}</span>{alarm_pill}</div>
            <div class="group">{movement}</div>
          </div>
        </div>
        """

    if snap.phase == PHASE_SCHEDULED and snap.scheduled_for is not None:
        day = "today" if snap.scheduled_for.date() == datetime.now().date() else "tomorrow"
        return f"""
        <div class="fa-video"><div class="fa-wait">
          <div class="big">{format_remaining(snap.seconds_remaining)}</div>
          <div class="title">Alarm set for {snap.scheduled_for:%H:%M} {day}</div>
          <div class="note">At that time the alarm rings and your camera opens. Complete
          {snap.target_reps} {_e(snap.exercise.lower())} to switch it off.</div>
        </div></div>
        """

    if snap.phase == PHASE_CHECKING:
        return """
        <div class="fa-video"><div class="fa-wait pulse">
          <div class="title">Checking camera and model…</div>
          <div class="note">This takes a moment.</div>
        </div></div>
        """

    if snap.phase in (PHASE_ACTIVE, PHASE_STOPPING):
        return """
        <div class="fa-video"><div class="fa-wait pulse">
          <div class="title">Starting camera…</div>
          <div class="note">Step back until your full body is in view.</div>
        </div></div>
        """

    return """
    <div class="fa-video"><div class="fa-wait">
      <div class="title">Camera preview</div>
      <div class="note">Your webcam feed with the pose skeleton appears here once the
      workout starts. Camera and pose processing run locally. Squat measurements are sent securely to Groq for AI validation.</div>
    </div></div>
    """


def _hud_html(snap):
    target = max(1, snap.target_reps)
    count = min(snap.count, target)
    pct = int(round(100 * count / target))

    if snap.phase == PHASE_IDLE:
        feedback = "Choose an exercise, set your target and start."
    elif snap.phase == PHASE_SCHEDULED:
        feedback = "Alarm is set. Keep this app running."
    elif snap.phase == PHASE_CHECKING:
        feedback = "Checking camera and model…"
    else:
        feedback = snap.feedback or "Get into position."

    ai_line = ""
    if snap.ai_message and snap.phase == PHASE_ACTIVE and snap.exercise == "Squats":
        ai_line = f'<div class="fa-ai"><b>AI check</b> {_e(snap.ai_message)}</div>'

    detail = _e(snap.detail) if snap.detail and snap.phase == PHASE_ACTIVE else "&nbsp;"
    return f"""
    <div class="fa-hud">
      <div class="fa-count">
        <span class="num">{count:02d}</span><span class="of">/ {target}</span>
      </div>
      <div class="fa-meta">
        <div class="fa-bar" role="progressbar" aria-valuemin="0" aria-valuemax="{target}" aria-valuenow="{count}"><i style="width:{pct}%"></i></div>
        <div class="row"><span>{_e(snap.exercise)}: {count} of {target} repetitions</span><span>{detail}</span></div>
        <div class="fa-feedback">{_e(feedback)}</div>
        {ai_line}
      </div>
    </div>
    """


def render_live(base_signature):
    """Auto-refreshing part. Reads the newest frame and counters from the controller."""
    snap = get_controller().snapshot()

    # If the session changed phase (completed, stopped, error...) redraw the whole page
    # so buttons, banners and the coach card match.
    if _signature(snap) != base_signature:
        st.rerun()

    if USE_DATA_URI_FRAME or not snap.frame_jpeg:
        _html(_video_html(snap))
    else:
        st.image(snap.frame_jpeg)
    _html(_hud_html(snap))


def render_completion(snap):
    exercise = snap.exercise
    if snap.saved:
        saved_line = "Saved to workout history."
    elif snap.history_error:
        saved_line = f"The workout finished, but history could not be saved: {snap.history_error}"
    else:
        saved_line = "Saving to workout history…"

    if snap.coach_status == "ready":
        coach_text = _e(snap.coach_summary)
        note = (
            '<div class="note">Groq AI was unavailable, so this is the standard summary.</div>'
            if snap.coach_fallback else ""
        )
    else:
        coach_text = "Your coach is writing a short summary…"
        note = ""

    _html(f"""
    <div class="fa-done">
      <h2>Workout complete</h2>
      <p class="lead">{snap.count} of {snap.target_reps} {_e(exercise.lower())} done. The alarm is off.</p>
      <div class="saved">{_e(saved_line)}</div>
      <div class="fa-coach">
        <div class="who">AI Fitness Coach</div>
        <div class="text">{coach_text}</div>
        {note}
      </div>
      <div class="fa-tip"><b>Next session:</b> {_e(get_exercise_tip(exercise))}</div>
    </div>
    """)
    _html('<div class="fa-gap"></div>')
    _, middle, _ = st.columns([1, 1, 1])
    with middle:
        st.button("Start another session", key="btn_new", on_click=_cb_reset)


# ----------------------------------------------------------------------------
# page
# ----------------------------------------------------------------------------
def main():
    ss = st.session_state
    ss.setdefault("mode", MODE_LABELS[0])
    ss.setdefault("exercise", EXERCISES[0])
    ss.setdefault("target_reps", DEFAULT_TARGET_REPS)
    ss.setdefault("alarm_time", dtime(6, 30))
    ss.setdefault("flash", "")

    st.markdown(CSS, unsafe_allow_html=True)

    controller = get_controller()
    snap = controller.snapshot()
    ai = controller.ai_status()

    render_header(snap, ai)

    left, right = st.columns([5, 8], gap="large")
    with left:
        render_setup(snap)
    with right:
        render_banners(snap)
        live = st.fragment(run_every=_poll_interval(snap))(render_live)
        live(_signature(snap))

    if snap.phase == PHASE_COMPLETED:
        render_completion(snap)

    _html('<div class="fa-foot">Camera and pose detection run locally. AI squat validation and coaching use the Groq cloud API.</div>')


main()

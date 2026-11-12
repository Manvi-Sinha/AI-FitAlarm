
import json
import os
import re
import threading

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MAX_ANGLE_HISTORY = 180

_client = None
_client_lock = threading.Lock()

TOOLS = [{
    "type": "function",
    "function": {
        "name": "propose_squat_decision",
        "description": "Assess one completed squat candidate using the supplied measurements.",
        "parameters": {
            "type": "object",
            "properties": {
                "is_valid_squat": {"type": "boolean"},
                "reason": {"type": "string"},
                "feedback": {"type": "string"}
            },
            "required": ["is_valid_squat", "reason", "feedback"]
        }
    }
}]

EXERCISE_TIPS = {
    "squats": (
        "Lower with control, keep your knees tracking in the "
        "direction of your toes, and use a comfortable depth."
    ),
    "jumping jacks": (
        "Maintain a steady rhythm, coordinate your arms and legs, "
        "and land softly."
    ),
    "high knees": (
        "Stay upright, alternate your knees smoothly, "
        "and choose a controlled pace."
    )
}

DEFAULT_TIP = (
    "Keep a controlled pace and focus on smooth movement."
)


def get_client():
    """Create the Groq client only when needed."""
    global _client

    with _client_lock:
        if _client is None:
            api_key = os.getenv("GROQ_API_KEY")

            if not api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is missing. Check your .env file."
                )

            _client = Groq(
                api_key=api_key,
                timeout=15.0,
                max_retries=1
            )

        return _client


def get_exercise_tip(exercise):
    return EXERCISE_TIPS.get(
        str(exercise).strip().lower(),
        DEFAULT_TIP
    )


def check_groq_status(timeout=5):
    """Check Groq API connectivity without blocking the UI."""
    outcome = {}

    def probe():
        try:
            client = get_client()
            client.models.list()
            outcome["online"] = True
        except Exception as error:
            outcome["error"] = str(error)[:180]

    worker = threading.Thread(target=probe, daemon=True)
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        return {
            "state": "offline",
            "detail": "Groq API check timed out."
        }

    if "error" in outcome:
        return {
            "state": "offline",
            "detail": outcome["error"]
        }

    return {
        "state": "online",
        "detail": f"Groq connected. Model: {MODEL}"
    }


# Compatibility with the existing session_manager.py.
def check_ollama_status(timeout=5):
    return check_groq_status(timeout)


def _validate_session(session):
    completed = int(session["completed_reps"])
    target = int(session["target_reps"])

    if not 1 <= target <= 100:
        raise ValueError("Target must be between 1 and 100.")

    if not 0 <= completed <= target:
        raise ValueError("Invalid completed repetition count.")

    raw_angles = session.get("knee_angles", [])

    if not isinstance(raw_angles, list):
        raw_angles = []

    angles = []

    for value in raw_angles[-MAX_ANGLE_HISTORY:]:
        try:
            angle = float(value)
        except (TypeError, ValueError):
            continue

        if 0 <= angle <= 180:
            angles.append(round(angle, 1))

    try:
        duration = float(
            session.get("candidate_duration_seconds", 0)
        )
    except (TypeError, ValueError):
        duration = 0.0

    return {
        "candidate_id": session.get("candidate_id"),
        "completed_reps": completed,
        "target_reps": target,
        "knee_angles": angles,
        "movement_state": str(
            session.get("movement_state", "UNKNOWN")
        )[:40],
        "movement_feedback": str(
            session.get("movement_feedback", "")
        )[:200],
        "candidate_duration_seconds": duration,
        "reached_depth": bool(
            session.get("reached_depth", False)
        ),
        "returned_to_standing": bool(
            session.get("returned_to_standing", False)
        ),
        "standing_threshold_degrees": 160,
        "squat_depth_threshold_degrees": 100,
        "minimum_duration_seconds": 0.5
    }


def decide_next_action(session):
    """Ask Groq to assess a squat candidate."""

    data = _validate_session(session)

    if data["completed_reps"] >= data["target_reps"]:
        return {
            "action": "COMPLETE",
            "is_valid_squat": False,
            "message": "Target reached.",
            "reason": "Application controls target completion."
        }

    angles = data["knee_angles"]

    if len(angles) < 6:
        return {
            "action": "CONTINUE",
            "is_valid_squat": False,
            "message": "Not enough movement data. Try another controlled squat.",
            "reason": "Insufficient angle sequence."
        }

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI squat-validation assistant. "
                "Assess one completed squat using the supplied knee-angle "
                "sequence. A valid squat should start near standing, "
                "lower below 100 degrees, and return above 160 degrees. "
                "Check the supplied measurements and movement flags. "
                "Do not invent measurements. If evidence is ambiguous, "
                "propose false. Python performs the final integrity checks "
                "and controls the repetition counter."
            )
        },
        {
            "role": "user",
            "content": json.dumps(data)
        }
    ]

    try:
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice={
                "type": "function",
                "function": {
                    "name": "propose_squat_decision"
                }
            },
            temperature=0
        )

        message = response.choices[0].message
        calls = message.tool_calls or []

        for call in calls:
            if call.function.name != "propose_squat_decision":
                continue

            args = json.loads(call.function.arguments)

            valid = args.get("is_valid_squat")

            if not isinstance(valid, bool):
                valid = False

            return {
                "action": "PROPOSE_VALID" if valid else "CONTINUE",
                "is_valid_squat": valid,
                "message": str(
                    args.get("feedback", "Move with control.")
                )[:240],
                "reason": str(
                    args.get("reason", "No explanation supplied.")
                )[:240]
            }

        raise ValueError("Groq returned no squat decision.")

    except Exception as error:
        return {
            "action": "CONTINUE",
            "is_valid_squat": False,
            "message": "AI validation unavailable. Try again.",
            "reason": str(error)[:180]
        }


def generate_workout_summary(workout):
    """Generate an AI-powered post-workout coaching summary."""

    exercise = str(workout.get("exercise", "Workout"))[:40]
    completed = int(workout.get("completed_reps", 0))
    target = int(workout.get("target_reps", 0))
    finished = bool(workout.get("completed", False))

    tip = get_exercise_tip(exercise)

    data = {
        "exercise": exercise,
        "completed_reps": completed,
        "target_reps": target,
        "completed": finished,
        "suggested_tip": tip
    }

    prompt = (
        "Write exactly three short sentences as a warm exercise coach. "
        "First acknowledge the exercise and repetition result. "
        "Second give the supplied exercise-specific tip. "
        "Third offer safe encouragement. Output only a normal paragraph. "
        "Do not use headings or bullet points. Use only the supplied data. "
        "Do not claim to have measured posture, calories, heart rate, "
        "or health status. Never recommend pushing through pain. "
        "Workout data: " + json.dumps(data)
    )

    try:
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful, concise and encouraging "
                        "exercise coach."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=180
        )

        summary = response.choices[0].message.content or ""

        summary = re.sub(
            r"(?i)\b(?:sentence\s*[1-3]|[1-3][.)])\s*[:.-]?\s*",
            "",
            summary
        )

        summary = re.sub(r"\s+", " ", summary).strip()

        return summary[:700]

    except Exception as error:
        return (
            f"Your {exercise} session has ended: "
            f"{completed}/{target} repetitions. "
            f"{tip} Rest if you feel discomfort. "
            f"(AI unavailable: {str(error)[:100]})"
        )

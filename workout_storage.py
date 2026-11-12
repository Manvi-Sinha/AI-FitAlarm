"""workout_storage.py

Automatic local saving of completed workouts to workout_history.json.

The record structure is exactly the one used by the original application:
    {"date", "time", "exercise", "completed_reps", "target_reps", "completed"}
"""

import json
import os
import tempfile
import threading
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(BASE_DIR, "workout_history.json")

_lock = threading.Lock()
_saved_session_ids = set()


def load_history(path=None):
    """Return the saved workouts as a list (empty if missing or unreadable)."""
    path = path or HISTORY_PATH
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def build_record(exercise, completed_reps, target_reps, now=None):
    """Create one history record in the original structure."""
    now = now or datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "exercise": str(exercise),
        "completed_reps": int(completed_reps),
        "target_reps": int(target_reps),
        "completed": bool(int(completed_reps) >= int(target_reps)),
    }


def _backup_unreadable_file(path):
    """Keep a corrupt history file instead of silently overwriting it."""
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            return
    except (OSError, json.JSONDecodeError):
        pass
    backup = f"{path}.corrupt-{datetime.now():%Y%m%d-%H%M%S}"
    try:
        os.replace(path, backup)
    except OSError:
        pass


def save_workout_session(record, session_id=None, path=None):
    """Append one completed workout to the history file.

    Returns (ok, error_message). Passing the same session_id twice saves only
    once, which protects against duplicate history entries.
    """
    path = path or HISTORY_PATH
    with _lock:
        if session_id is not None and session_id in _saved_session_ids:
            return True, None

        try:
            _backup_unreadable_file(path)
            history = load_history(path)
            history.append(record)

            directory = os.path.dirname(path) or "."
            fd, temp_path = tempfile.mkstemp(
                prefix="workout_history_", suffix=".tmp", dir=directory
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as file:
                    json.dump(history, file, indent=4, ensure_ascii=False)
                os.replace(temp_path, path)
            except Exception:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise

            if session_id is not None:
                _saved_session_ids.add(session_id)
            return True, None
        except OSError as error:
            return False, str(error)

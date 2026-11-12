import queue
import threading

from local_agent import decide_next_action, generate_workout_summary


class AgentBridge:
    """Runs Groq AI requests in background threads."""

    def __init__(self, output_queue=None):
        self.output_queue = (
            output_queue if output_queue is not None else queue.Queue()
        )

        self.results = queue.Queue()
        self.requests = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()

        self.worker = threading.Thread(
            target=self._worker_loop,
            daemon=True
        )
        self.worker.start()

    def submit(self, session):
        if self.stop_event.is_set():
            return False

        try:
            self.requests.put_nowait(dict(session))
            return True
        except queue.Full:
            return False

    def get_result_nowait(self):
        return self.results.get_nowait()

    def _worker_loop(self):
        while not self.stop_event.is_set():

            try:
                session = self.requests.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                result = decide_next_action(session)
                result["candidate_id"] = session.get("candidate_id")

                self.results.put(result)
                self.output_queue.put(("agent_feedback", result))

            except Exception as error:
                result = {
                    "candidate_id": session.get("candidate_id"),
                    "is_valid_squat": False,
                    "message": "AI validation failed.",
                    "reason": str(error)
                }

                self.results.put(result)
                self.output_queue.put(("agent_error", result))

    def request_summary(self, workout):
        """Generate an AI workout summary without blocking the GUI."""

        def task():
            try:
                summary = generate_workout_summary(workout)
            except Exception as error:
                summary = (
                    "Workout completed! Keep movements controlled "
                    f"and rest when needed. AI Coach error: {error}"
                )

            self.output_queue.put(("coach_summary", summary))

        if not self.stop_event.is_set():
            threading.Thread(
                target=task,
                daemon=True
            ).start()

    def close(self):
        self.stop_event.set()
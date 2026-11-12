# AI FitAlarm

**Wake Up. Move. Conquer.** An alarm that only stops when you finish your reps.
Squats, jumping jacks or high knees, counted from your webcam with MediaPipe pose
detection. Squats are verified by a local Qwen 2.5 3B model; an AI coach writes a
short summary afterwards. Everything runs on your computer.

## Folder structure

```
AI-FitAlarm/
├── app.py                       Streamlit interface (draws state, forwards clicks)
├── session_manager.py           NEW: owns the camera thread, schedule, alarm, AI bridge
├── exercise_engine.py           MediaPipe + the three exercise counters (ported from main.py)
├── local_agent.py               Qwen squat validation + AI coach (your file, 2 helpers added)
├── agent_bridge.py              Background Qwen worker (your file, unchanged)
├── alarm_manager.py             Melody + non-blocking winsound playback + schedule helpers
├── workout_storage.py           workout_history.json saving
├── selftest.py                  Offline self-test (no camera / Ollama / browser needed)
├── .streamlit/config.toml       Dark theme
├── pose_landmarker_lite.task    <- copy YOUR existing model file here
├── requirements.txt
└── README.md
```

`workout_history.json` and `ai_alarm_melody.wav` are created automatically.

## Install (Windows, Python 3.11)

```
cd AI-FitAlarm
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `pose_landmarker_lite.task` into this folder.

## Check Ollama and Qwen

```
ollama --version
ollama list                 # must show qwen2.5:3b
ollama pull qwen2.5:3b      # only if it is missing
ollama run qwen2.5:3b "Say hi"
```

If Ollama is not running, start the Ollama app (or run `ollama serve`).
The green/amber "Qwen online / offline" chip in the app header shows the live status.

## Run

```
python selftest.py          # optional: offline logic check
streamlit run app.py
```

## Test each feature

1. **Squats.** Pick Squats, target 3, Start Workout. Stand with hips, knees and ankles
   visible, squat below parallel and stand up. The status reads "AI Checking", then the
   count goes up only if Qwen approves. Needs Ollama running.
2. **Jumping Jacks.** Start with feet together and arms down. Jump with arms overhead
   and feet wide (chip shows "Arms Open"), then return ("Arms Closed") = 1 rep.
3. **High Knees.** Raise the left knee to hip height, lower it, raise the right, lower it
   = 1 rep. One leg alone never counts.
4. **Alarm stops automatically.** On Windows the melody plays from the moment the camera
   opens and stops the instant the last rep is counted (header chip changes to Completed,
   the video pill says "Alarm stopped"). Use Stop workout to silence it manually.
5. **Scheduled alarm.** Choose "Schedule alarm", set a time 1-2 minutes ahead, Start Alarm.
   The countdown runs; at the time the alarm rings and the camera opens. Keep the
   `streamlit run` window open and the PC awake. Closing the browser tab is fine.
6. **History.** After completing a workout open `workout_history.json`; one new entry:
   `date, time, exercise, completed_reps, target_reps, completed`. Stopped or failed
   workouts are not saved (same as the original app).
7. **AI Fitness Coach.** Appears on the completion card. With Ollama stopped you get a
   standard summary plus a note that the local AI was unavailable.

## Troubleshooting

| Problem | Fix |
|---|---|
| "Model file not found" | Put `pose_landmarker_lite.task` next to `app.py`. |
| "Could not open the webcam" | Close Zoom/Teams/Camera app. Try `CAMERA_INDEX = 1` in `exercise_engine.py`. |
| Squats never count | Squats need Qwen. Check the header chip, `ollama list`, `ollama pull qwen2.5:3b`. |
| No sound | The alarm needs Windows (`winsound`). A banner explains if playback failed. |
| Video area stays empty but counting works | Set `USE_DATA_URI_FRAME = False` in `app.py`. |
| Cards or buttons look unstyled | The CSS targets Streamlit >= 1.40 internals. Upgrade: `pip install -U streamlit`. Function is unaffected. |
| "Previous session still releasing the camera" | Wait a second and press Start again. |
| `ModuleNotFoundError` | Activate the venv and run `pip install -r requirements.txt`. |

## Design notes and limitations

* **Camera ownership.** One `WorkoutController` (module singleton) runs one camera thread.
  Streamlit reruns only read its snapshot, so reruns cannot open a second camera or thread.
  The live area is an auto-refreshing fragment (about 8 updates per second); detection runs
  at full camera speed on the background thread regardless of UI refresh.
* **Local only.** The webcam and alarm belong to the machine running Streamlit. Do not host
  this on a remote server. All browser tabs show the same single session.
* **Squats depend on Qwen** (original behaviour). If Ollama is down, squat reps cannot be
  counted. The app warns you before the workout starts; Stop workout always works.
* **Camera failure at alarm time.** As in the original app, if the camera cannot open when a
  scheduled alarm fires, an error is shown and the alarm does not ring.
* **Changed from the original:** after the target is reached the camera is released after
  1 s instead of 5 s (the Streamlit completion card replaces the OpenCV window);
  up to ~1 s of failed camera reads is tolerated while it warms up; counters freeze at the
  target so a rep can never be counted twice; a corrupt history file is backed up instead
  of overwritten. All thresholds, state machines and the Qwen prompts are unchanged.

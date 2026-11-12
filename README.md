# ⚡ AI FitAlarm — Wake Up. Move. Conquer.

**AI FitAlarm** is an AI-powered fitness alarm system that encourages users to get out of bed and complete physical exercises before turning off their alarm.

Instead of dismissing an alarm with a simple tap, users must complete a configured number of exercises. The system uses computer vision for real-time exercise tracking and Groq-powered AI validation for squat repetitions.

## ✨ Features

- **Smart Fitness Alarm:** An alarm that requires users to complete physical activity.
- **Real-Time Pose Detection:** Uses MediaPipe Pose Landmarker and OpenCV to track body landmarks through a webcam.
- **AI-Assisted Squat Validation:** Groq AI evaluates squat movement data and proposes validation decisions before repetitions are counted.
- **Multiple Exercises:** Supports Squats, Jumping Jacks and High Knees.
- **Automated Repetition Counting:** Tracks exercise movements and counts valid repetitions.
- **Customizable Targets:** Allows users to select their exercise and repetition target.
- **Alarm Scheduling:** Schedule an alarm for a preferred time or start a workout immediately.
- **AI Workout Coach:** Provides post-workout feedback and exercise guidance.
- **Workout History:** Stores completed workout sessions locally.
- **Non-Blocking Processing:** Uses background threads to manage webcam processing, AI requests and alarm operations.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Computer Vision | OpenCV |
| Pose Estimation | MediaPipe Pose Landmarker |
| AI Integration | Groq API |
| AI Model | `openai/gpt-oss-20b` |
| Numerical Processing | NumPy |
| Environment Configuration | python-dotenv |
| Alarm Management | Python threading, winsound |
| Data Storage | JSON |
| Development Environment | VS Code |

## 🧠 System Architecture

AI FitAlarm follows a modular architecture in which the user interface, exercise detection, AI communication, alarm management and session handling are separated.

### Workflow

1. **User Interaction:** The user selects an exercise, repetition target and alarm mode through the Streamlit interface.
2. **Session Management:** The WorkoutController manages the workout lifecycle, including starting, scheduling, stopping and completing sessions.
3. **Pose Detection:** OpenCV captures webcam frames, while MediaPipe estimates body landmarks.
4. **Exercise Recognition:** The exercise engine processes landmarks and applies movement-specific rules.
5. **AI Validation:** For squats, movement measurements are passed through AgentBridge to Groq AI for a proposed validation decision.
6. **Repetition Counting:** The exercise engine checks the AI decision along with its own movement-validation conditions before counting a squat.
7. **Alarm Control:** The alarm continues until the required exercise target is completed.
8. **AI Coaching and History:** The system generates workout feedback and saves completed session records locally.

### Architecture Overview

```text
             Streamlit UI
                  |
                  v
          WorkoutController
                  |
                  v
           Exercise Engine
                  |
          +-------+-------+
          |               |
          v               v
     OpenCV Camera    Exercise Trackers
          |               |
          v               v
    MediaPipe Pose    Movement Data
                          |
                          v
                     AgentBridge
                          |
                          v
                       Groq API
                          |
                          v
                 Squat Validation
                          |
                          v
                  Rep Count Update
                          |
                          v
                    Alarm Manager
                          |
                          v
                  Workout Completion
                          |
                 +--------+--------+
                 |                 |
                 v                 v
             AI Coach         Workout History
```

## 📁 Project Structure

```text
AI-FitAlarm/
│
├── .streamlit/
│   └── config.toml
│
├── app.py
├── session_manager.py
├── exercise_engine.py
├── local_agent.py
├── agent_bridge.py
├── alarm_manager.py
├── workout_storage.py
│
├── pose_landmarker_lite.task
├── requirements.txt
├── README.md
│
└── selftest.py
```

### Main Components

| File | Responsibility |
|---|---|
| `app.py` | Streamlit user interface |
| `session_manager.py` | Workout lifecycle and session management |
| `exercise_engine.py` | Pose processing and exercise tracking |
| `local_agent.py` | Groq API integration, AI validation and coaching |
| `agent_bridge.py` | Background AI request handling |
| `alarm_manager.py` | Alarm playback and scheduling |
| `workout_storage.py` | Workout history and JSON storage |
| `selftest.py` | Automated offline tests |

## ⚙️ Installation and Setup

### Prerequisites

- Windows operating system
- Python 3.13
- Working webcam
- Internet connection for Groq AI features
- Groq API key

### 1. Clone the Repository

```bash
git clone https://github.com/Manvi-Sinha/AI-FitAlarm.git
```

Navigate into the project directory:

```bash
cd AI-FitAlarm
```

### 2. Create a Virtual Environment

```powershell
py -3.13 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the MediaPipe Model

Ensure that `pose_landmarker_lite.task` is present in the project root directory.

If it is not included in your downloaded project, obtain the compatible MediaPipe Pose Landmarker Lite model and place it there.

### 5. Configure Groq API

Create a `.env` file in the root directory.

Add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Replace the placeholder with your own API key.

**Important:** Never upload your `.env` file or expose your API key publicly.

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔍 Exercise Tracking

AI FitAlarm currently supports three exercises:

**1. Squats**
- Tracks hip, knee and ankle landmarks.
- Calculates knee angles.
- Checks squat depth and return to standing.
- Uses Groq AI to propose a validation decision.
- Applies additional local validation before counting repetitions.

**2. Jumping Jacks**
- Tracks arm and leg positions.
- Identifies open and closed movement states.
- Counts completed movement cycles.

**3. High Knees**
- Tracks left and right knee elevation.
- Identifies alternating knee movements.
- Counts completed repetitions based on movement rules.

## 🔐 Privacy and Data Handling

- Webcam capture and pose processing are performed locally.
- Squat movement measurements are sent to the Groq cloud API for AI validation.
- AI coaching requests use the Groq API.
- Workout history is stored locally in JSON format.
- API credentials are stored in a local `.env` file.

## ⚠️ Current Limitations

- Exercise detection depends on camera positioning, lighting and visibility of body landmarks.
- Groq-powered features require an internet connection and a valid API key.
- Windows alarm playback uses the `winsound` module.
- AI validation is used alongside local movement checks; it does not replace them.
- The system currently supports three exercises.

## 🚀 Future Enhancements

- Add more exercise categories.
- Improve exercise validation and feedback.
- Enhance the user interface and workout analytics.
- Add personalized workout recommendations.
- Improve session reliability and background-thread cleanup.
- Expand workout history and performance tracking.

## 👩‍💻 Developer

**Manvi Sinha**

B.Tech — Computer Science and Engineering (AI & ML)

Narula Institute of Technology

**Project:** AI FitAlarm — Wake Up. Move. Conquer.

---

*Built with Python, Computer Vision and AI to make waking up more productive.*
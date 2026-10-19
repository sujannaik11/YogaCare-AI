# 🧘 YogaCare AI — AI-Powered Yoga Posture Detection & Guidance

**YogaCare AI** is a computer vision-based yoga and wellness application that helps users practise yoga with guided instructions and real-time posture analysis. It uses pose estimation, joint-angle calculations, posture evaluation, and voice feedback to support a more interactive yoga practice.

## 🌟 Overview

The project combines artificial intelligence, computer vision, and wellness guidance in one application. Users can explore yoga practices by category, view reference images, follow guided routines, and use webcam-based pose tracking for supported asanas.

## ✨ Key Features

- **40 Yoga Practices:** A categorized yoga library with reference images.
- **7 Practice Categories:** Warm-up and mobility, standing poses, seated poses, kneeling and back mobility, strength and core, flexibility and recovery, and breathing and meditation.
- **Real-Time Pose Detection:** Webcam-based body landmark detection and skeleton visualization.
- **Posture Analysis:** Joint-angle calculations and rule-based feedback for supported yoga poses.
- **Live Feedback:** Posture scores, corrective guidance, and pose-hold timing.
- **Bilingual Instructions:** English and Kannada guidance.
- **Voice Assistant:** Spoken instructions and coaching cues.
- **Guided Yoga Routine:** A 22-posture sequence with phase-wise navigation.
- **Workout History:** Session logging and performance summaries.
- **Modular Architecture:** Separate components for detection, evaluation, voice guidance, routines, and analytics.

## 🧠 AI and Computer Vision

### 1. YOLOv8-Pose
Uses the pretrained Ultralytics YOLOv8 nano pose-estimation model to detect body keypoints from webcam frames.

### 2. MediaPipe Pose Landmarker
Provides an alternative pose-estimation engine for detecting body landmarks.

### 3. Joint-Angle Calculation
Calculates angles between three body landmarks using vector mathematics to help evaluate posture alignment.

### 4. Posture Evaluation Engine
Applies pose-specific biomechanical rules to generate alignment feedback for supported poses, including Tree Pose, Warrior II, Downward-Facing Dog, Plank, Cobra, and Mountain Pose.

### 5. Exponential Moving Average Smoothing
Smooths angle measurements to reduce jitter in live webcam feedback.

**Note:** Pose estimation identifies body landmarks. The posture evaluation layer uses configured geometric and rule-based checks; it is not a medical diagnostic system.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| OpenCV | Webcam access and image processing |
| Ultralytics YOLOv8-Pose | Body keypoint detection |
| MediaPipe | Alternative pose landmark detection |
| NumPy | Numerical calculations |
| Streamlit | Web-based user interface |
| PyTorch | Deep learning model runtime |
| pyttsx3 | Text-to-speech guidance |
| pytest | Automated testing |

## 🗂️ Project Structure

```text
YogaCare-AI/
├── app.py
├── config.py
├── pose_detector.py
├── pose_reference.py
├── angle_calculator.py
├── posture_analyzer.py
├── yoga_library.py
├── routine_manager.py
├── voice_assistant.py
├── workout_logger.py
├── health_check.py
├── benchmark.py
├── test_live_posture.py
├── requirements.txt
├── README.md
├── VIVA_AND_PROJECT_REPORT.md
├── assets/
│   ├── category_thumbnails/
│   └── reference_poses/
├── models/
├── poses/
└── tests/
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10–3.13, subject to compatibility with installed dependencies
- A working webcam for live pose tracking
- Git (optional)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YogaCare-AI.git
cd YogaCare-AI
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

Some deep learning dependencies can be large. Install a compatible PyTorch build for your system if the default dependency installation fails.

### 4. Verify model files

Check the `models/` directory for the pose-estimation weights required by your selected detection engine. If model weights are not included in the repository, follow the project's model setup instructions.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`.

### 6. Run tests

```bash
python -m pytest tests/ -v
```

Some tests may require model weights, webcam access, or additional runtime dependencies.

## 🧪 Main Modules

| Module | Responsibility |
|---|---|
| `pose_detector.py` | Body landmark detection |
| `angle_calculator.py` | Joint-angle calculations and smoothing |
| `posture_analyzer.py` | Posture evaluation and feedback |
| `yoga_library.py` | Yoga practices and category navigation |
| `routine_manager.py` | Guided routine sequencing |
| `voice_assistant.py` | Spoken coaching cues |
| `workout_logger.py` | Workout history and analytics |
| `health_check.py` | Environment and system diagnostics |
| `benchmark.py` | Performance profiling |
| `tests/` | Automated unit and integration tests |

## 🚀 Future Enhancements

- Support for additional yoga poses and body types.
- More detailed posture feedback and accessibility options.
- Improved low-light and camera-angle handling.
- Personalized practice plans and progress visualizations.
- Expanded mobile and browser compatibility.

## ⚠️ Disclaimer

YogaCare AI is an educational project intended to support yoga practice. Its automated posture feedback may be inaccurate and should not replace advice from a qualified yoga instructor or healthcare professional. Practise within your comfort level and stop if you experience pain or dizziness.

## 👨‍💻 Author

**Sujan Naik**  
Computer Science and Engineering Student

## 📄 License

No license has been specified. Add a suitable license if you intend to permit others to reuse or modify the source code.

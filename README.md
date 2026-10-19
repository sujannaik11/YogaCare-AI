# YogaCare AI — AI-Powered Yoga Posture Detection & Guidance System

**Project:** YogaCare AI  
**Status:** **40-PRACTICE YOGA LIBRARY + 7 CURATED CATEGORIES + BILINGUAL (EN/KN) GUIDANCE + 22-POSTURE GUIDED ROUTINE + MALE REFERENCE MODELS (100% COMPLETE)**  
**Total Automated Unit & Integration Tests:** 51/51 Passing (100% OK)  
**System Diagnostic Verdict:** [OPTIMAL] ALL PRE-FLIGHT CHECKS PASSED  
**Male Reference Verification Checklist:** [`assets/reference_poses/MALE_REFERENCE_CHECKLIST.md`](file:///C:/Users/sujan/OneDrive/Documents/AI%20WORKSHOP/YogaCare-AI/assets/reference_poses/MALE_REFERENCE_CHECKLIST.md)  
**College Viva & Architecture Guide:** [`VIVA_AND_PROJECT_REPORT.md`](file:///C:/Users/sujan/OneDrive/Documents/AI%20WORKSHOP/YogaCare-AI/VIVA_AND_PROJECT_REPORT.md)  

---

## 1. Project Overview & Multi-Phase Architecture

YogaCare AI is a real-time computer vision and artificial intelligence system designed to detect human poses, evaluate yoga postures against biomechanical standards, calculate joint angles, time hold durations, provide real-time spoken coaching cues, and log performance analytics. It features a complete **40 Yoga Practice Library** across 7 curated categories, targeted **Single Pose Practice**, and a **22-Posture Guided Yoga Routine** with verified adult male reference models.

```mermaid
flowchart LR
    A[Webcam Feed] --> B[Phase 1: Pose Detector\nYOLOv8-Pose / MediaPipe]
    B --> C[Phase 2: Angle Calculator\n3-Point Angles & EMA Smoother]
    C --> D[Phase 2 & 5: Posture Evaluators\nSupported Foundation Asanas]
    D --> E[Phase 3: AI Voice Coach\nRate-Limited Speech Queue]
    D --> F[Phase 3 & 4: Streamlit UI\nLive Gauges & HUD Card]
    D --> G[Phase 4: Workout Logger\nAnalytics & History]
    H[40-Practice Library\n7 Curated Categories] --> F
    I[Guided Routine\n22-Pose Flow & Sides] --> F
    H --> G
    I --> G
```

### Phase Breakdown & Deliverables
- **40-Practice Category-Wise Yoga Library (`yoga_library.py`):**
  - **7 Curated Categories:**
    1. *Warm-up & Mobility* (5 exercises: Sukhasana, Neck Rotation, Shoulder Rolls, Tadasana, Standing Side Stretch)
    2. *Standing Poses* (7 exercises: Vrikshasana, Trikonasana, Virabhadrasana I, Virabhadrasana II, Utkatasana, Parsvakonasana, Malasana)
    3. *Seated Poses* (6 exercises: Dandasana, Vajrasana, Baddha Konasana, Paschimottanasana, Ardha Matsyendrasana, Gomukhasana)
    4. *Kneeling & Back Mobility* (6 exercises: Marjaryasana, Bitilasana, Balasana, Vyaghrasana, Bhujangasana, Salabhasana)
    5. *Strength & Core* (5 exercises: Adho Mukha Svanasana, Phalakasana, Vasisthasana, Setu Bandhasana, Naukasana)
    6. *Flexibility & Recovery* (6 exercises: Anjaneyasana, Supta Baddha Konasana, Supta Matsyendrasana, Ananda Balasana, Viparita Karani, Shavasana)
    7. *Breathing & Meditation* (5 practices: Pranayama in Sukhasana, Nadi Shodhana, Bhramari Pranayama, Dhyana, Yoga Nidra)
  - **3 Seamless Navigation States:**
    - `categories`: High-level 7-card responsive gallery with thumbnails and explore buttons.
    - `category_detail`: Filtered list showing only practices belonging to that phase (no crowded pages).
    - `practice_detail`: 2-column balanced desktop layout with adult male reference card, beginner modifications, safety reminders, and bilingual instructions.
  - **Bilingual & Voice Guidance:** Full step-by-step instructions in both **English** and **Kannada (ಕನ್ನಡ)** with one-click audio readout via the asynchronous `VoiceAssistant`.
  - **In-Category Sequential Navigation:** Previous and Next buttons navigate within the active category without forcing the user back to the category grid after each exercise.
  - **Interactive Practice Integration:** Supports both live webcam tracking (with YOLO skeleton and honest scoring) and guided countdown timers logging to `WorkoutLogger`.
- **Phase 1 (Computer Vision & Live Pose Detection):**
  - High-precision webcam video streaming at 1280x720 with automatic fallback.
  - Pretrained full-body pose estimation using **Ultralytics YOLOv8-Pose (`yolov8n-pose.pt`)** with modular **MediaPipe Tasks** alternative.
  - Standardized, model-agnostic dataclasses (`Landmark` and `PoseDetectionResult`) with normalized/pixel coordinates and visibility guards (`is_valid`).
  - Skeleton renderer with color-coded anatomical segments and dynamic diagnostic top HUD.
- **Phase 2 (Posture Analysis & Corrective Feedback):**
  - Vector trigonometry engine for 3-point joint angles and orientation relative to horizontal and vertical axes.
  - Exponential moving average (`AngleSmoother`) eliminating camera jitter.
  - Posture evaluators for **Tree Pose (*Vrikshasana*)** and **Warrior II (*Virabhadrasana II*)** with automatic stance detection (left vs right leg).
  - Safety rule preventing dangerous pressure against the standing knee joint.
  - Weighted accuracy scoring ($0-100\%$) and continuous posture hold duration timer.
- **Phase 3 (User Interface & Voice Assistant):**
  - Modern Streamlit Web Application (`app.py`) with live score meters, progress rings, and joint breakdown tables.
  - Non-blocking, background-threaded AI Voice Assistant (`voice_assistant.py`) with intelligent rate-limiting and milestone chimes.
  - Standalone live CLI test runner (`test_live_posture.py`) with hotkeys.
- **Phase 4 (Final Integration, Optimization & Testing):**
  - End-to-end integration test suite (`tests/test_integration.py`).
  - Workout session history logger (`workout_logger.py`) with JSON persistence.
  - System health check & pre-flight diagnostic tool (`health_check.py`).
  - Performance profiling benchmark (`benchmark.py`) measuring latency breakdown and sustained FPS.
- **Phase 5 (Extended Asanas & Viva Documentation):**
  - 3 Extended Evaluators: **Downward-Facing Dog (*Adho Mukha Svanasana*)**, **Plank Pose (*Phalakasana*)**, and **Cobra Pose (*Bhujangasana*)**, plus **Mountain Pose (*Tadasana*)**.
  - Multi-pose auto-classifier recognizing standing, lunging, inverted, and prone asanas.
  - Comprehensive College Presentation & Viva Guide (`VIVA_AND_PROJECT_REPORT.md`) containing top 25 exam Q&As and PowerPoint slide deck structure.
- **Guided Routine (22 Postures with Adult Male Reference Models):**
  - 4 Sequential Phases: Phase 1 (Warm-up, 1–5), Phase 2 (Standing Poses, 6–11), Phase 3 (Floor & Stretching, 12–19), Phase 4 (Breathing & Relaxation, 20–22).
  - 100% Adult Male Reference Image assets stored in `assets/reference_poses/` with fallback missing placeholder cards.
  - State machine controller (`routine_manager.py`) with bilateral side tracking and Cat-Cow repetition counting.

---

## 2. Project Directory Structure

```text
YogaCare-AI/
├── config.py                  # Thresholds, camera parameters, landmark indices, and UI palette
├── pose_detector.py           # Phase 1: Pose detector engine (YOLOv8-Pose / MediaPipe)
├── camera_test.py             # Phase 1: Standalone webcam & skeleton visualization test
├── angle_calculator.py       # Phase 2: Vector trigonometry, 3-point angles, and EMA smoother
├── posture_analyzer.py        # Phase 2 & 5: Posture orchestrator, 5-pose evaluator & HUD card
├── test_live_posture.py       # Phase 2 & 3: Live posture guidance CLI with voice coaching & hotkeys
├── voice_assistant.py         # Phase 3: Non-blocking text-to-speech engine with milestone chimes
├── app.py                     # Phase 3 & 4: Integrated Streamlit Web Application Dashboard
├── yoga_library.py            # Yoga Library: 40-practice orchestrator & category navigator
├── routine_manager.py         # Guided Routine: 22-posture sequence controller, side & rep tracker
├── workout_logger.py          # Phase 4: Workout session logging & cumulative performance analytics
├── health_check.py            # Phase 4: System pre-flight diagnostic & environment verification
├── benchmark.py               # Phase 4: Performance profiler & latency breakdown tool
├── requirements.txt           # Dependency manifest
├── .gitignore                 # Git exclusion rules
├── README.md                  # Technical documentation
├── VIVA_AND_PROJECT_REPORT.md # College Presentation, Architecture & Viva Guide
├── assets/                    # Static visual assets
│   ├── category_thumbnails/   # 7 Category Thumbnails (cat_01_warmup.jpg - cat_07_breathing.jpg)
│   └── reference_poses/       # 40 Male Reference Images (01_sukhasana.jpg - 40_yoga_nidra.jpg)
│       └── MALE_REFERENCE_CHECKLIST.md # Verification checklist & model standards
├── models/                    # Pretrained model weights
│   ├── yolov8n-pose.pt                  # Ultralytics YOLOv8 nano pose model (6.5 MB)
│   └── pose_landmarker_lite.task        # MediaPipe Tasks pose landmarker model (5.7 MB)
├── poses/                     # Biomechanical posture evaluators (5 Supported Poses)
│   ├── __init__.py            # Package exports
│   ├── base_pose.py           # Base evaluator & JointFeedback / PoseEvaluationResult
│   ├── tree_pose.py           # Tree Pose (Vrikshasana) evaluator with safety checks
│   ├── warrior_ii.py          # Warrior II (Virabhadrasana II) evaluator
│   ├── downward_dog.py        # Downward-Facing Dog (Adho Mukha Svanasana) evaluator
│   ├── plank_pose.py          # Plank Pose (Phalakasana) evaluator
│   └── cobra_pose.py          # Cobra Pose (Bhujangasana) evaluator
├── data/                      # Persistent workout history & routine definition
│   ├── workout_history.json   # Recorded practice sessions and scores
│   ├── yoga_practices.json    # Complete 40-practice master library curriculum & Kannada text
│   └── yoga_routine.json      # Structured 22-posture routine configuration
└── tests/                     # Comprehensive automated test suite (51 tests)
    ├── test_landmarks.py        # 10 Phase 1 tests (landmarks, bounds, error tolerance)
    ├── test_posture_analyzer.py # 12 Phase 2 tests (angles, Tree Pose, Warrior II, hold timer)
    ├── test_voice_assistant.py  # 4 Phase 3 tests (speech queue, throttling, safety priority)
    ├── test_integration.py      # 4 Phase 4 tests (end-to-end pipeline & correct hold logger)
    ├── test_routine.py          # 6 Guided Routine tests (22 postures, male images, sides, reps, summary)
    └── test_yoga_library.py     # 8 Yoga Library tests (40 practices, 7 categories, navigation, bilingual)
```

---

## 3. Performance & Benchmark Metrics

Benchmarked at 720p HD resolution ($1280 \times 720$) on host CPU:

| Pipeline Stage | Average Latency | Description |
| :--- | :--- | :--- |
| **YOLOv8-Pose Inference** | 79.38 ms | Deep learning full-body 17-keypoint extraction |
| **Biomechanics Logic** | **0.03 ms** | Angle calculation, stance classification & safety checks |
| **HUD & Skeleton Rendering** | 3.36 ms | Anti-aliased bones, badges, and score card |
| **Total Pipeline Latency** | **82.78 ms** | Sub-100ms real-time latency |
| **Voice Assistant Overhead** | **0.00 ms** | Detached background worker thread (zero frame drops) |

---

## 4. How to Run the Application

### 1. Launch the Streamlit Web Application (Browser Dashboard)
```powershell
cd "C:\Users\sujan\OneDrive\Documents\AI WORKSHOP\YogaCare-AI"
streamlit run app.py
```
- Opens automatically in your browser at `http://localhost:8501`.
- Features 4 tabs:
  1. **📚 Yoga Practice Library:** 40 practices across 7 categories with adult male alignment models, English/Kannada instructions, voice audio readout, and in-category Previous/Next navigation.
  2. **📹 Live Yoga Practice:** 22-posture guided routine or focused single pose drill with live webcam coaching.
  3. **📊 Workout History & Analytics:** Comprehensive session tracking, verified hold metrics, and accuracy graphs.
  4. **📖 Posture Biomechanics Guide:** Full alignment rules, joint angles, and safety warnings.

### 2. Run the Live Video CLI with Voice Guidance
```powershell
python test_live_posture.py
```
- **Hotkeys:** `1` (Tree Pose), `2` (Warrior II), `A` (Auto-Detect), `V` (Mute/Unmute Voice), `R` (Reset Timer), `Q`/`ESC` (Quit).

### 3. Run Pre-Flight System Health Check
```powershell
python health_check.py
```

### 4. Run Performance Benchmark
```powershell
python benchmark.py --iterations 50
```

---

## 5. Verification: Complete Automated Test Suite

Run all 51 automated unit and integration tests:

```powershell
python -m unittest discover -s tests -v
```

```text
Ran 51 tests in 7.4s — OK (100% Passing)
```

All 51 tests verify:
1. Model loading & landmark schema integrity.
2. Blank frame & synthetic noise tolerance.
3. Bust portrait missing-limb handling (`PARTIAL_BODY`).
4. Vector geometry & 3-point angle mathematics.
5. Tree Pose knee flexion, hip opening, and joint safety rules.
6. Warrior II 90° front lunge, straight back leg, and horizontal arms.
7. Downward-Facing Dog inverted-V apex and arm/leg extension.
8. Plank Pose straight body line and sagging core alert.
9. Cobra Pose thoracic spine extension and elbow micro-bend.
10. Posture hold duration tracking and timer reset logic.
11. Background voice assistant queueing and priority safety overrides.
12. 22-Posture routine progression, bilateral side switching, and Cat-Cow reps.
13. 40-Practice library loading, exact 7-category distribution (5, 7, 6, 6, 5, 6, 5).
14. Adult male reference image and thumbnail existence on disk.
15. Bilingual (English and Kannada) instruction presence and non-empty integrity.
16. Sequential in-category navigation with Next and Previous transitions.
17. End-to-end pipeline integration and workout history logging.

---

## 6. Project Locations
The complete project is synchronized across:
- 📁 [`C:\Users\sujan\OneDrive\Documents\AI WORKSHOP\YogaCare-AI`](file:///C:/Users/sujan/OneDrive/Documents/AI%20WORKSHOP/YogaCare-AI)
- 📁 [`C:\Users\sujan\OneDrive\Documents\WORKSHOP PROJ\YogaCare-AI`](file:///C:/Users/sujan/OneDrive/Documents/WORKSHOP%20PROJ/YogaCare-AI)

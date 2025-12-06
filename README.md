AI Workout Form Checker – Bicep Curl

This project analyzes human exercise form using MediaPipe Pose and OpenCV.
It detects key body joints, calculates elbow angle, counts repetitions, and gives real-time posture warnings.

🎯 Features
✔ Pose Landmarks

Detects 33 human body landmarks

Extracts shoulder, elbow, wrist, and hip coordinates

✔ Rep Counting

Uses elbow angle to detect “up” and “down” movement

Counts reps automatically

✔ Real-Time Warnings

Provides feedback using rule-based posture analysis:

Shoulder Swinging

Back Leaning

Arm Asymmetry

Elbow Flaring

✔ Clean Visualization

Dots only, no skeleton lines

Perfect for visual demonstrations

📂 Project Structure
recorded.py     # Main program
sampleVideo.mp4            # Test video (optional)
README.md
requirements.txt

🛠 Installation
pip install -r requirements.txt

▶️ How to Run
python recorded.py


To use your own video:

Open the file → change this line:

video_path = "videos/sampleVideo.mp4"

🔍 How It Works
1. Pose Detection

MediaPipe Pose extracts landmark coordinates every frame.

2. Angle Calculation

Elbow angle is computed using 3-point geometry:

shoulder → elbow → wrist

3. Rep Logic

Arm straight (angle > 150°) → “down”

Arm curled (angle < 40°) → “up” → ✔ rep counted

4. Form Analysis

Rule-based checks ensure proper form:

Shoulder stability

Back straight

Symmetry between arms

Elbow tracking

5. Smoothing

A rolling average reduces jitter and noise in angles.
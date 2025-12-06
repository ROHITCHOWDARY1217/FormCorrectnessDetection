import cv2
import numpy as np
import mediapipe as mp
from collections import deque

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
def smooth_angle(buffer, new_angle, size=5):
    buffer.append(new_angle)
    if len(buffer) > size:
        buffer.popleft()
    return sum(buffer) / len(buffer)

def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return angle
path = "videos/mySampleVideo.mp4"
cap = cv2.VideoCapture(0)

SHOULDER_MOVE_LIMIT = 45
BACK_LEAN_LIMIT     = 60
ASYMMETRY_LIMIT     = 55
ELBOW_FLARE_LIMIT   = 70

angle_buffer = deque()

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    rep_count = 0
    stage = "down"

    ref_shoulder_y = None
    ref_hip_y = None
    ref_elbow_x = None

    frame_counter = 0  # update reference dynamically

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (480, 720))
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(img_rgb)

        warning_text = ""

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            # keypoints
            L_shoulder = (int(lm[11].x * w), int(lm[11].y * h))
            L_elbow    = (int(lm[13].x * w), int(lm[13].y * h))
            L_wrist    = (int(lm[15].x * w), int(lm[15].y * h))

            R_shoulder = (int(lm[12].x * w), int(lm[12].y * h))
            R_elbow    = (int(lm[14].x * w), int(lm[14].y * h))
            R_wrist    = (int(lm[16].x * w), int(lm[16].y * h))

            hip = (int(lm[23].x * w), int(lm[23].y * h))

            # angles
            L_angle = calc_angle(L_shoulder, L_elbow, L_wrist)
            R_angle = calc_angle(R_shoulder, R_elbow, R_wrist)

            raw_mean_angle = (L_angle + R_angle) / 2

            # -------- smoothing --------
            mean_angle = smooth_angle(angle_buffer, raw_mean_angle)

            # --------------------------------
            # rep counting
            # --------------------------------
            if mean_angle > 150:
                stage = "down"
            if mean_angle < 40 and stage == "down":
                stage = "up"
                rep_count += 1

            # --------------------------------
            # dynamic reference update
            # every 100 frames (~3 sec)
            # --------------------------------
            frame_counter += 1
            if frame_counter % 100 == 0:
                ref_shoulder_y = L_shoulder[1]
                ref_hip_y = hip[1]
                ref_elbow_x = L_elbow[0]

            # --------------------------------
            # WARNINGS (RELAXED)
            # --------------------------------

            # Shoulder swing
            if ref_shoulder_y is None:
                ref_shoulder_y = L_shoulder[1]
            if abs(L_shoulder[1] - ref_shoulder_y) > SHOULDER_MOVE_LIMIT:
                warning_text = "Don't swing your shoulders!"

            # Back bent
            if ref_hip_y is None:
                ref_hip_y = hip[1]
            if abs(hip[1] - ref_hip_y) > BACK_LEAN_LIMIT:
                warning_text = "Keep your back straight!"

            # Asymmetry
            if abs(L_angle - R_angle) > ASYMMETRY_LIMIT:
                warning_text = "Balance both arms!"

            # Elbow flare
            if ref_elbow_x is None:
                ref_elbow_x = L_elbow[0]
            if abs(L_elbow[0] - ref_elbow_x) > ELBOW_FLARE_LIMIT:
                warning_text = "Keep elbows closer!"

            # Draw dots only
            mp_draw.draw_landmarks(
                frame,
                result.pose_landmarks,
                connections=[],  
                landmark_drawing_spec=mp_draw.DrawingSpec(
                    color=(0, 255, 0), thickness=0, circle_radius=5
                )
            )

            # UI overlays
            cv2.putText(frame, f"Reps: {rep_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            cv2.putText(frame, f"Angle: {int(mean_angle)}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if warning_text:
                cv2.putText(frame, warning_text, (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Bicep Curl - Form Checker", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

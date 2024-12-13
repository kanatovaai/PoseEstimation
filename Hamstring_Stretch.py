import cv2
import mediapipe as mp
import numpy as np
import time
import os
from pygame import mixer
import tkinter as tk
import threading

def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Endpoint
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[1])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def run_exercise(status_dict):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    reps = 0
    timer_duration = 6  
    is_timer_active = False
    timer_remaining = timer_duration
    warning_message = None 
    stop_exercise = False
    HOLD_TIME = 20
    hold_start_time = None
    current_leg = 0
    posture_correct = False

    def stop_exercise_callback():
        nonlocal stop_exercise
        stop_exercise = True

    def create_tkinter_window():
        root = tk.Tk()
        root.title("Control Panel")
        root.geometry("300x100")
        root.configure(bg="#C5EBE8")
        label = tk.Label(
            root,
            text="Click Done to terminate",
            font=("Arial", 14),
            bg="#C5EBE8",
            fg="#008878"
        )
        label.pack(pady=10)
        btn_done = tk.Button(
            root,
            text="Done",
            command=lambda: [stop_exercise_callback(), root.destroy()],
            font=("Arial", 14),
            bg="#FF6347",
            fg="white",
            width=10
        )
        btn_done.pack(pady=10)
        root.mainloop()

    threading.Thread(target=create_tkinter_window, daemon=True).start()

    mixer.init()
    success_path = os.path.join("sounds", "success.wav")
    success_sound = mixer.Sound(success_path)
    countdown_path = os.path.join("sounds", "countdown.wav")
    countdown_sound = mixer.Sound(countdown_path)
    visible_path = os.path.join("sounds", "visible.wav")
    visible_sound = mixer.Sound(visible_path)
    great_path = os.path.join("sounds", "great.wav")
    great_sound = mixer.Sound(great_path)

    last_lower_sound_time = None 
    countdown_complete = False

    def display_countdown(image, seconds_remaining):
        overlay = image.copy()
        alpha = 0.6  
        cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.putText(
            image,
            str(seconds_remaining),
            (image.shape[1] // 2 - 50, image.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            12,
            (255, 255, 255),
            16,
            cv2.LINE_AA
        )

    start_time = time.time()
    countdown_sound.play()
    while time.time() - start_time < timer_duration:
        ret, frame = cap.read()
        if not ret:
            break
        seconds_remaining = int(timer_duration - (time.time() - start_time))
        display_countdown(frame, seconds_remaining)
        cv2.imshow("Step Reaction Training", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    countdown_complete = True

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            if stop_exercise:
                status_dict["Tap_Leg"] = True
                break

            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Default no-warning each frame
            warning_message = None
            knee_angle = None
            hip_angle = None

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    # Required landmarks based on current leg
                    if current_leg == 0:  # Left leg exercise
                        required_landmarks = {
                            'Left Ankle': mp_pose.PoseLandmark.LEFT_ANKLE.value,
                            'Left Knee': mp_pose.PoseLandmark.LEFT_KNEE.value,
                            'Left Hip': mp_pose.PoseLandmark.LEFT_HIP.value,
                            'Left Shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                            'Left Foot': mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
                        }
                    else:  # Right leg exercise
                        required_landmarks = {
                            'Right Ankle': mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                            'Right Knee': mp_pose.PoseLandmark.RIGHT_KNEE.value,
                            'Right Hip': mp_pose.PoseLandmark.RIGHT_HIP.value,
                            'Right Shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                            'Right Foot': mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
                        }

                    missing_landmarks = []
                    for name, idx in required_landmarks.items():
                        visibility = landmarks[idx].visibility
                        if visibility < 0.5:
                            missing_landmarks.append(name)

                    if missing_landmarks:
                        warning_message = f"Adjust Position: {', '.join(missing_landmarks)} not detected!"
                    else:
                        # Extract coordinates
                        left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                        left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

                        right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                        right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                        right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

                        # Compute angles for the current leg
                        if current_leg == 0:  # Left leg
                            knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                            hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                        else:  # Right leg
                            knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                            hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

                        # Check posture
                        if knee_angle  and hip_angle  :
                            posture_correct = True
                            if hold_start_time is None:
                                hold_start_time = time.time()
                            elapsed = time.time() - hold_start_time
                            hold_remaining = HOLD_TIME - elapsed

                            if hold_remaining > 0:
                                # Display hold countdown
                                cv2.putText(image, f"Hold: {int(hold_remaining)}s", (10, 110), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                            else:
                                # Completed hold
                                reps += 1
                                success_sound.play()
                                warning_message = "Great job! Switch legs."
                                current_leg = 1 - current_leg
                                hold_start_time = None
                                posture_correct = False
                        else:
                            posture_correct = False
                            hold_start_time = None
                            warning_message = "Adjust your position (straighten leg & lean forward)."
                else:
                    warning_message = "Pose not detected. Make sure full body is visible."
                    current_time = time.time()
                    if last_lower_sound_time is None or (current_time - last_lower_sound_time) >= 5:
                        visible_sound.play()
                        last_lower_sound_time = current_time

            except Exception as e:
                warning_message = "Pose not detected. Make sure full body is visible."
                print("Error:", e)
                current_time = time.time()
                if last_lower_sound_time is None or (current_time - last_lower_sound_time) >= 5:
                    visible_sound.play()
                    last_lower_sound_time = current_time

            # Draw UI
            overlay = image.copy()
            feedback_box_height = 60
            cv2.rectangle(overlay, (0, 0), (640, feedback_box_height), (232, 235, 197), -1)
            counter_box_height = 60
            counter_box_width = 180
            cv2.rectangle(overlay, (0, 480 - counter_box_height), (counter_box_width, 480), (232, 235, 197), -1)
            cv2.rectangle(overlay, (640 - counter_box_width, 480 - counter_box_height), (640, 480), (232, 235, 197), -1)

            alpha = 0.5
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            # Display warnings and messages
            if warning_message:
                color = (0, 255, 0) if "Great" in warning_message else (0, 0, 255)
                cv2.putText(image, warning_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # Display timer if active
            if is_timer_active:
                cv2.putText(image, str(int(timer_remaining)), (20, 480 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Display REPS
            cv2.putText(image, 'REPS', (640 - counter_box_width + 10, 480 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(reps), (640 - counter_box_width + 8, 480 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Display angles if available
            if knee_angle is not None and hip_angle is not None:
                cv2.putText(image, f"Knee Angle: {int(knee_angle)}°", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"Hip Angle: {int(hip_angle)}°", (10, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            cv2.imshow('Step Reaction Training', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    status_dict["Tap_Leg"] = True

if __name__ == "__main__":
    status_dict = {"Tap_Leg": False}
    run_exercise(status_dict)

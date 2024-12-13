import cv2
import mediapipe as mp
import numpy as np
import time
import os
from pygame import mixer
import tkinter as tk
import threading
import random

def run_exercise(status_dict):
    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    reps = 0
    max_reps=5
    timer_duration = 6  
    is_timer_active = False
    timer_remaining = timer_duration
    warning_message = None 
    stop_exercise=False

    def stop_exercise_callback():
        nonlocal stop_exercise
        stop_exercise = True

       # Create Tkinter window for "Done" button
    def create_tkinter_window():
        root = tk.Tk()
        root.title("Control Panel")
        root.geometry("300x100")
        root.configure(bg="#C5EBE8")

        label = tk.Label(
            root,
            text="Step Reaction Training",
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

    # Start the Tkinter window in a separate thread
    threading.Thread(target=create_tkinter_window, daemon=True).start()



    mixer.init()
    success_path = os.path.join("sounds", "success.wav")
    success_sound = mixer.Sound(success_path)
    countdown_path=os.path.join("sounds", "countdown.wav")
    countdown_sound=mixer.Sound(countdown_path)
    last_lower_sound_time = None  
    visible_path=os.path.join("sounds", "visible.wav")
    visible_sound=mixer.Sound(visible_path)
    great_path=os.path.join("sounds", "great.wav")
    great_sound=mixer.Sound(great_path)


    countdown_complete = False
    def display_countdown(image, seconds_remaining):
        overlay = image.copy()
        alpha = 0.6  # Transparency factor

        # Create a semi-transparent rectangle for the countdown text
        cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Display the countdown number in the center of the screen
        cv2.putText(
            image,
            str(seconds_remaining),
            (image.shape[1] // 2 - 50, image.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            12,  # Font size
            (255, 255, 255),
            16,
            cv2.LINE_AA
        )
        

    # Perform the countdown
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

    # Set flag after countdown
    countdown_complete = True

    #Spot to foot mapping
    foot_mapping = {
        "extreme_left": "left_foot",
        "left_center": "left_foot",
        "right_center": "right_foot",
        "extreme_right": "right_foot"
    }

    dynamic_spots={}

    def calibrate_spots(landmarks):
        #hips as reference for spot positions
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        mid_x = (left_hip.x + right_hip.x) / 2
        mid_y = (left_hip.y + right_hip.y) / 2

        dynamic_spots["extreme_left"] = (left_hip.x - 0.2, mid_y + 0.2)  # Left spot
        dynamic_spots["left_center"] = (left_hip.x - 0.1, mid_y + 0.2)   # Left-center spot
        dynamic_spots["right_center"] = (right_hip.x + 0.1, mid_y + 0.2) # Right-center spot
        dynamic_spots["extreme_right"] = (right_hip.x + 0.2, mid_y + 0.2) # Right spot

    current_spot=None
    current_spot_color=(0,255,0)#green spot

    def select_next_spot():
        return random.choice(list(dynamic_spots.keys()))
    
                    
    # Setup Mediapipe Pose with specified confidence levels
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        calibrated=False

        while cap.isOpened() and reps<max_reps:
            
            if stop_exercise:  # Check if "Done" button was pressed
                status_dict["Tap_Leg"] = True
                break


            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB and make it non-writable to improve performance
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            warning_message = None  # Reset warning message for each frame
            
             # If calibration hasn't been done yet, calibrate
            if not calibrated and results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                calibrate_spots(landmarks)
                calibrated = True
                continue

             # Draw the current spot on the screen
            if not current_spot and calibrated:
                current_spot = select_next_spot()

            
            if calibrated:
                spot_x, spot_y=dynamic_spots[current_spot]
                height, width, _ = image.shape
                spot_coords = (int(spot_x * width), int(spot_y * height))

                # Ensure spot coordinates are within the screen boundaries
                if 0 <= spot_coords[0] < width and 0 <= spot_coords[1] < height:
                    cv2.circle(image, spot_coords, 30, current_spot_color, -1)
                else:
                    print(f"Spot {current_spot} is out of bounds: {spot_coords}")
                try:
                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark

                        # Check if required landmarks are detected
                        required_landmarks = {
                            'Right Foot': mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
                            'Left Foot': mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
                        }
                        missing_landmarks = []
                        for name, idx in required_landmarks.items():
                            visibility = landmarks[idx].visibility
                            if visibility < 0.5 or np.isnan(landmarks[idx].x) or np.isnan(landmarks[idx].y):
                                missing_landmarks.append(name)

                        if missing_landmarks:
                            warning_message = f"Adjust Position: {', '.join(missing_landmarks)} not detected!"

                        else:
                            # Get coordinates for hip, knee, and ankle
                            left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                            right_foot_index = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]


                            # Exercise logic with state machine
                            required_foot=foot_mapping[current_spot]
                            foot_coords=left_foot_index if required_foot=="left_foot" else right_foot_index
                            foot_x, foot_y=foot_coords
                            distance = ((foot_x - spot_x)**2 + (foot_y - spot_y)**2)**0.5

                            # If distance is within a threshold, count as a successful tap
                    if distance < 0.05:  # Adjust threshold as needed
                        reps += 1
                        current_spot = None  # Select the next spot
                        mixer.Sound("success.wav").play()  #

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

            # Overlay for feedback
            overlay = image.copy()
            feedback_box_height = 60
            cv2.rectangle(overlay, (0, 0), (640, feedback_box_height), (232, 235, 197), -1)
            counter_box_height = 60
            counter_box_width = 180
            cv2.rectangle(overlay, (0, 480 - counter_box_height), (counter_box_width, 480), (232, 235, 197), -1)
            cv2.rectangle(overlay, (640 - counter_box_width, 480 - counter_box_height), (640, 480), (232, 235, 197), -1)

            # Blend overlay with the original image to make boxes transparent
            alpha = 0.5  # Transparency factor
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            # Display warning message
            if warning_message:
                if warning_message == "Good Job! Keep Going":
                    cv2.putText(image, warning_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, warning_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # Display timer if active
            if is_timer_active:
                cv2.putText(image, str(int(timer_remaining)), (20, 480 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Render repetition counter
            cv2.putText(image, 'REPS', (640 - counter_box_width + 10, 480 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(reps), (640 - counter_box_width + 8, 480 - 10),  # Show the counter
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Draw pose landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            cv2.imshow('Step Reaction Training', image)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    status_dict["Tap_Leg"]= True

if __name__ == "__main__":
    status_dict={"Tap_Leg": False}
    run_exercise(status_dict)

import cv2
import mediapipe as mp
import numpy as np
import time
import os
from pygame import mixer

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points a, b, and c.
    """
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Endpoint

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    # Normalize the angle within [0, 180]
    if angle > 180.0:
        angle = 360 - angle

    return angle

def run_exercise():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose 

   
    # Set up camera feed
    cap = cv2.VideoCapture(0)

    # Curl counter variables
    counter = 0
    tot_count=5
    reps=0
    good_job_message_time=None
    message_duration=2
    stage=None
    mixer.init()
    beep_path=os.path.join("sounds", "beep.wav")
    beep_sound=mixer.Sound(beep_path)
    

    # Setup Mediapipe Pose with specified confidence levels
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB and make it non-writable to improve performance
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            warning_message=None

            # Extract pose landmarks
            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    #Check if required ones are detected
                    required_landmarks={
                        'Left Shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                        'Left Elbow': mp_pose.PoseLandmark.LEFT_ELBOW.value,
                        'Left Wrist': mp_pose.PoseLandmark.LEFT_WRIST.value

                    }
                    missing_landmarks=[]
                    for name,idx in required_landmarks.items():
                        visibility=landmarks[idx].visibility
                        if visibility<0.5 or np.isnan(landmarks[idx].x) or np.isnan(landmarks[idx].y):
                            missing_landmarks.append(name)
                    
                    if missing_landmarks:
                        warning_message=f"Adjust Position: {', '.join(missing_landmarks)} not detected!"

                    else:

                        # Get coordinates for shoulder, elbow, and wrist
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                        # Calculate the angle between shoulder, elbow, and wrist
                        angle = calculate_angle(shoulder, elbow, wrist)

                        # Visualize the angle at the elbow
                        cv2.putText(image, str(int(angle)),
                                    tuple(np.multiply(elbow, [640, 480]).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                        # Curl counter logic
                        if angle > 160:
                            stage = "down"
                        if angle < 30 and stage == 'down':
                            stage = "up"
                            counter += 1
                            beep_sound.play()
                            warning_message="Good Job! Keep Going"
                            good_job_message_time=time.time()
                            if counter==tot_count:
                                reps+=1
                                warning_message="Good Job! Keep Going"
                                good_job_message_time=time.time()
                                
                else:
                    warning_message="Pose not detected. Make sure full body is visible."

            except Exception as e:
                warning_message="Pose not detected. Make sure full body is visible."
                print("Error:", e)


            overlay=image.copy()
            feedback_box_height = 60
            cv2.rectangle(overlay, (0, 0), (640, feedback_box_height), (232, 235, 197), -1)
            counter_box_height = 60
            counter_box_width = 180
            cv2.rectangle(overlay, (0, 480 - counter_box_height), (counter_box_width, 480), (232, 235, 197), -1)
            cv2.rectangle(overlay, (640 - counter_box_width, 480 - counter_box_height), (640, 480), (232, 235, 197), -1)

            # Blend overlay with the original image to make boxes transparent
            alpha = 0.5  # Transparency factor
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            if good_job_message_time:
                elapsed_time = time.time() - good_job_message_time
                if elapsed_time < message_duration:
                    warning_message = "Good Job! Keep Going"  # Ensure the message persists

            if warning_message:
                if warning_message=="Good Job! Keep Going":
                    cv2.putText(image, warning_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255,0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, warning_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            # Render curl counter on image
            # Counter box (blue background) at the top-left corner
            counter_box_height = 60
            counter_box_width = 180


            # Create a blue box for the counter at the bottom-right corner
            counter_box_height = 60
            counter_box_width = 180
            #EXERCISE COUNTER
            cv2.putText(image, str(counter), (20, 480 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (20,0,0), 3, cv2.LINE_AA)
            if counter%5==0:
                counter=0
            # REPETITION COUNTER
            cv2.putText(image, 'REPS', (640 - counter_box_width + 10, 480 - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA) #text says reps
            cv2.putText(image, str(reps), (640 - counter_box_width + 8, 480 - 10),# Show the counter
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
            
            # Draw pose landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            
            cv2.imshow('Elbow Up and Down', image)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_exercise()

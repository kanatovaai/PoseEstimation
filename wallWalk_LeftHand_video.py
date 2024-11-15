import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


video_path = "poseVideos/8.mp4"  
#Also use 9.mp4

# Set up video feed
cap = cv2.VideoCapture(video_path)

# Hand raise counter variables
counter = 0
stage = None

# Setup Mediapipe Pose with specified confidence levels
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame_skip = 2  
    while cap.isOpened():
        for _ in range(frame_skip):  
            ret, frame = cap.read()
            if not ret:
                break

        # If no frames are left, exit
        if not ret:
            break

        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract pose landmarks
        try:
            landmarks = results.pose_landmarks.landmark
           
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            
            if wrist[1] < shoulder[1]:  # Wrist is higher than the shoulder (hand is up)
                stage = "up"
            elif wrist[1] > shoulder[1]:  # Wrist is lower than the shoulder (hand is down)
                if stage == "up":  
                    stage = "down"
                    counter += 1
                    print("Hand raised and lowered:", counter)

        except:
            pass

        
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw pose landmarks on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        
        cv2.imshow('Finger Wall Walk', image)

        #
        if cv2.waitKey(1) == ord('q'):  
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import numpy as np

def run_exercise():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose



    cap = cv2.VideoCapture(0)

    # Hand raise counter variables
    counter = 0
    stage = None
    reps=0

    # Setup Mediapipe Pose with specified confidence levels
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

        
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            
            try:
                landmarks = results.pose_landmarks.landmark
                
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                if left_wrist[1] < left_shoulder[1]:  # Left wrist is higher than the left shoulder (hand is up)
                    stage = "up"
                elif left_wrist[1] > left_shoulder[1]:  # Left wrist is lower than the left shoulder (hand is down)
                    if stage == "up":  # Only count when the hand comes down after being up
                        stage = "down"
                        counter += 1
                        if counter==5:
                            rep+=1
                            counter=0
                        #print("Hand raised and lowered:", counter)

            except:
                pass

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

            # Display timer if active
            if counter:
                cv2.putText(image, str(int(counter)), (20, 480 - 10),
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
            # Draw pose landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        
            cv2.imshow('Finger Wall Walk', image)

            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_exercise()

import cv2
import mediapipe as mp
import time
import os
from Common import calculate_angle, create_feedback_overlay, perform_countdown

def run_exercise():
    # Mediapipe setup
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Open video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Fullscreen setup
    cv2.namedWindow('Leg Placement Exercise', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Leg Placement Exercise', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    counter = 0  # Counter for repetitions
    sets = 0  # Sets counter
    stage = None
    feedback = "Get Ready"

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Unable to read frame from the camera.")
                break

            # Pre-process the frame
            frame = cv2.flip(frame, 1)  # Mirror the frame for user convenience
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark

                    # Get relevant landmarks for the right leg
                    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    # Calculate angle and determine feedback
                    angle = calculate_angle(right_ankle, right_knee, right_hip)
                    if 85 < angle < 95:
                        if stage != "good":
                            feedback = "Good Placement!"
                            stage = "good"
                    else:
                        feedback = "Adjust Your Position"
                        stage = "adjust"

                else:
                    feedback = "Pose Not Detected. Ensure Full Body is Visible."

            except Exception as e:
                print(f"Error processing frame: {e}")
                feedback = "Error in Pose Detection"

            # Overlay feedback and counters on the image
            image = create_feedback_overlay(image, warning_message=feedback, counter=counter, reps=sets)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )

            # Display the frame
            cv2.imshow('Leg Placement Exercise', image)

            # Break loop on 'q' key press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_exercise()
import cv2
import mediapipe as mp
import numpy as np
import time


def calculate_angle(a, b, c):
    """
    Calculate the angle between three points (a, b, c).
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def run_exercise():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(0)

    counter = 0
    stage = None
    feedback = "Get Ready"
    sets = 0
    last_feedback = feedback
    feedback_time = time.time()
    feedback_hold_duration = 5
    feedback_locked = False
    cv2.namedWindow('Leg Stretch Exercise', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Leg Stretch Exercise', 800, 600)

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
                # Get relevant landmarks
                left_foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Calculate the angle at the left knee using ankle, knee, and hip
                angle = calculate_angle(left_ankle, left_knee, left_hip)

                # Set a threshold angle (for example, if the knee is bent enough to count)
                if angle > 160:  # Change this threshold as needed
                    stage = "start"
                else:
                    if stage != "down":
                        stage = "down"
                        counter += 1
                        feedback = "Keep going!"
                        feedback_locked = True
                        feedback_time = time.time()
                        if counter == 6:
                            sets += 1
                            counter = 0

                # Ensure that the landmarks exist before calculating the line
                if landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value] and \
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value] and \
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]:
                    # Vertical line from the left foot index to the middle of the left thigh
                    # Left foot index (point 31) and middle of left thigh (average of left knee and left hip)
                    foot_x, foot_y = left_foot_index
                    knee_x, knee_y = left_knee
                    hip_x, hip_y = left_hip

                    # Middle of left thigh is the midpoint between the knee and hip
                    middle_thigh_x = (knee_x + hip_x) / 2
                    middle_thigh_y = (knee_y + hip_y) / 2

                    # Convert the normalized coordinates to pixel values
                    foot_pixel = np.multiply(left_foot_index, [640, 480])
                    middle_thigh_pixel = np.multiply([foot_x, middle_thigh_y], [640, 480])
                    knee_pixel = np.multiply(left_knee, [640, 480])

                    foot_pixel = tuple(foot_pixel.astype(int))
                    middle_thigh_pixel = tuple(middle_thigh_pixel.astype(int))
                    knee_pixel = tuple(knee_pixel.astype(int))
                    cv2.line(image, foot_pixel, middle_thigh_pixel, (0, 255, 255), 2)

                    # Check if the left knee has crossed the vertical line
                    if knee_pixel[0] > foot_pixel[0]:
                        feedback = "GOOD"
                    else:
                        feedback = "Down"

            except Exception as e:
                feedback = "Get Ready"

            current_time = time.time()
            if feedback_locked and current_time - feedback_time >= feedback_hold_duration:
                feedback_locked = False

            # Determine feedback color
            if "Get Ready" in feedback:
                feedback_color = (255, 165, 0)  # Yellow for 'Get Ready'
            elif "Good" in feedback:
                feedback_color = (0, 255, 0)
            elif "Keep" in feedback:
                feedback_color = (0, 255, 0)  # Green for positive feedback
            else:
                feedback_color = (0, 0, 255)  # Red for negative feedback

            overlay = image.copy()

            # Feedback overlay
            cv2.rectangle(overlay, (0, 0), (640, 60), (232, 235, 197), -1)
            alpha = 0.2
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            cv2.putText(image, feedback, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2, cv2.LINE_AA)

            # Bottom-left counter
            cv2.rectangle(overlay, (0, 420), (160, 480), (232, 235, 197), -1)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            cv2.putText(image, str(counter), (20, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)

            # Bottom-right sets
            cv2.rectangle(overlay, (480, 420), (640, 480), (232, 235, 197), -1)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            cv2.putText(image, 'REPS', (490, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, str(sets), (570, 470), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

            # Draw the straight vertical line from left foot index to middle of the left thigh
            if 'foot_pixel' in locals() and 'middle_thigh_pixel' in locals():
                cv2.line(image, foot_pixel, middle_thigh_pixel, (0, 255, 255), 5)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            cv2.imshow('Leg Stretch Exercise', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_exercise()
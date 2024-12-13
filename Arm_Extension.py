import cv2
import mediapipe as mp
import numpy as np
import time


def calculate_angle(a, b, c):
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
    cv2.namedWindow('Arm Extension', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Arm Extension', 800, 600)

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
                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                leftwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                #calculate angle
                angle = calculate_angle(leftwrist, shoulder, wrist)
                cv2.putText(image, str(int(angle)),
                            tuple(np.multiply(shoulder, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                if angle > 160:
                    stage = "start"
                    if not feedback_locked:
                        feedback = "Good form! Keep it up!"
                elif angle < 70 and stage == "start":
                    stage = "down"
                    counter += 1
                    feedback = "Keep going!"
                    feedback_locked = True
                    feedback_time = time.time()
                    if counter == 6:
                        sets += 1
                        counter = 0
                elif angle < 40 or angle > 177:
                    feedback = "Bad form! Adjust your arm position!"
                    feedback_locked = False
            except Exception as e:
                feedback = "Get Ready"

            current_time = time.time()
            if feedback_locked and current_time - feedback_time >= feedback_hold_duration:
                feedback_locked = False

            if "Get Ready" in feedback:
                feedback_color = (255, 165, 0)  # Yellow for 'Get Ready'
            elif "Good" in feedback or "Keep" in feedback:
                feedback_color = (0, 255, 0)  # Green for positive feedback
            else:
                feedback_color = (0, 0, 255)

            overlay = image.copy()

            #feedback
            cv2.rectangle(overlay, (0, 0), (640, 60), (232, 235, 197), -1)
            alpha = 0.2
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            cv2.putText(image, feedback, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2, cv2.LINE_AA)

            #bottom-right
            cv2.rectangle(overlay, (0, 420), (160, 480), (232, 235, 197), -1)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            cv2.putText(image, str(counter), (20, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)

            #bottom-left
            cv2.rectangle(overlay, (480, 420), (640, 480), (232, 235, 197), -1)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

            cv2.putText(image, 'REPS', (490, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, str(sets), (570, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            cv2.imshow('Arm Extension', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_exercise()
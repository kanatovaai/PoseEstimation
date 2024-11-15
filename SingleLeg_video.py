import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe pose and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


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


# Path to your video file
video_path = "poseVideos/10.mp4"  # Replace this with the path to your video

# Set up video feed
cap = cv2.VideoCapture(video_path)

# Curl counter variables
counter = 0
stage = None
frame_skip = 2  # Skipping frames to make video play faster
frame_count = 0

# Setup Mediapipe Pose with specified confidence levels
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames for faster video
        if frame_count % frame_skip != 0:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            foot = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            # Calculate the angle between shoulder, elbow, and wrist
            angle = calculate_angle(hip, knee, ankle)
            cv2.putText(image, str(int(angle)),
                        tuple(np.multiply(hip, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            
            # straight leg
            if angle > 58:  # Arm fully extended, in "down" position
                feed = "too down"
            elif angle < 45 :  # Arm in "up" position after full curl
                feed = "too up"
            else:
                feed="good"
            
            #counter
            if heel[1] < foot[1]:
                stage="up"
            elif heel[1]> foot[1]:
                if stage=="up":
                    stage="down"
                    counter+=1
                

        except:
            pass

       # Render the counter and stage text on the image
        cv2.rectangle(image, (0, 0), (225, 73), (255, 0, 0), -1)  # Blue box background (BGR: 255, 0, 0)
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'Stage', (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage,
                    (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'Feedback', (140, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, feed,
                    (130, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw pose landmarks on the image
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Display the output
        cv2.imshow('Single Leg Raise', image)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

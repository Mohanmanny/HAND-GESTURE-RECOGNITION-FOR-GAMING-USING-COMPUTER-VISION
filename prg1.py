import mediapipe as mp
import cv2
import numpy as np
import win32api
import pyautogui

# Initialize MediaPipe and OpenCV
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Video capture
video = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

def detect_gesture(hand_landmarks, image_width, image_height):
    gestures = {
        "Palm": False,
        "I": False,
        "Fist": False,
        "Thumb": False,
        "Index": False,
        "OK": False,
        "Smile": False,
        "Swing": False
    }
    
    # Normalize landmark coordinates
    landmarks = [(lm.x * image_width, lm.y * image_height) for lm in hand_landmarks.landmark]
    
    # Gesture conditions
    if all([
        landmarks[mp_hands.HandLandmark.THUMB_TIP][1] < landmarks[mp_hands.HandLandmark.WRIST][1],
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP][1] < landmarks[mp_hands.HandLandmark.WRIST][1],
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP][1] < landmarks[mp_hands.HandLandmark.WRIST][1],
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP][1] < landmarks[mp_hands.HandLandmark.WRIST][1],
        landmarks[mp_hands.HandLandmark.PINKY_TIP][1] < landmarks[mp_hands.HandLandmark.WRIST][1]
    ]):
        gestures["Palm"] = True
    
    if (
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP][1] < landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP][1] and
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP][1] > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP][1] and
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP][1] > landmarks[mp_hands.HandLandmark.RING_FINGER_TIP][1] and
        landmarks[mp_hands.HandLandmark.PINKY_TIP][1] > landmarks[mp_hands.HandLandmark.PINKY_TIP][1]
    ):
        gestures["I"] = True
    
    if all([
        landmarks[mp_hands.HandLandmark.THUMB_TIP][1] > landmarks[mp_hands.HandLandmark.THUMB_TIP][1],
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP][1] > landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP][1],
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP][1] > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP][1],
        landmarks[mp_hands.HandLandmark.RING_FINGER_TIP][1] > landmarks[mp_hands.HandLandmark.RING_FINGER_TIP][1],
        landmarks[mp_hands.HandLandmark.PINKY_TIP][1] > landmarks[mp_hands.HandLandmark.PINKY_TIP][1]
    ]):
        gestures["Fist"] = True
    
    if (
        landmarks[mp_hands.HandLandmark.THUMB_TIP][0] > landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP][0] and
        landmarks[mp_hands.HandLandmark.THUMB_TIP][1] < landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP][1]
    ):
        gestures["Thumb"] = True
    
    if (
        landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP][1] < landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP][1] and
        all(landmarks[i][1] > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP][1] for i in 
            [mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP])
    ):
        gestures["Index"] = True
    
    if (
        landmarks[mp_hands.HandLandmark.THUMB_TIP][0] < landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP][0] and
        np.linalg.norm(np.array(landmarks[mp_hands.HandLandmark.THUMB_TIP]) - np.array(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP])) < 30
    ):
        gestures["OK"] = True
    
    if (
        landmarks[mp_hands.HandLandmark.THUMB_TIP][1] < landmarks[mp_hands.HandLandmark.THUMB_TIP][1] and
        landmarks[mp_hands.HandLandmark.PINKY_TIP][1] < landmarks[mp_hands.HandLandmark.PINKY_TIP][1]
    ):
        gestures["Smile"] = True
    
    if (
        landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP][1] < landmarks[mp_hands.HandLandmark.WRIST][1] and
        abs(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP][0] - landmarks[mp_hands.HandLandmark.WRIST][0]) > 40
    ):
        gestures["Swing"] = True
    
    # Return detected gesture name
    detected_gestures = [gesture for gesture, detected in gestures.items() if detected]
    return detected_gestures[0] if detected_gestures else "Unknown"

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("Failed to capture video frame.")
            break

        # Flip the frame and process
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape

        results = hands.process(image)

        # Convert back to BGR for OpenCV visualization
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(44, 250, 250), thickness=2),
                )

                # Detect gesture
                gesture = detect_gesture(hand_landmarks, image_width, image_height)
                
                # Display the gesture name
                cv2.putText(image, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Get the tip of the index finger
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # Map to screen coordinates
                x_px = int(index_finger_tip.x * image_width)
                y_px = int(index_finger_tip.y * image_height)

                cv2.circle(image, (x_px, y_px), 10, (0, 255, 0), -1)

                screen_x = int(index_finger_tip.x * screen_width)
                screen_y = int(index_finger_tip.y * screen_height)

                # Set cursor position
                win32api.SetCursorPos((screen_x, screen_y))

                # Example click detection: check if index finger is close to the thumb
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                distance = np.linalg.norm(
                    np.array([index_finger_tip.x - thumb_tip.x, index_finger_tip.y - thumb_tip.y])
                )

                if distance < 0.05:  # Adjust threshold as needed
                    pyautogui.click()

        # Display the result
        cv2.imshow('Hand Tracking', image)

        # Exit on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
video.release()
cv2.destroyAllWindows()

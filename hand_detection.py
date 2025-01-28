import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize OpenCV for webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # Convert the frame to RGB (MediaPipe needs RGB images)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    result = hands.process(frame_rgb)

    # Draw landmarks on the hands
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the output
    cv2.imshow("Hand Tracking", frame)

    # Close window on pressing ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

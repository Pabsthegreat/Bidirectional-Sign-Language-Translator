import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load the trained model
model = tf.keras.models.load_model("sign_language_model.h5")

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Define class labels (make sure they match your training labels)
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame to avoid mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame to detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x, x_min)
                y_min = min(y, y_min)
                x_max = max(x, x_max)
                y_max = max(y, y_max)

            # Expand bounding box
            padding = 20
            x_min = max(0, x_min - padding)
            y_min = max(0, y_min - padding)
            x_max = min(w, x_max + padding)
            y_max = min(h, y_max + padding)

            # Extract and preprocess the hand region
            hand_img = frame[y_min:y_max, x_min:x_max]
            hand_img = cv2.resize(hand_img, (224, 224))
            hand_img = hand_img / 255.0  # Normalize
            hand_img = np.expand_dims(hand_img, axis=0)

            # Predict sign
            prediction = model.predict(hand_img)
            class_id = np.argmax(prediction)
            sign = labels[class_id]

            # Display the predicted sign
            cv2.putText(frame, f"Predicted: {sign}", (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Hand Tracking - Sign Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model = load_model("models/gesture_recognition_model.h5")
labels = ['A', 'B', 'C', ...]  # Replace with your dataset labels

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect hand using your hand_detection module (placeholder function here)
    hand_roi = detect_hand(frame)  # Replace with your actual function

    if hand_roi is not None:
        # Preprocess ROI
        hand_roi = cv2.resize(hand_roi, (64, 64))
        hand_roi = hand_roi / 255.0
        hand_roi = np.expand_dims(hand_roi, axis=0)

        # Predict gesture
        prediction = model.predict(hand_roi)
        class_index = np.argmax(prediction)
        gesture = labels[class_index]

        # Display result
        cv2.putText(frame, f"Gesture: {gesture}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Sign Language Translator", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder and load the fitted encoder
label_encoder = LabelEncoder()
label_encoder.fit(np.load('labels.npy'))  # Fit it based on the training labels

# Load your trained model
model = tf.keras.models.load_model('sign_language_model_improved.h5')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe hand detector (if not already done)
import mediapipe as mp
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to RGB (MediaPipe requires RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and extract hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks and connections
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get bounding box coordinates
            x_min = min([landmark.x for landmark in hand_landmarks.landmark])
            y_min = min([landmark.y for landmark in hand_landmarks.landmark])
            x_max = max([landmark.x for landmark in hand_landmarks.landmark])
            y_max = max([landmark.y for landmark in hand_landmarks.landmark])
            
            # Convert normalized coordinates to pixel values
            height, width, _ = frame.shape
            x_min = int(x_min * width)
            y_min = int(y_min * height)
            x_max = int(x_max * width)
            y_max = int(y_max * height)
            
            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Normalize hand landmarks and prepare the data for prediction
            hand_points = []
            for landmark in hand_landmarks.landmark:
                hand_points.append(landmark.x / width)  # Normalize x
                hand_points.append(landmark.y / height)  # Normalize y
                hand_points.append(landmark.z)  # z remains unnormalized
            
            # Prepare the hand data for the model
            hand_points = np.array(hand_points).reshape(1, -1)  # Reshape for single sample prediction

            # Make prediction
            prediction = model.predict(hand_points)
            predicted_class = np.argmax(prediction, axis=1)

            # Decode the predicted label
            predicted_sign = label_encoder.inverse_transform(predicted_class)

            # Display the predicted sign
            cv2.putText(frame, f"Predicted Sign: {predicted_sign[0]}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('Sign Language Translator', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import os
import mediapipe as mp

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Set up directories
dataset_dir = 'sign_language_dataset/Indian'  # Path to your image dataset
output_dir = 'sign_language_dataset'  # Path to save extracted hand points

# Prepare to store hand points and labels
hand_points = []
labels = []

# Loop through all images in the dataset directory
for folder_name in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder_name)
    
    # Ensure it is a directory (for different sign classes)
    if os.path.isdir(folder_path):
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            print(image_path)
            # Load the image
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            # Convert the image to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image for hand landmarks
            results = hands.process(image_rgb)

            # Extract hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Normalize hand landmarks to fit between 0 and 1
                    hand_points_single = []
                    for lm in hand_landmarks.landmark:
                        hand_points_single.append(lm.x / width)  # Normalize x
                        hand_points_single.append(lm.y / height)  # Normalize y
                        hand_points_single.append(lm.z)  # z remains unnormalized
                    hand_points.append(hand_points_single)

                    # Store the label (folder name as label)
                    labels.append(folder_name)

# Convert to numpy arrays
hand_points = np.array(hand_points)
labels = np.array(labels)

# Save extracted data to .npy files for future use
np.save(os.path.join(output_dir, 'hand_points.npy'), hand_points)
np.save(os.path.join(output_dir, 'labels.npy'), labels)

print("Hand points extraction complete!")

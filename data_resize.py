import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

# Define the dataset directory and target size
dataset_dir = 'sign_language_dataset/Indian'  # Update with the actual path to your dataset
target_size = (224, 224)  # Resize images to 224x224
train_dir = 'sign_language_dataset/train/Indian'  # Define your training directory path
valid_dir = 'sign_language_dataset/val/Indian'  # Define your validation directory path

# Create the directories for training and validation
os.makedirs(train_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Iterate over the classes and images
class_names = os.listdir(dataset_dir)
for class_name in class_names:
    # Create class directories in train and valid directories
    train_class_dir = os.path.join(train_dir, class_name)
    valid_class_dir = os.path.join(valid_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(valid_class_dir, exist_ok=True)

    class_images = os.listdir(os.path.join(dataset_dir, class_name))
    
    # Split images into train and validation sets (80-20 split)
    train_images, valid_images = train_test_split(class_images, test_size=0.2, random_state=42)
    
    # Copy images to the respective directories
    for image_name in train_images:
        img_path = os.path.join(dataset_dir, class_name, image_name)
        img = Image.open(img_path)
        img = img.resize(target_size)  # Resize image
        img.save(os.path.join(train_class_dir, image_name))

    for image_name in valid_images:
        img_path = os.path.join(dataset_dir, class_name, image_name)
        img = Image.open(img_path)
        img = img.resize(target_size)  # Resize image
        img.save(os.path.join(valid_class_dir, image_name))

print("Dataset has been resized and split into train and validation sets.")

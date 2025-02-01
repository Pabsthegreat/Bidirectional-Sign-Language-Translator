from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import os

# Set directories for training and validation
train_dir = 'sign_language_dataset/train/Indian'  # update with your path
valid_dir = 'sign_language_dataset/Indian'  # update with your path

# 1. Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,                # Normalize pixel values to [0, 1]
    rotation_range=30,             # Random rotation
    width_shift_range=0.2,         # Random horizontal shift
    height_shift_range=0.2,        # Random vertical shift
    shear_range=0.2,               # Random shear
    zoom_range=0.2,                # Random zoom
    horizontal_flip=True,          # Random horizontal flip
    fill_mode='nearest'            # Fill any empty pixels
)

valid_datagen = ImageDataGenerator(rescale=1./255)  # Only rescale for validation

# 2. Load the dataset using ImageDataGenerator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),         # Resize to 224x224
    batch_size=32,
    class_mode='categorical',       # Categorical labels for multi-class classification
    shuffle=True
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(224, 224),         # Resize to 224x224
    batch_size=32,
    class_mode='categorical',       # Categorical labels
    shuffle=False
)


# 3. Build the Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model
base_model.trainable = False

# Create the custom classifier
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(35, activation='softmax')  # 35 classes for ISL
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train the Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,                     # You can increase epochs based on your resources
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // valid_generator.batch_size
)

# Save the trained model
model.save('sign_language_model.h5')

print("Training complete! Model saved.")

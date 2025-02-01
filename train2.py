import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# Load Data
X = np.load("hand_points.npy")
y = np.load("labels.npy")

# One-Hot Encode Labels
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))
num_classes = y_onehot.shape[1]

# Compute Class Weights
y_labels = np.argmax(y_onehot, axis=1)  # Convert one-hot to integer labels
class_weights = compute_class_weight('balanced', classes=np.unique(y_labels), y=y_labels)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Split Data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.2, stratify=y_labels, random_state=42)

# Define Improved Model
model = Sequential([
    Dense(256, kernel_regularizer=l2(0.01), input_shape=(X.shape[1],)),  
    BatchNormalization(),
    LeakyReLU(),
    Dropout(0.4),

    Dense(128, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    LeakyReLU(),
    Dropout(0.4),

    Dense(64, kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    LeakyReLU(),
    Dropout(0.4),

    Dense(num_classes, activation='softmax')
])

# Compile Model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005),  # Increased LR
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

# Train Model
history = model.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val),
    epochs=50, 
    batch_size=32,  
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr]
)

# Save Model & Encoder
model.save("sign_language_model_improved.h5")
np.save("onehot_encoder.npy", encoder.categories_)

# Model Summary
model.summary()

# Plot Accuracy & Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')

plt.show()

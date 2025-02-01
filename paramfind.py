import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split

# Load data
X = np.load("hand_points.npy")
y = np.load("labels.npy")

# One-Hot Encode Labels
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))
num_classes = y_onehot.shape[1]

# Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Train Model with different epochs
# Train Model with different architectures
# Train Model with different dropout rates
# Train Model with different optimizers
def train_with_optimizer(optimizer_name):
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    if optimizer_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    elif optimizer_name == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    elif optimizer_name == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit model with the selected optimizer
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Try different optimizers
optimizers = ['adam', 'sgd', 'rmsprop']
for optimizer_name in optimizers:
    print(f"Training with optimizer: {optimizer_name}")
    train_with_optimizer(optimizer_name)

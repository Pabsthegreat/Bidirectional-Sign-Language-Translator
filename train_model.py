from tensorflow.keras import layers, models

# Example model creation
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(10, activation='softmax')
])

model.summary()

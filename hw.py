import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

# Load mnist data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Build LeNet-5 model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(6, kernel_size=(5, 5), activation="sigmoid", padding="same", input_shape=(28, 28, 1)),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(16, kernel_size=(5, 5), activation="sigmoid"),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation="sigmoid"),
    tf.keras.layers.Dense(84, activation="sigmoid"),
    tf.keras.layers.Dense(10, activation="sigmoid")
])

# Compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=10)

# Evaluate model
model.evaluate(x_test, y_test)

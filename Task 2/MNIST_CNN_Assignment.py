# Deep Learning with TensorFlow: MNIST Handwritten Digits Classification
# ---------------------------------------------------------------
# This script builds, trains, and evaluates a CNN model to classify handwritten digits from the MNIST dataset.
# Goal:
# 1. Build a CNN model to classify handwritten digits.
# 2. Achieve >95% test accuracy.
# 3. Visualize the model’s predictions on 5 sample images.

# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load and preprocess the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values to range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Reshape images to include channel dimension (for CNN input)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Hypothetical Bug: Incorrect reshape causing dimension mismatch
# Original: x_train = x_train.reshape(-1, 28, 28, 1)  # Correct
# Buggy: x_train = x_train.reshape(-1, 28, 28)  # Missing channel dimension, will cause error in Conv2D
# Fix: Ensure channel dimension is included

# Step 2: Define the CNN model architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

# Step 3: Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Fixed: Use sparse_categorical_crossentropy for integer labels
              metrics=['accuracy'])

# Display model summary
print("\nModel Summary:")
model.summary()

# Step 4: Train the model
print("\nTraining the CNN model...")
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Step 5: Evaluate model performance on test data
print("\nEvaluating model on test data...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

# Step 6: Visualize predictions on 5 sample test images
print("\nVisualizing sample predictions...")
predictions = model.predict(x_test[:5])

plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title(f"Pred: {np.argmax(predictions[i])}\nTrue: {y_test[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Step 7 (Optional): Plot accuracy and loss curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Step 8: Save the trained model
print("\nSaving the trained model...")
model.save('mnist_cnn_model.h5')
print("Model saved as 'mnist_cnn_model.h5'")

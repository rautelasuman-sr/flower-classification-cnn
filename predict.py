
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("flower_model.keras")

# Correct TF Flowers label order
class_names = [
    "dandelions",
    "daisy",
    "tulips",
    "sunflowers",
    "roses"
]

# Image to test
image_path = "test_images/daisy.jpg"

# Load image
image = Image.open(image_path).convert("RGB")
image = image.resize((180, 180))

# Convert to array
image_array = np.array(image, dtype=np.float32) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Prediction
prediction = model.predict(image_array, verbose=0)[0]

# Show ALL probabilities
print("\nPrediction probabilities:")

for name, probability in zip(class_names, prediction):
    print(f"{name}: {probability * 100:.2f}%")

# Final prediction
predicted_index = np.argmax(prediction)

print("\n-----------------------------")
print("Final Prediction:", class_names[predicted_index])
print("Confidence:", round(prediction[predicted_index] * 100, 2), "%")
print("-----------------------------")


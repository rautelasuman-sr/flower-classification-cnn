import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# -----------------------------
# 1. Load Dataset
# -----------------------------
(dataset_train, dataset_test), info = tfds.load(
    "tf_flowers",
    split=["train[:80%]", "train[80%:]"],
    as_supervised=True,
    with_info=True,
    data_dir="./dataset"
)

class_names = info.features["label"].names

print("Classes:", class_names)
print("Number of classes:", len(class_names))

# -----------------------------
# 2. Image Settings
# -----------------------------
IMG_SIZE = 180
BATCH_SIZE = 32

def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

dataset_train = dataset_train.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset_test = dataset_test.map(
    preprocess,
    num_parallel_calls=tf.data.AUTOTUNE
)

dataset_train = dataset_train.shuffle(1000).batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

dataset_test = dataset_test.batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

# -----------------------------
# 3. Create CNN Model
# -----------------------------
model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(len(class_names), activation="softmax")
])

# -----------------------------
# 4. Compile Model
# -----------------------------
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# 5. Show Model
# -----------------------------
model.summary()

# -----------------------------
# 6. Train Model
# -----------------------------
history = model.fit(
    dataset_train,
    validation_data=dataset_test,
    epochs=10
)

# -----------------------------
# 7. Save Model
# -----------------------------
model.save("flower_model.keras")

print("\nModel training completed successfully!")
print("Model saved as flower_model.keras")

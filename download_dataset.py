import tensorflow_datasets as tfds

dataset, info = tfds.load(
    "tf_flowers",
    split="train",
    as_supervised=True,
    with_info=True,
    data_dir="./dataset"
)

print("Dataset downloaded successfully!")
print("Number of images:", info.splits["train"].num_examples)
print("Classes:", info.features["label"].names)

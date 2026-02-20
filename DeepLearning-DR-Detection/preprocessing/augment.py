import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config.config import TRAIN_DIR

# Augmentation configuration
datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)


def augment_class(class_path, num_augmented=2):
    """
    Augment images inside a class folder
    num_augmented = number of augmented copies per image
    """

    for img_name in os.listdir(class_path):

        img_path = os.path.join(class_path, img_name)

        if not img_name.lower().endswith(("png", "jpg", "jpeg")):
            continue

        img = datagen.flow_from_directory(
            directory=os.path.dirname(class_path),
            target_size=(299, 299),
            batch_size=1,
            class_mode=None,
            save_to_dir=class_path,
            save_prefix="aug",
            save_format="jpg"
        )

        break  # Prevent infinite loop


def augment_dataset():
    print("Starting Data Augmentation...")

    for class_name in os.listdir(TRAIN_DIR):
        class_path = os.path.join(TRAIN_DIR, class_name)

        if os.path.isdir(class_path):
            print(f"Augmenting Class: {class_name}")
            augment_class(class_path)

    print("Augmentation Completed Successfully.")


if __name__ == "__main__":
    augment_dataset()

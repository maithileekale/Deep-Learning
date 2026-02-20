import os
import cv2
import numpy as np
from tqdm import tqdm
from config.config import TRAIN_DIR, TEST_DIR

# Image size (must match model)
IMAGE_SIZE = (299, 299)


def preprocess_image(img_path):
    """
    Reads image, resizes to 299x299,
    applies normalization and CLAHE contrast enhancement.
    """
    img = cv2.imread(img_path)

    if img is None:
        return None

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, IMAGE_SIZE)

    # Apply CLAHE (contrast enhancement)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    return img


def process_dataset(directory):
    """
    Preprocess all images inside training or testing folder
    """
    print(f"Processing folder: {directory}")

    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)

        if not os.path.isdir(class_path):
            continue

        for img_name in tqdm(os.listdir(class_path)):
            img_path = os.path.join(class_path, img_name)

            processed = preprocess_image(img_path)

            if processed is not None:
                cv2.imwrite(img_path, cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    print("Starting Preprocessing...")

    process_dataset(TRAIN_DIR)
    process_dataset(TEST_DIR)

    print("Preprocessing Completed Successfully.")

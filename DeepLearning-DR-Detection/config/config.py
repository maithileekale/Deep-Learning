import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Image configuration (InceptionV3 requirement)
IMAGE_SIZE = (299, 299)
BATCH_SIZE = 16
EPOCHS = 25
NUM_CLASSES = 5

# Data directories
TRAIN_DIR = os.path.join(BASE_DIR, "data", "training")
TEST_DIR = os.path.join(BASE_DIR, "data", "testing")

# Model path
MODEL_PATH = os.path.join(BASE_DIR, "model", "saved_model.h5")

# Evaluation folder
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")

import os
import sys

# Add project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from config.config import *

os.makedirs(EVAL_DIR, exist_ok=True)

model = load_model(MODEL_PATH)

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

predictions = model.predict(test_generator)

y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

accuracy = accuracy_score(y_true, y_pred)

# Save Accuracy to file
with open(os.path.join(EVAL_DIR, "accuracy.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"))
plt.close()

# Classification Report
report = classification_report(y_true, y_pred)

with open(os.path.join(EVAL_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

print("Evaluation Complete.")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

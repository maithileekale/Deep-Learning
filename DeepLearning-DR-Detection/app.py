import os
import sys
import uuid
import json
import numpy as np
import requests
from datetime import datetime

from flask import Flask, render_template, request, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---------------------------------
# Fix Python Path (Important)
# ---------------------------------
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from config.config import MODEL_PATH

# ---------------------------------
# Flask Initialization
# ---------------------------------
app = Flask(__name__)
app.secret_key = "dr_detection_secret"

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

if not os.path.isdir(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ---------------------------------
# Load Trained Model
# ---------------------------------
model = load_model(MODEL_PATH)
print("✅ DR Detection Model Loaded Successfully")

# -------------------------------
# Hugging Face API Configuration
# -------------------------------
HF_API_TOKEN = "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"

HF_MODEL_URL = "https://api-inference.huggingface.co/models/ibm-granite/granite-3.1-3b-a800m-instruct"

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}


# -------------------------------
# Helper Functions
# -------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


class_labels = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}


def predict_dr(img_path):
    img = preprocess_image(img_path)
    preds = model.predict(img)
    print("Raw Predictions:", preds)
    predicted_class = np.argmax(preds)
    confidence = round(np.max(preds) * 100, 2)
    return class_labels[predicted_class], confidence


# -------------------------------
# Hugging Face API Explanation
# -------------------------------
def generate_explanation(prediction, confidence):
    """
    Generates explanation using HuggingFace API.
    If API fails, returns predefined clinical explanation.
    """

    # 🔹 Predefined fallback explanations
    fallback_explanations = {
        "No DR": f"""
The retinal scan shows no visible signs of diabetic retinopathy.

This indicates that there is currently no detectable damage to the blood vessels in the retina.
However, regular eye check-ups are still strongly recommended.

Precautions:
• Maintain controlled blood sugar levels
• Monitor blood pressure
• Follow a balanced diet
• Annual eye screening
""",

        "Mild DR": f"""
Mild Diabetic Retinopathy detected with minimal microaneurysms.

This is the earliest stage of diabetic eye disease and may not affect vision significantly.

Precautions:
• Strict blood sugar control
• Regular ophthalmologist visits
• Monitor for vision changes
• Maintain healthy lifestyle habits
""",

        "Moderate DR": f"""
Moderate Diabetic Retinopathy detected.

At this stage, some retinal blood vessels may become blocked, affecting oxygen supply to the retina.

Precautions:
• Immediate consultation with an eye specialist
• Tight glucose control
• Possible need for medical monitoring
• Avoid smoking and manage cholesterol
""",

        "Severe DR": f"""
Severe Diabetic Retinopathy detected.

Multiple blood vessels are blocked, increasing risk of retinal damage and vision loss.

Precautions:
• Urgent ophthalmology evaluation
• Possible laser treatment consideration
• Strict diabetes management
• Regular follow-up examinations
""",

        "Proliferative DR": f"""
Proliferative Diabetic Retinopathy detected.

This is an advanced stage where new abnormal blood vessels form in the retina, which can lead to serious vision complications.

Precautions:
• Immediate specialist treatment required
• Possible laser or surgical intervention
• Strict diabetes and blood pressure management
• Continuous retinal monitoring
"""
    }

    # 🔹 Try HuggingFace API (if enabled)
    try:
        prompt = f"""
A diabetic retinopathy detection model predicted:
Class: {prediction}
Confidence: {confidence}%.

Provide a simple medical explanation and precautions.
"""

        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}"
        }

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 150
            }
        }

        response = requests.post(HF_API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            return result[0]["generated_text"]

    except Exception as e:
        print("HuggingFace API failed:", e)

    # 🔹 If API fails → use fallback explanation
    return fallback_explanations.get(prediction, "Medical explanation unavailable.")


# -------------------------------
# Local JSON Database
# -------------------------------
def save_to_database(data):
    try:
        with open("database.json", "r") as file:
            records = json.load(file)
    except:
        records = []

    records.append(data)

    with open("database.json", "w") as file:
        json.dump(records, file, indent=4)

# -------------------------------
# User Management (Secure JSON Storage)
# -------------------------------
def load_users():
    try:
        with open("users.json", "r") as file:
            return json.load(file)
    except:
        return []

def save_users(users):
    with open("users.json", "w") as file:
        json.dump(users, file, indent=4)

def get_user(username):
    users = load_users()
    for user in users:
        if user["username"] == username:
            return user
    return None

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return redirect("/dashboard")


@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        flash("Please login to access dashboard.", "error")
        return redirect("/login")

    return render_template("dashboard.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "" or not allowed_file(file.filename):
        return "Invalid file type"

    filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    prediction, confidence = predict_dr(path)

    explanation = generate_explanation(prediction, confidence)

    record = {
        "user": session.get("user", "anonymous"),
        "image": filename,
        "prediction": prediction,
        "confidence": confidence,
        "explanation": explanation,
        "timestamp": str(datetime.now())
    }

    save_to_database(record)

    return render_template(
        "prediction.html",
        prediction=prediction,
        confidence=confidence,
        explanation=explanation,
        image_path=path
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if get_user(username):
            flash("Username already exists. Please choose another.", "error")
            return redirect("/register")

        hashed_password = generate_password_hash(password)

        users = load_users()
        users.append({
            "username": username,
            "password": hashed_password
        })
        save_users(users)

        flash("Registration successful! Please login.", "success")
        return redirect("/login")

    return render_template("register.html")



@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = get_user(username)

        if not user:
            flash("User not registered. Please register first.", "error")
            return redirect("/login")

        if not check_password_hash(user["password"], password):
            flash("Incorrect password.", "error")
            return redirect("/login")

        session["user"] = username
        flash("Login successful!", "success")
        return redirect("/dashboard")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect("/login")


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)

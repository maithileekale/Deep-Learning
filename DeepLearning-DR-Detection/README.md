# Diabetic Retinopathy Detection using Deep Learning

## Project Overview

This project is a Deep Learning based web application for detecting **Diabetic Retinopathy (DR)** from retinal images.

The system uses a trained Convolutional Neural Network (Xception-based architecture) InceptionV3 to classify retinal fundus images into 5 categories:

- 0 → No DR
- 1 → Mild DR
- 2 → Moderate DR
- 3 → Severe DR
- 4 → Proliferative DR

The application includes:

- User Registration & Login System
- Image Upload Feature
- Real-time Prediction
- Confidence Score Display
- Prediction History Storage
- Web-based Dashboard (Flask)



## Project Architecture
DeepLearning-DR-Detection/
│
├── app.py
├── requirements.txt
├── Procfile
├── README.md
│
├── config/
├── data/
├── model/
├── preprocessing/
├── evaluation/
├── static/
├── templates/
├── docs/




## Model Details

- Architecture: Transfer Learning (Xception)
- Input Size: 299x299x3
- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Final Test Accuracy: ~80%
- Dataset: 5-class Diabetic Retinopathy dataset



## Installation Guide

### Step 1: Clone Repository

git clone <your-repo-link>
cd DeepLearning-DR-Detection

### Step 2: Create Virtual Environment

conda create -n dr_detection python=3.8
conda activate dr_detection

### Step 3: Install Dependencies

pip install -r requirements.txt

### Step 4: Run Application

python app.py

Open browser:

http://127.0.0.1:5000




## Web Application Flow

1. Register User
2. Login
3. Upload Retinal Image
4. Model Predicts DR Class
5. Result + Confidence Displayed
6. Logout



## Evaluation

- Accuracy Plot
- Loss Plot
- Confusion Matrix
- Classification Report

All evaluation outputs are stored in the `evaluation/` folder.



## Deployment

The project can be deployed using:

- Render
- Railway
- Heroku

Ensure:

- `Procfile` is present
- `requirements.txt` is complete



## Future Improvements

- Add Grad-CAM Visualization
- Improve Class Imbalance using Class Weights
- Deploy with GPU backend
- Add Prediction History Dashboard



## Developed By

Maithilee Kale  
B.Tech – AI & ML Engineer  

# ProdigyInfotech_Machine-Learning
# Machine Learning Mini Projects

#Machine Learning & AI Projects
This repository contains 4 machine learning and computer vision projects, demonstrating regression, clustering, classification, and real‑time gesture recognition.

Task‑01: House Price Prediction using Linear Regression
Objective
Predict house prices based on square footage, bedrooms, and bathrooms using a Linear Regression model.

Key Features
Supervised regression model.

Input features: Square footage, bedrooms, bathrooms.

Evaluates with Mean Squared Error (MSE) and R² Score.

Workflow
Load dataset of house features and prices.

Split into training and testing sets.

Train a Linear Regression model.

Evaluate and predict house prices.

Use Cases
Real estate price estimation

Assisting buyers and sellers in property valuation

Task‑02: Customer Segmentation using K‑Means Clustering
Objective
Segment retail store customers based on their purchase history using K‑Means clustering, enabling targeted marketing.

Key Features
Unsupervised learning (Clustering).

Groups customers with similar purchase patterns.

Uses the Elbow Method to select the optimal number of clusters.

Workflow
Collect customer purchase/frequency data.

Normalize data for clustering.

Apply K‑Means to group customers.

Visualize clusters in 2D/3D.

Use Cases
Market segmentation for targeted campaigns

Personalized offers and promotions

Customer behavior analytics

Task‑03: Image Classification using SVM (Cats vs Dogs)
Objective
Classify images of cats and dogs using a Support Vector Machine (SVM) model from the Kaggle dataset.

Key Features
Supervised classification model using SVM.

Images are resized and converted to numerical features.

Evaluates performance with Accuracy Score and Confusion Matrix.

Workflow
Load Cats vs Dogs image dataset.

Preprocess images (resize, normalize, flatten).

Train an SVM classifier.

Evaluate accuracy and predict new images.

Use Cases
Basic image classification

Animal detection in computer vision

Demonstrates SVM for small datasets

Task‑04: Hand Gesture Recognition
Objective
Recognize and classify hand gestures from images or video streams, enabling gesture‑based control systems.

Key Features
Uses CNN or MediaPipe landmarks for gesture detection.

Supports real‑time webcam gesture recognition.

Detects gestures like Thumbs Up, Peace, Fist, Stop, etc.

Workflow
Collect or load a hand gesture dataset.

Preprocess images or extract 21 key hand landmarks.

Train a CNN or ML classifier for gesture recognition.

Predict gestures in real‑time using webcam or test images.

Use Cases
Touchless AR/VR or gaming interfaces

Sign language recognition

Smart home/device control with gestures

How to Run
Clone the repository.

Install required dependencies from requirements.txt.

Run individual Python files:

task01_linear_regression.py

task02_kmeans_clustering.py

task03_svm_classification.py

task04_hand_gesture_recognition.py


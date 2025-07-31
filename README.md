# ProdigyInfotech_Machine-Learning
# Machine Learning Mini Projects

This repository contains two mini projects implemented using Python and Scikit-Learn:

1. House Price Prediction (Linear Regression)
2. Customer Segmentation (K-Means Clustering)

1. House Price Prediction - Linear Regression

Objective:
Predict the price of a house based on:
- Square Footage
- Number of Bedrooms
- Number of Bathrooms

Technologies Used:
- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib

Steps Implemented:
1. Created a sample dataset of houses with square footage, bedrooms, bathrooms, and price.
2. Split the data into **training** and **testing** sets.
3. Trained a **Linear Regression model** using `scikit-learn`.
4. Predicted house prices for test data.
5. Evaluated the model using **Mean Squared Error (MSE)** and **R² Score**.
6. Visualized **Actual vs Predicted Prices** with a scatter plot.


2. Customer Segmentation - K-Means Clustering

Objective:
Group customers of a retail store into segments based on:
- Annual Income
- Spending Score

This helps businesses understand customer behavior and target specific groups.

Technologies Used:
- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn

Steps Implemented
1. Created a sample dataset of customers with annual income and spending score.
2. Standardized the data using `StandardScaler` for better clustering.
3. Determined the optimal number of clusters using the **Elbow Method**.
4. Applied **K-Means Clustering** to segment customers.
5. Visualized the clusters and centroids with a scatter plot.
6. Labeled the clusters to identify customer groups (High, Medium, Low value).

📊 Sample Outputs
House Price Prediction
- R² Score shows how well the model predicts.
- Scatter plot comparing **Actual vs Predicted Prices**.

Customer Segmentation
- Visualizes customers grouped into clusters.
- Each color represents a unique segment.

💡 Future Improvements
- Use a **real-world dataset** for more accurate predictions.
- Deploy the models using **Flask / Streamlit** for a web interface.
- Automatically label clusters for better business insights.

## 📂 Project Structure
1. Project Overview
Goal: Recognize and classify hand gestures for gesture-based control.

Approach:

Collect or use a hand gesture dataset (images or video frames).
Preprocess images (grayscale, resize, normalize).
Train a Convolutional Neural Network (CNN).
Test on images or real-time webcam feed.



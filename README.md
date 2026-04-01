# Fraud Detection System
# Overview
This project is an end-to-end fraud detection system using Machine Learning.

# Tech Stack
Python
- Scikit-learn
- FastAPI
- Pandas, NumPy

#Features
- Data preprocessing & feature engineering
- Handling imbalanced data using SMOTE
- Logistic Regression model
- Model deployment using FastAPI

-#  How to Run
pip install -r requirements.txt
python main.py

#Docker Building and running statements
docker build -t fraud_detection_app .
docker run -d -p 8000:8000 fraud_detection_app

#Acessing API
http://localhost:8000(Home page)
http://localhost:8000/docs(predict the fraud_detection)

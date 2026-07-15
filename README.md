 Fake News Detection using Machine Learning and Deep Learning

📌 Project Overview

This project detects whether a news article is **Fake** or **True** using Machine Learning and Deep Learning techniques.

The system performs text preprocessing, feature extraction, model training, evaluation, and deployment through a Streamlit web application.


🚀 Features

- Text Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- TF-IDF Feature Extraction
- Machine Learning Models
  - Logistic Regression
  - Naive Bayes
  - Random Forest
- Deep Learning Models
  - Artificial Neural Network (ANN)
  - Long Short-Term Memory (LSTM)
- Model Evaluation
- Streamlit Web Application
- Real-time News Prediction


 📊 Dataset

Dataset Source:

- Fake.csv
- True.csv

The datasets contain news articles labeled as Fake and True.


 🧹 Text Preprocessing

- Lowercase Conversion
- HTML Tag Removal
- URL Removal
- Punctuation Removal
- Stopword Removal
- Lemmatization


 🤖 Machine Learning Models

- Logistic Regression
- Naive Bayes
- Random Forest


 🧠 Deep Learning Models

- Artificial Neural Network (ANN)
- Long Short-Term Memory (LSTM)

 📈 Results

| Model         | Test Accuracy |
| Random Forest | 99%           |
| ANN           | 99.88%        | 
| LSTM          | 99.81%        |



💻 Streamlit Application

The project includes a Streamlit web application where users can:

- Paste a news article
- Predict whether it is Fake or True
- View prediction confidence
by running the application:
streamlit run app/app.py


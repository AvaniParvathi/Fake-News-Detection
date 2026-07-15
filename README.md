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



📂 Project Structure

fake_news_detection/
│
├── app/
│   ├── app.py
│   └── predictor.py
│
├── data/
│
├── models/
│   ├── best_ann_model.keras
│   ├── best_lstm_model.keras
│   ├── random_forest_model.pkl
│   ├── tokenizer.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   ├── 1.eda.ipynb
│   ├── 2.ML models.ipynb
│   └── 3.DL models.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── text_cleaner.py
│   ├── feature_engineering.py
│   └── ...
│
├── requirements.txt
├── README.md
└── .gitignore




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


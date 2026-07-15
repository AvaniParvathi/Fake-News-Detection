import os
import sys
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------
# Project Paths
# ---------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")

sys.path.append(SRC_DIR)

from text_cleaner import clean_text

# ---------------------------------------------------
# Load Model & Tokenizer
# ---------------------------------------------------

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_ann_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "models", "tokenizer.pkl")

ann_model = load_model(MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)

MAX_LENGTH = 300

# ---------------------------------------------------
# Prediction Function
# ---------------------------------------------------

def predict_news(news_text):
    """
    Predict whether the given news article is Fake or True.
    """

    # Clean the input text
    cleaned_text = clean_text(news_text)

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([cleaned_text])

    # Pad sequence
    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    # Predict probability
    probability = ann_model.predict(
        padded_sequence,
        verbose=0
    )[0][0]

    # Determine prediction
    if probability >= 0.5:
        prediction = "🟢 True News"
        confidence = probability
    else:
        prediction = "🔴 Fake News"
        confidence = 1 - probability

    return prediction, confidence
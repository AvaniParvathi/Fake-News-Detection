import streamlit as st

from predictor import predict_news

# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ---------------------------------
# Title
# ---------------------------------

st.title("📰 Fake News Detection System")

st.write(
    """
    This application uses an Artificial Neural Network (ANN)
    to classify whether a news article is **Fake** or **True**.
    """
)

# ---------------------------------
# User Input
# ---------------------------------

news_text = st.text_area(
    "Paste the news article below:",
    height=250
)

# ---------------------------------
# Prediction Button
# ---------------------------------

if st.button("Predict"):

    if news_text.strip() == "":
        st.warning("Please enter a news article.")
    else:

        prediction, confidence = predict_news(news_text)

        st.subheader("Prediction")

        if "True" in prediction:
            st.success(prediction)
        else:
            st.error(prediction)

        st.subheader("Confidence")

        st.write(f"{confidence*100:.2f}%")
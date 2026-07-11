from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from preprocessing import preprocess_data


def prepare_features():

    # Load cleaned dataset
    news_df = preprocess_data()

    # Features and labels
    X = news_df["content"]
    y = news_df["label"]

    # TF-IDF Feature Extraction
    tfidf = TfidfVectorizer(max_features=5000)

    X = tfidf.fit_transform(X)

    # Split into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return news_df, X_train, X_test, y_train, y_test, tfidf


if __name__ == "__main__":

    news_df, X_train, X_test, y_train, y_test, tfidf = prepare_features()

    print("Training Data :", X_train.shape)
    print("Testing Data  :", X_test.shape)
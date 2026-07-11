import pandas as pd
from text_cleaner import clean_text


def preprocess_data():
    """
    Loads, preprocesses, and returns the cleaned news dataset.
    """

    # Load datasets
    fake_df = pd.read_csv("../data/fake.csv")
    true_df = pd.read_csv("../data/true.csv")

    # Create labels
    fake_df["label"] = 0
    true_df["label"] = 1

    # Merge datasets
    news_df = pd.concat([fake_df, true_df], ignore_index=True)

    # Remove duplicate rows
    news_df = news_df.drop_duplicates()

    # Create a single content column
    news_df["content"] = news_df["title"] + " " + news_df["text"]

    # Clean the text
    news_df["content"] = news_df["content"].apply(clean_text)

    # Keep only required columns
    news_df = news_df[["content", "label"]]

    return news_df


if __name__ == "__main__":

    news_df = preprocess_data()

    print("=" * 50)
    print("First 5 Rows")
    print("=" * 50)
    print(news_df.head())

    print("\nDataset Shape:", news_df.shape)

    print("\nClass Distribution:")
    print(news_df["label"].value_counts())
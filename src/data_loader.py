import pandas as pd

# Load Fake News dataset
fake_df = pd.read_csv("data/Fake.csv")

# Load True News dataset
true_df = pd.read_csv("data/True.csv")

# Display the first five rows of each dataset
print("===== Fake News Dataset =====")
print(fake_df.head())

print("\n")

print("===== True News Dataset =====")
print(true_df.head())

print("\n========== Dataset Shape ==========")

print(f"Fake News Dataset Shape: {fake_df.shape}")
print(f"True News Dataset Shape: {true_df.shape}")
from preprocess import process_earc_dataset
import os

# Create output directory if it doesn't exist
os.makedirs("data/processed", exist_ok=True)

# Process papers with correct path (space instead of underscore)
df = process_earc_dataset(base_path="data/raw/EARC Dataset/Reference")

# Save processed papers
df.to_pickle("data/processed/processed_papers.pkl")
print("Papers processed and saved successfully!") 
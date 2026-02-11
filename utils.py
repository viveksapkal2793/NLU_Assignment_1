import csv
import random

def load_bbc_data(csv_path):
    """Load and balance BBC news dataset (SPORT vs POLITICS only).
    
    Returns:
        list: Balanced dataset of (text, label) tuples, shuffled randomly
    """
    data = []
    sport_data = []
    politics_data = []

    # Read CSV file and filter sport and politics categories
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row['category'].lower()
            text = row['text']
            
            if category == 'sport':
                sport_data.append((text, "SPORT"))
            elif category == 'politics':
                politics_data.append((text, "POLITICS"))

    # Balance the dataset: use equal number of samples from each class
    n = min(len(sport_data), len(politics_data))
    sport_data = sport_data[:n]
    politics_data = politics_data[:n]

    # Combine and shuffle to prevent ordering bias
    data = sport_data + politics_data
    random.shuffle(data)
    return data


def train_test_split(data, ratio=0.8):
    split = int(len(data) * ratio)
    return data[:split], data[split:]

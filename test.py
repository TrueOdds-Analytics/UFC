import pandas as pd
import numpy as np

def main():
    # Load the combined sorted fights data
    file_path = 'data/combined_sorted_fighter_stats.csv'
    df = pd.read_csv(file_path)

    # Calculate Elo difference
    # Predict winner based on Elo difference
    df['predicted_winner'] = (df['elo_difference'] > 0).astype(int)

    # Calculate accuracy
    correct_predictions = (df['predicted_winner'] == df['winner']).sum()
    total_fights = len(df)
    accuracy = correct_predictions / total_fights

    print(f"Total fights: {total_fights}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f}")

    # Additional metrics
    true_positives = ((df['predicted_winner'] == 1) & (df['winner'] == 1)).sum()
    true_negatives = ((df['predicted_winner'] == 0) & (df['winner'] == 0)).sum()
    false_positives = ((df['predicted_winner'] == 1) & (df['winner'] == 0)).sum()
    false_negatives = ((df['predicted_winner'] == 0) & (df['winner'] == 1)).sum()

    print(f"\nTrue Positives: {true_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")

    # Calculate class-wise accuracy
    class_1_accuracy = true_positives / (true_positives + false_negatives)
    class_0_accuracy = true_negatives / (true_negatives + false_positives)

    print(f"\nClass 1 (Higher Elo wins) Accuracy: {class_1_accuracy:.4f}")
    print(f"Class 0 (Lower Elo wins) Accuracy: {class_0_accuracy:.4f}")

if __name__ == "__main__":
    main()
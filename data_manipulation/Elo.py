import pandas as pd
import re
from datetime import datetime

def calculate_elo_ratings(file_path, initial_rating=1500):
    # Read and sort the CSV file
    df = pd.read_csv(file_path)
    df['fight_date'] = pd.to_datetime(df['fight_date'])
    df = df.sort_values(by=['fight_date', 'id'])

    # Weight class factors (unchanged)
    weight_class_factors = {
        'Flyweight': {'ko': 1.3, 'submission': 1.2, 'decision': 0.7},
        'Bantamweight': {'ko': 1.25, 'submission': 1.15, 'decision': 0.80},
        'Featherweight': {'ko': 1.2, 'submission': 1.1, 'decision': 0.9},
        'Lightweight': {'ko': 1.15, 'submission': 1.05, 'decision': 0.95},
        'Welterweight': {'ko': 1.1, 'submission': 1.05, 'decision': 1.0},
        'Middleweight': {'ko': 1.0, 'submission': 1.0, 'decision': 1.0},
        'Light Heavyweight': {'ko': 0.9, 'submission': 0.95, 'decision': 1.0},
        'Heavyweight': {'ko': 0.8, 'submission': 0.9, 'decision': 1.0},
        'Catch Weight': {'ko': 1.0, 'submission': 1.0, 'decision': 1.0},
        'Open Weight': {'ko': 1.0, 'submission': 1.0, 'decision': 1.0},
        'Tournament': {'ko': 1.0, 'submission': 1.0, 'decision': 1.0},
        'UFC Superfight Championship': {'ko': 1.0, 'submission': 1.0, 'decision': 1.0}
    }

    # Helper functions (mostly unchanged)
    def get_weight_class_factor(weight_class, result):
        for key, factors in weight_class_factors.items():
            if re.search(key, weight_class, re.IGNORECASE):
                if result in [0, 3]:  # KO/TKO
                    return factors['ko']
                elif result == 2:  # Submission
                    return factors['submission']
                else:  # Decision or other
                    return factors['decision']
        return 1.0  # Default factor if no match is found

    def is_title_fight(weight_class):
        return 'title' in weight_class.lower()

    def get_margin_factor(method):
        margin_factors = {3: 6, 0: 6, 1: 5, 2: 3}
        return margin_factors.get(method, 1.0)

    def get_age_factor(age):
        if age < 27:
            return 1.15
        elif 27 <= age < 32:
            return 1.0
        else:
            return 0.85

    def get_additional_factors(win_streak, loss_streak, years_experience, days_since_last_fight):
        streak_factor = 1 + (win_streak * 0.02) - (loss_streak * 0.02)
        experience_factor = min(1 + (years_experience * 0.01), 1.2)
        inactivity_factor = max(1 - (days_since_last_fight / 365 * 0.1), 0.9)
        return streak_factor * experience_factor * inactivity_factor

    # New function for dynamic K-factor
    def get_dynamic_k_factor(fights, rating):
        base_k = 20
        if fights < 5:
            return base_k * 2  # Higher K-factor for new fighters
        elif fights < 15:
            return base_k * 1.5
        elif rating > 2400:
            return base_k / 2  # Lower K-factor for high-rated fighters
        else:
            return base_k

    # Initialize Elo ratings and fight count dictionaries
    elo_ratings = {}
    fight_counts = {}

    # First pass: Calculate Elo ratings
    for index, fighter in df.iterrows():
        opponent = df[(df['id'] == fighter['id']) & (df['fighter'] != fighter['fighter'])].iloc[0]

        fighter_rating = elo_ratings.get(fighter['fighter'], initial_rating)
        opponent_rating = elo_ratings.get(opponent['fighter'], initial_rating)

        fighter_fights = fight_counts.get(fighter['fighter'], 0)
        opponent_fights = fight_counts.get(opponent['fighter'], 0)

        # Store pre-fight Elo
        df.at[index, 'pre_fight_elo'] = fighter_rating

        # Calculate Elo change factors
        weight_class_factor = get_weight_class_factor(fighter['weight_class'], fighter['result'])
        title_multiplier = 1.5 if is_title_fight(fighter['weight_class']) else 1.0
        margin_factor = get_margin_factor(fighter['result'])
        age_factor = get_age_factor(fighter['age'])
        additional_factor = get_additional_factors(
            fighter['win_streak'], fighter['loss_streak'],
            fighter['years_of_experience'], fighter['days_since_last_fight']
        )

        # Calculate dynamic K-factor
        k = get_dynamic_k_factor(fighter_fights, fighter_rating)

        # Calculate Elo change
        expected_score = 1 / (1 + 10 ** ((opponent_rating - fighter_rating) / 400))
        actual_score = 1 if fighter['winner'] == 1 else 0
        elo_change = k * margin_factor * weight_class_factor * age_factor * additional_factor * title_multiplier * (actual_score - expected_score)

        # Update Elo rating for the fighter
        new_rating = fighter_rating + elo_change
        elo_ratings[fighter['fighter']] = new_rating

        # Update fight counts
        fight_counts[fighter['fighter']] = fighter_fights + 1
        fight_counts[opponent['fighter']] = opponent_fights + 1

        # Update DataFrame column for post-fight Elo
        df.at[index, 'fight_outcome_elo'] = new_rating

    # Second pass: Calculate accuracy (unchanged)
    correct_predictions = total_predictions = total_fights = fights_with_elo_difference = 0
    threshold = -1

    for index, fighter in df.iterrows():
        opponent = df[(df['id'] == fighter['id']) & (df['fighter'] != fighter['fighter'])].iloc[0]

        elo_difference = fighter['pre_fight_elo'] - opponent['pre_fight_elo']
        df.at[index, 'elo_difference'] = elo_difference

        if abs(elo_difference) > threshold:
            fights_with_elo_difference += 1
            predicted_winner = 1 if elo_difference > 0 else 0
            actual_winner = fighter['winner']
            if predicted_winner == actual_winner:
                correct_predictions += 1
            total_predictions += 1
        total_fights += 1

    # Calculate and print statistics
    prediction_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    elo_difference_percentage = (fights_with_elo_difference / total_fights * 100) if total_fights > 0 else 0

    print(f"\nPrediction Accuracy: {prediction_accuracy:.2f}%")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Total Predictions: {total_predictions}")
    print(f"Total Fights: {total_fights}")
    print(f"Percentage of fights with Elo difference > {threshold}: {elo_difference_percentage:.2f}%")

    print("\nFighters with the Highest Elo Ratings:")
    print("-------------------------------------")
    top_fighters = pd.Series(elo_ratings).sort_values(ascending=False).head(50)
    print(top_fighters)

    # Save the updated DataFrame
    df.to_csv(file_path, index=False)

    return df

if __name__ == "__main__":
    file_path = "../data/combined_rounds.csv"
    calculate_elo_ratings(file_path)
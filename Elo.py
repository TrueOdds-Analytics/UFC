import pandas as pd
import numpy as np

def calculate_elo_ratings(file_path, k=20, initial_rating=1500):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Sort the DataFrame by id and fight_date
    df = df.sort_values(by=['id', 'fight_date'])

    # Weight class mapping
    weight_class_map = {
        0: 'Featherweight', 1: 'Lightweight', 2: 'Heavyweight', 3: 'Bantamweight',
        4: 'Welterweight', 5: 'Light Heavyweight', 6: 'Middleweight', 7: 'Catch Weight Bout',
        8: 'Open Weight Bout', 9: 'Flyweight', 10: 'Tournament',
        11: 'UFC Superfight Championship Bout'
    }

    # Weight class factors for KO and Decision
    weight_class_factors = {
        'Flyweight': {'ko': 1.3, 'decision': 0.7},
        'Bantamweight': {'ko': 1.25, 'decision': 0.80},
        'Featherweight': {'ko': 1.2, 'decision': 0.9},
        'Lightweight': {'ko': 1.15, 'decision': 0.95},
        'Welterweight': {'ko': 1.1, 'decision': 1.0},
        'Middleweight': {'ko': 1.0, 'decision': 1.0},
        'Light Heavyweight': {'ko': 0.9, 'decision': 1.0},
        'Heavyweight': {'ko': 0.8, 'decision': 1.0},
        'Catch Weight Bout': {'ko': 1.0, 'decision': 1.0},
        'Open Weight Bout': {'ko': 1.0, 'decision': 1.0},
        'Tournament': {'ko': 1.0, 'decision': 1.0},
        'UFC Superfight Championship Bout': {'ko': 1.0, 'decision': 1.0}
    }

    # Initialize the Elo ratings as float
    elo_ratings = pd.Series(float(initial_rating), index=df['fighter'].unique())

    # Define the margin factors
    margin_factors = {3: 6, 0: 6, 1: 5, 2: 3}

    # Function to get margin factor
    def get_margin_factor(method):
        return margin_factors.get(method, 1.0)

    # Function to calculate age factor
    def get_age_factor(age):
        if age < 27:
            return 1.15  # Young fighters may improve more rapidly
        elif 27 <= age < 32:
            return 1.0  # Prime fighting age
        else:
            return 0.85  # Older fighters may decline more rapidly

    # Group the DataFrame by 'id' to pair fighters
    grouped = df.groupby('id')

    correct_predictions = 0
    total_predictions = 0
    total_fights = 0
    fights_with_elo_difference = 0

    # Iterate over each group (fight)
    for _, fight in grouped:
        if len(fight) != 2:
            continue  # Skip if there aren't exactly two fighters

        fighter1, fighter2 = fight.iloc[0], fight.iloc[1]
        weight_class = weight_class_map[fighter1['weight_class']]

        # Get current ratings for both fighters
        fighter1_rating = elo_ratings[fighter1['fighter']]
        fighter2_rating = elo_ratings[fighter2['fighter']]

        # Check Elo difference
        elo_difference = abs(fighter1_rating - fighter2_rating)
        if elo_difference > 50:
            fights_with_elo_difference += 1
            predicted_winner = 1 if fighter1_rating > fighter2_rating else 0
            actual_winner = fighter1['winner']

            # Update prediction accuracy
            if predicted_winner == actual_winner:
                correct_predictions += 1
            total_predictions += 1

        total_fights += 1

        # Calculate the margin factor
        margin_factor = get_margin_factor(fighter1['result'])

        # Determine if the fight ended in KO or decision
        is_ko = fighter1['result'] in [0, 3]  # Assuming 0 and 3 represent KO/TKO
        result_type = 'ko' if is_ko else 'decision'

        # Get the appropriate weight class factor
        weight_class_factor = weight_class_factors[weight_class][result_type]

        # Calculate age factors
        age_factor1 = get_age_factor(fighter1['age'])
        age_factor2 = get_age_factor(fighter2['age'])

        # Calculate the expected scores
        expected_score1 = 1 / (1 + 10 ** ((fighter2_rating - fighter1_rating) / 400))
        expected_score2 = 1 - expected_score1

        # Determine actual scores
        actual_score1 = 1 if fighter1['winner'] == 1 else 0
        actual_score2 = 1 - actual_score1

        # Calculate the rating changes, scaled by weight class factor and age factor
        rating_change1 = k * margin_factor * weight_class_factor * age_factor1 * (actual_score1 - expected_score1)
        rating_change2 = k * margin_factor * weight_class_factor * age_factor2 * (actual_score2 - expected_score2)

        # Update the fighters' Elo ratings
        elo_ratings[fighter1['fighter']] += rating_change1
        elo_ratings[fighter2['fighter']] += rating_change2

        # Update the 'elo' column in the DataFrame with the current Elo ratings
        df.loc[df['id'] == fighter1['id'], 'elo'] = fighter1_rating
        df.loc[df['id'] == fighter2['id'], 'elo'] = fighter2_rating

    # Calculate prediction accuracy and percentage of fights with Elo difference > 100
    prediction_accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    elo_difference_percentage = (fights_with_elo_difference / total_fights * 100) if total_fights > 0 else 0

    print(f"\nPrediction Accuracy: {prediction_accuracy:.2f}%")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Total Predictions: {total_predictions}")
    print(f"Total Fights: {total_fights}")
    print(f"Percentage of fights with Elo difference > 50: {elo_difference_percentage:.2f}%")

    # Display top fighters
    print("\nFighters with the Highest Elo Ratings:")
    print("-------------------------------------")
    top_fighters = elo_ratings.sort_values(ascending=False).head(25)
    print(top_fighters)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path, index=False)

    return df, prediction_accuracy, elo_ratings, elo_difference_percentage

# Example usage
file_path = "data/combined_rounds.csv"
updated_df, accuracy, final_elo_ratings, elo_diff_percentage = calculate_elo_ratings(file_path)
import pandas as pd
import re
import os
from datetime import datetime
import shutil


def calculate_elo_ratings(file_path, initial_rating=1500):
    """
    Calculate Elo ratings for UFC fighters based on their fight history.
    Modifies the input CSV file by adding pre_fight_elo, fight_outcome_elo, and elo_difference columns.
    """
    # Print file path for debugging
    print(f"Processing Elo ratings for file: {file_path}")

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        return None

    # Read and sort the CSV file
    df = pd.read_csv(file_path)
    print(f"Successfully read file with {len(df)} rows")

    # Convert date column
    df['fight_date'] = pd.to_datetime(df['fight_date'])
    df = df.sort_values(by=['fight_date', 'id']).reset_index(drop=True)

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

    # Helper functions (unchanged)
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

    # Make sure we have these columns
    if 'pre_fight_elo' not in df.columns:
        df['pre_fight_elo'] = initial_rating
    if 'fight_outcome_elo' not in df.columns:
        df['fight_outcome_elo'] = initial_rating
    if 'elo_difference' not in df.columns:
        df['elo_difference'] = 0.0

    print("Starting Elo rating calculations...")

    # Get unique fight IDs sorted by date
    unique_fights = df.groupby('id')['fight_date'].first().sort_values()
    processed_fights = 0

    # Process fights in pairs
    for fight_id, fight_date in unique_fights.items():
        fight_rows = df[df['id'] == fight_id]

        if len(fight_rows) != 2:
            print(f"Warning: Fight {fight_id} has {len(fight_rows)} rows, skipping...")
            continue

        # Get both fighter rows
        fighter1_row = fight_rows.iloc[0]
        fighter2_row = fight_rows.iloc[1]

        fighter1_idx = fighter1_row.name
        fighter2_idx = fighter2_row.name

        fighter1_name = fighter1_row['fighter']
        fighter2_name = fighter2_row['fighter']

        # Get CURRENT ratings for both fighters BEFORE any updates
        fighter1_rating = elo_ratings.get(fighter1_name, initial_rating)
        fighter2_rating = elo_ratings.get(fighter2_name, initial_rating)

        # Get fight counts
        fighter1_fights = fight_counts.get(fighter1_name, 0)
        fighter2_fights = fight_counts.get(fighter2_name, 0)

        # Store pre-fight ELOs for both fighters
        df.at[fighter1_idx, 'pre_fight_elo'] = fighter1_rating
        df.at[fighter2_idx, 'pre_fight_elo'] = fighter2_rating

        # Store ELO differences
        df.at[fighter1_idx, 'elo_difference'] = fighter1_rating - fighter2_rating
        df.at[fighter2_idx, 'elo_difference'] = fighter2_rating - fighter1_rating

        # Calculate factors for fighter 1
        weight_class_factor1 = get_weight_class_factor(fighter1_row['weight_class'], fighter1_row['result'])
        title_multiplier1 = 1.5 if is_title_fight(fighter1_row['weight_class']) else 1.0
        margin_factor1 = get_margin_factor(fighter1_row['result'])
        age_factor1 = get_age_factor(fighter1_row['age'])
        additional_factor1 = get_additional_factors(
            fighter1_row['win_streak'], fighter1_row['loss_streak'],
            fighter1_row['years_of_experience'], fighter1_row['days_since_last_fight']
        )
        k1 = get_dynamic_k_factor(fighter1_fights, fighter1_rating)

        # Calculate factors for fighter 2
        weight_class_factor2 = get_weight_class_factor(fighter2_row['weight_class'], fighter2_row['result'])
        title_multiplier2 = 1.5 if is_title_fight(fighter2_row['weight_class']) else 1.0
        margin_factor2 = get_margin_factor(fighter2_row['result'])
        age_factor2 = get_age_factor(fighter2_row['age'])
        additional_factor2 = get_additional_factors(
            fighter2_row['win_streak'], fighter2_row['loss_streak'],
            fighter2_row['years_of_experience'], fighter2_row['days_since_last_fight']
        )
        k2 = get_dynamic_k_factor(fighter2_fights, fighter2_rating)

        # Calculate expected scores using PRE-FIGHT ratings
        expected_score1 = 1 / (1 + 10 ** ((fighter2_rating - fighter1_rating) / 400))
        expected_score2 = 1 / (1 + 10 ** ((fighter1_rating - fighter2_rating) / 400))

        # Get actual scores
        actual_score1 = 1 if fighter1_row['winner'] == 1 else 0
        actual_score2 = 1 if fighter2_row['winner'] == 1 else 0

        # Calculate ELO changes
        elo_change1 = k1 * margin_factor1 * weight_class_factor1 * age_factor1 * additional_factor1 * title_multiplier1 * (
                    actual_score1 - expected_score1)
        elo_change2 = k2 * margin_factor2 * weight_class_factor2 * age_factor2 * additional_factor2 * title_multiplier2 * (
                    actual_score2 - expected_score2)

        # Calculate new ratings
        new_rating1 = fighter1_rating + elo_change1
        new_rating2 = fighter2_rating + elo_change2

        # Update post-fight ELOs in dataframe
        df.at[fighter1_idx, 'fight_outcome_elo'] = new_rating1
        df.at[fighter2_idx, 'fight_outcome_elo'] = new_rating2

        # NOW update the ELO ratings dictionary for future fights
        elo_ratings[fighter1_name] = new_rating1
        elo_ratings[fighter2_name] = new_rating2

        # Update fight counts
        fight_counts[fighter1_name] = fighter1_fights + 1
        fight_counts[fighter2_name] = fighter2_fights + 1

        processed_fights += 1
        if processed_fights % 1000 == 0:
            print(f"Processed {processed_fights} fights...")

    print(f"Completed ELO calculation for {processed_fights} fights")
    print("Calculating prediction accuracy...")

    # Calculate accuracy
    correct_predictions = 0
    total_predictions = 0
    total_fights = 0
    fights_with_elo_difference = 0
    threshold = -1

    for fight_id in df['id'].unique():
        fight_rows = df[df['id'] == fight_id]
        if len(fight_rows) != 2:
            continue

        fighter1_row = fight_rows.iloc[0]
        fighter2_row = fight_rows.iloc[1]

        elo_difference = fighter1_row['pre_fight_elo'] - fighter2_row['pre_fight_elo']

        if abs(elo_difference) > threshold:
            fights_with_elo_difference += 1
            predicted_winner = 1 if elo_difference > 0 else 0
            actual_winner = fighter1_row['winner']

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

    # Save the data
    try:
        # Save using a temporary file first
        temp_file = file_path + '.tmp'
        print(f"Saving Elo data to temporary file: {temp_file}")
        df.to_csv(temp_file, index=False)

        # Move the temporary file to the target location
        print(f"Moving temporary file to final location: {file_path}")
        shutil.move(temp_file, file_path)

        # Verify the file was saved correctly
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"File saved successfully. Size: {file_size} bytes")
        else:
            print("ERROR: File not found after save attempt")
    except Exception as e:
        print(f"ERROR saving data: {str(e)}")
        print("Attempting direct save...")
        try:
            df.to_csv(file_path, index=False)
            print("Direct save completed")
        except Exception as e2:
            print(f"Direct save also failed: {str(e2)}")

    # Verify the data was saved with Elo columns
    print(f"\nVerifying Elo data was added to {file_path}...")
    try:
        verification_df = pd.read_csv(file_path)
        for col in ['pre_fight_elo', 'fight_outcome_elo', 'elo_difference']:
            if col in verification_df.columns:
                non_null = verification_df[col].notna().sum()
                print(f"Verification: {col} column exists with {non_null} non-null values")
            else:
                print(f"ERROR: {col} column not found in saved file")
    except Exception as e:
        print(f"Verification failed: {str(e)}")

    return df


if __name__ == "__main__":
    # When run directly, use the project root relative path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../.."))
    file_path = os.path.join(project_root, "data/processed/combined_rounds.csv")

    print(f"Running Elo calculation on {file_path}")
    calculate_elo_ratings(file_path)
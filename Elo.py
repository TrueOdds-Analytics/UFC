import pandas as pd


def calculate_elo_ratings(file_path, k=20, initial_rating=1500):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Sort the DataFrame by fight date, oldest first
    df = df.sort_values(by=['fight_date', 'id'])

    # Weight class mapping and factors
    weight_class_map = {
        0: 'Featherweight', 1: 'Lightweight', 2: 'Heavyweight', 3: 'Bantamweight',
        4: 'Welterweight', 5: 'Light Heavyweight', 6: 'Middleweight', 7: 'Catch Weight Bout',
        8: 'Open Weight Bout', 9: 'Flyweight', 10: 'Tournament',
        11: 'UFC Superfight Championship Bout'
    }
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

    # Initialize Elo ratings dictionary
    elo_ratings = {}

    # Helper functions
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

    # Statistics tracking
    correct_predictions = 0
    total_predictions = 0
    total_fights = 0
    fights_with_elo_difference = 0

    # Process each fight row by row
    for _, fight in df.iterrows():
        fighter1 = fight
        fighter2 = df[(df['id'] == fighter1['id']) & (df['fighter'] != fighter1['fighter'])].iloc[0]

        # Get or set initial Elo ratings
        # Get or set initial Elo ratings
        fighter1_rating = elo_ratings.get(fighter1['fighter'], initial_rating)
        fighter2_rating = elo_ratings.get(fighter2['fighter'], initial_rating)

        # Calculate win probabilities
        fighter1_win_prob = 1 / (1 + 10 ** ((fighter2_rating - fighter1_rating) / 400))

        # Update win probability for fighter1 in DataFrame
        df.loc[df.index == fighter1.name, 'elo_win_probability'] = fighter1_win_prob

        # Prediction tracking
        threshold = 50
        elo_difference = abs(fighter1_rating - fighter2_rating)
        if elo_difference > threshold:
            fights_with_elo_difference += 1
            predicted_winner = 1 if fighter1_rating > fighter2_rating else 0
            actual_winner = fighter1['winner']
            if predicted_winner == actual_winner:
                correct_predictions += 1
            total_predictions += 1
        total_fights += 1

        # Determine the winning and losing fighters
        if fighter1['winner'] == 1:
            winner, loser = fighter1, fighter2
        else:
            winner, loser = fighter2, fighter1

        winner_rating = elo_ratings.get(winner['fighter'], initial_rating)
        loser_rating = elo_ratings.get(loser['fighter'], initial_rating)

        # Calculate Elo change factors
        weight_class = weight_class_map[winner['weight_class']]
        margin_factor = get_margin_factor(winner['result'])
        is_ko = winner['result'] in [0, 3]
        weight_class_factor = weight_class_factors[weight_class]['ko' if is_ko else 'decision']
        age_factor = get_age_factor(winner['age'])
        additional_factor = get_additional_factors(
            winner['win_streak'], winner['loss_streak'],
            winner['years_of_experience'], winner['days_since_last_fight']
        )

        # Calculate Elo change
        expected_score = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
        actual_score = 1  # Winner always gets a score of 1
        elo_change = k * margin_factor * weight_class_factor * age_factor * additional_factor * (
                    actual_score - expected_score)

        # Update Elo ratings for next fight
        elo_ratings[winner['fighter']] = winner_rating + elo_change
        elo_ratings[loser['fighter']] = loser_rating - elo_change

        # Update 'elo' column in DataFrame for fighter1 with the rating used for this fight
        df.loc[df.index == fighter1.name, 'elo'] = fighter1_rating

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
    top_fighters = pd.Series(elo_ratings).sort_values(ascending=False).head(25)
    print(top_fighters)

    # Save the updated DataFrame
    df.to_csv(file_path, index=False)

    return df

if __name__ == "__main__":
    file_path = "data/combined_rounds.csv"
    calculate_elo_ratings(file_path)

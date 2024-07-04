import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def create_specific_matchup_data(file_path, fighter_name, opponent_name, n_past_fights, name=False):
    df = pd.read_csv(file_path)

    # Convert fighter names to lowercase
    fighter_name = fighter_name.lower()
    opponent_name = opponent_name.lower()
    df['fighter'] = df['fighter'].str.lower()
    df['fighter_b'] = df['fighter_b'].str.lower()

    # Load the removed features from the file
    with open('data/train test data/removed_features.txt', 'r') as file:
        removed_features = file.read().split(',')

    # Define the features to include for averaging, excluding identifiers, non-numeric features, and removed features
    columns_to_exclude = ['fighter', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
                          'result', 'winner', 'weight_class', 'scheduled_rounds',
                          'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']
    features_to_include = [col for col in df.columns if
                           col not in columns_to_exclude and col not in removed_features and 'age' not in col.lower()]

    matchup_data = []

    # Get the last 'n' fights for Fighter A and Fighter B before the specified matchup
    fighter_df = df[(df['fighter'] == fighter_name)].sort_values(by='fight_date', ascending=False).head(n_past_fights)
    opponent_df = df[(df['fighter'] == opponent_name)].sort_values(by='fight_date', ascending=False).head(n_past_fights)

    # Check if either fighter doesn't have enough fights
    if len(fighter_df) < n_past_fights or len(opponent_df) < n_past_fights:
        print("Specific matchup failure: One of the fighters doesn't have enough fights.")
        return None

    # Calculate the average of the relevant columns over the past 'n' fights
    fighter_features = fighter_df[features_to_include].mean().values
    opponent_features = opponent_df[features_to_include].mean().values

    # Create new columns for the specified features for each of the last three fights
    results_fighter = fighter_df[['result', 'winner', 'weight_class', 'scheduled_rounds']].head(3).values.flatten()
    results_opponent = opponent_df[['result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']].head(
        3).values.flatten()

    # Get user input for current fight odds and ages
    current_fight_open_odds = float(input(f"Enter current open odds for {fighter_name}: "))
    current_fight_open_odds_b = float(input(f"Enter current open odds for {opponent_name}: "))
    current_fight_age = float(input(f"Enter current age for {fighter_name}: "))
    current_fight_age_b = float(input(f"Enter current age for {opponent_name}: "))

    # Calculate differentials
    current_fight_open_odds_diff = current_fight_open_odds - current_fight_open_odds_b
    current_fight_age_diff = current_fight_age - current_fight_age_b

    # Add current fight information to the features
    current_fight_info = [current_fight_open_odds, current_fight_open_odds_b, current_fight_open_odds_diff,
                          current_fight_age, current_fight_age_b, current_fight_age_diff]

    combined_features = np.concatenate(
        [fighter_features, opponent_features, results_fighter, results_opponent, current_fight_info])

    # Get the most recent fight date among the averaged fights
    most_recent_date = max(fighter_df['fight_date'].max(), opponent_df['fight_date'].max())

    # Add the combined features, most recent fight date, and fighter names to the dataset
    matchup_data.append([fighter_name, opponent_name, most_recent_date] + combined_features.tolist())

    # Define column names for the new DataFrame
    results_columns = []
    for i in range(1, 4):
        results_columns += [f"result_fight_{i}", f"winner_fight_{i}", f"weight_class_fight_{i}",
                            f"scheduled_rounds_fight_{i}"]
        results_columns += [f"result_b_fight_{i}", f"winner_b_fight_{i}", f"weight_class_b_fight_{i}",
                            f"scheduled_rounds_b_fight_{i}"]

    column_names = ['fighter', 'fighter_b', 'fight_date'] + \
                   [f"{feature}_fighter_avg_last_{n_past_fights}" for feature in features_to_include] + \
                   [f"{feature}_fighter_b_avg_last_{n_past_fights}" for feature in features_to_include] + \
                   results_columns + \
                   ['current_fight_open_odds', 'current_fight_open_odds_b', 'current_fight_open_odds_diff',
                    'current_fight_age', 'current_fight_age_b', 'current_fight_age_diff']

    # Convert the matchup data into a DataFrame
    matchup_df = pd.DataFrame(matchup_data, columns=column_names)

    # Drop the specified columns from the removed features
    matchup_df = matchup_df.drop(columns=[col for col in removed_features if col in matchup_df.columns], axis=1)

    # Drop the 'fight_date' column
    matchup_df = matchup_df.drop(['fight_date'], axis=1)

    # Remove 'fighter' and 'fighter_b' columns if name is False
    if not name:
        matchup_df = matchup_df.drop(['fighter', 'fighter_b'], axis=1)

    # Save the specific matchup data to a CSV file
    csv_name = f'specific_matchup.csv'
    matchup_df.to_csv(f'data/{csv_name}', index=False)

    print("Specific matchup success. Data saved to CSV.")
    return matchup_df


def preprocess_data(data):
    # Convert specified columns to category type
    category_columns = [
        'result_fight_1', 'winner_fight_1', 'weight_class_fight_1', 'scheduled_rounds_fight_1',
        'result_b_fight_1', 'winner_b_fight_1', 'weight_class_b_fight_1', 'scheduled_rounds_b_fight_1',
        'result_fight_2', 'winner_fight_2', 'weight_class_fight_2', 'scheduled_rounds_fight_2',
        'result_b_fight_2', 'winner_b_fight_2', 'weight_class_b_fight_2', 'scheduled_rounds_b_fight_2',
        'result_fight_3', 'winner_fight_3', 'weight_class_fight_3', 'scheduled_rounds_fight_3',
        'result_b_fight_3', 'winner_b_fight_3', 'weight_class_b_fight_3', 'scheduled_rounds_b_fight_3'
    ]

    data[category_columns] = data[category_columns].astype("category")
    return data


def load_model(model_path):
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(model_path)
    return model


def predict_outcome(model, specific_data):
    # Make predictions using the loaded model
    predictions = model.predict(specific_data)
    probabilities = model.predict_proba(specific_data)

    return predictions[0], probabilities[0]


def evaluate_model(model, test_data, test_labels):
    predictions = []
    probabilities = []

    for _, row in test_data.iterrows():
        pred, prob = predict_outcome(model, row.to_frame().T)
        predictions.append(pred)
        probabilities.append(prob)

    predictions = pd.Series(predictions)

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return predictions, probabilities


if __name__ == "__main__":
    # Load the trained model
    model_path = 'models/xgboost/model_0.7260_338_features.json'  # Replace with the actual model file path
    model = load_model(model_path)

    # Load and preprocess the test set
    test_data = pd.read_csv('data/train test data/test_data.csv')

    # Separate features and labels
    test_labels = test_data['winner']
    test_features = test_data.drop('winner', axis=1)

    # Preprocess the test features
    test_features = preprocess_data(test_features)

    # Evaluate the model on the test set
    predictions, probabilities = evaluate_model(model, test_features, test_labels)

    # Print some example predictions
    print("\nExample predictions:")
    for i in range(min(10, len(test_data))):
        true_winner = "Fighter A" if test_labels.iloc[i] == 1 else "Fighter B"
        predicted_winner = "Fighter A" if predictions[i] == 1 else "Fighter B"
        winning_probability = probabilities[i][1] if predictions[i] == 1 else probabilities[i][0]

        print(f"Fight {i + 1}")
        print(f"True Winner: {true_winner}")
        print(f"Predicted Winner: {predicted_winner}")
        print(f"Probability of {predicted_winner} winning: {winning_probability:.2%}")
        print("---")

    # Calculate overall prediction accuracy
    correct_predictions = sum(predictions == test_labels)
    total_predictions = len(test_labels)
    overall_accuracy = correct_predictions / total_predictions

    print(f"\nOverall Prediction Accuracy: {overall_accuracy:.2%}")
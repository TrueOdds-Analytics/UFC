from data_cleaner import create_specific_matchup_data
import xgboost as xgb


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

    # Remove 'fighter' and 'fighter_b' columns
    data = data.drop(['fighter', 'fighter_b'], axis=1)

    return data


def load_model(model_path):
    model = xgb.XGBClassifier(enable_categorical=True)
    model.load_model(model_path)
    return model


def predict_outcome(model, specific_data, fighter_name, opponent_name):
    # Preprocess the specific data
    specific_data = preprocess_data(specific_data)

    # Make predictions using the loaded model
    predictions = model.predict(specific_data)
    probabilities = model.predict_proba(specific_data)

    if predictions[0] == 1:
        print(f"{fighter_name} wins")
        print(f"Probability of {fighter_name} winning: {probabilities[0][1]:.2%}")
    else:
        print(f"{opponent_name} wins")
        print(f"Probability of {opponent_name} winning: {probabilities[0][0]:.2%}")

    return None


if __name__ == "__main__":
    # Load the trained model
    model_path = 'models/model_0.7079_0_features_removed.json'  # Replace with the actual model file path
    model = load_model(model_path)

    while True:
        # Get fighter names from user input
        fighter_name = input("Enter the name of Fighter A: ")
        if fighter_name.lower() == 'q':
            break

        opponent_name = input("Enter the name of Fighter B: ")

        # Generate specific matchup data
        specific_data = create_specific_matchup_data('data/combined_sorted_fighter_stats.csv', fighter_name,
                                                     opponent_name, n_past_fights=3)

        if specific_data is None:
            print("Insufficient data for the specified fighters. Please try again.")
            continue

        # Predict the outcome for the specific matchup
        predict_outcome(model, specific_data, fighter_name, opponent_name)

from data_cleaner import create_matchup_data
import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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
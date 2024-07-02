import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt


def preprocess_data(data):
    # Convert specified columns to category type
    category_columns = [
        'result_fight_1', 'winner_fight_1', 'weight_class_fight_1', 'scheduled_rounds_fight_1',
        'result_b_fight_1', 'winner_b_fight_1', 'scheduled_rounds_b_fight_1',
        'result_fight_2', 'winner_fight_2', 'scheduled_rounds_fight_2',
        'result_b_fight_2', 'winner_b_fight_2', 'scheduled_rounds_b_fight_2',
        'result_fight_3', 'winner_fight_3', 'scheduled_rounds_fight_3',
        'result_b_fight_3', 'winner_b_fight_3', 'scheduled_rounds_b_fight_3'
    ]

    data[category_columns] = data[category_columns].astype("category")

    # Remove 'fighter' and 'fighter_b' columns if they exist
    columns_to_drop = ['fighter', 'fighter_b']
    data = data.drop([col for col in columns_to_drop if col in data.columns], axis=1)

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
        winner = fighter_name
        loser = opponent_name
        win_probability = probabilities[0][1]
    else:
        winner = opponent_name
        loser = fighter_name
        win_probability = probabilities[0][0]

    return winner, loser, win_probability


def evaluate_model(model, test_data):
    X_test = test_data.drop(['winner', 'fighter', 'fighter_b'], axis=1, errors='ignore')
    y_test = test_data['winner']

    X_test = preprocess_data(X_test)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    return accuracy, auc, y_pred, y_pred_proba


def plot_roc_curve(y_true, y_pred_proba):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def test_model_on_csv(model_path, csv_path):
    # Load the model
    model = load_model(model_path)

    # Load the test data
    test_data = pd.read_csv(csv_path)

    # Evaluate the model
    accuracy, auc, y_pred, y_pred_proba = evaluate_model(model, test_data)

    # Print results
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test AUC: {auc:.4f}")

    # Plot ROC curve
    plot_roc_curve(test_data['winner'], y_pred_proba)

    # Make predictions for each matchup
    results = []
    for _, row in test_data.iterrows():
        specific_data = row.drop(['winner', 'fighter', 'fighter_b']).to_frame().T
        winner, loser, win_probability = predict_outcome(model, specific_data, row['fighter'], row['fighter_b'])
        results.append({
            'Fighter A': row['fighter'],
            'Fighter B': row['fighter_b'],
            'Predicted Winner': winner,
            'Predicted Loser': loser,
            'Win Probability': win_probability,
            'Actual Winner': row['fighter'] if row['winner'] == 1 else row['fighter_b'],
            'Correct Prediction': (winner == row['fighter']) == (row['winner'] == 1)
        })

    # Create a DataFrame with the results
    results_df = pd.DataFrame(results)

    # Display some sample predictions
    print("\nSample predictions:")
    print(results_df.head(10))

    # Save predictions to a new CSV file
    output_path = csv_path.replace('.csv', '_with_predictions.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    # Calculate and print the percentage of correct predictions
    correct_predictions = results_df['Correct Prediction'].sum()
    total_predictions = len(results_df)
    correct_percentage = (correct_predictions / total_predictions) * 100
    print(f"\nPercentage of correct predictions: {correct_percentage:.2f}%")


if __name__ == "__main__":
    model_path = 'models/xgboost/model_0.7723_0_features_removed.json'
    csv_path = 'data/train test data/val_data.csv'

    test_model_on_csv(model_path, csv_path)
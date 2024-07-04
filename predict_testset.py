import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import numpy as np


def preprocess_data(data):
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
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        model = xgb.XGBClassifier(enable_categorical=True)
        model.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def calculate_profit(odds, stake):
    if odds < 0:
        return (100 / abs(odds)) * stake
    else:
        return (odds / 100) * stake


def calculate_kelly_fraction(p, b, kelly_fraction):
    q = 1 - p
    full_kelly = max(0, (p - q) / b)  # Ensure non-negative fraction
    return full_kelly * kelly_fraction  # Apply fractional Kelly


def evaluate_bets(y_test, y_pred_proba, test_data, confidence_threshold, initial_bankroll=10000, kelly_fraction=0.125):
    total_stake = 0
    total_return = 0
    correct_bets = 0
    total_bets = 0
    confident_predictions = 0
    correct_confident_predictions = 0
    kelly_bankroll = initial_bankroll
    kelly_total_stake = 0
    kelly_total_return = 0

    for i in range(len(test_data)):
        true_winner = "Fighter A" if y_test.iloc[i] == 1 else "Fighter B"
        winning_probability = max(y_pred_proba[i])
        predicted_winner = "Fighter A" if y_pred_proba[i][1] > y_pred_proba[i][0] else "Fighter B"

        if winning_probability >= confidence_threshold:
            confident_predictions += 1
            odds = test_data.iloc[i]['current_fight_open_odds'] if predicted_winner == "Fighter A" else \
            test_data.iloc[i]['current_fight_open_odds_b']

            # Fixed stake betting
            stake = 10
            profit = calculate_profit(odds, stake)

            # Fractional Kelly Criterion betting
            if odds > 0:
                b = odds / 100
            else:
                b = 100 / abs(odds)
            kelly_bet_size = calculate_kelly_fraction(winning_probability, b, kelly_fraction)
            kelly_stake = kelly_bankroll * kelly_bet_size
            kelly_profit = calculate_profit(odds, kelly_stake)

            total_stake += stake
            kelly_total_stake += kelly_stake
            total_bets += 1

            if predicted_winner == true_winner:
                total_return += stake + profit
                kelly_total_return += kelly_stake + kelly_profit
                kelly_bankroll += kelly_profit
                correct_bets += 1
                correct_confident_predictions += 1
            else:
                kelly_bankroll -= kelly_stake

            if confident_predictions <= 10:  # Print only the first 10 confident predictions
                print(f"Fight {i + 1}")
                print(f"True Winner: {true_winner}")
                print(f"Predicted Winner: {predicted_winner}")
                print(f"Confidence: {winning_probability:.2%}")
                print(f"Odds: {odds}")
                print(f"Fixed Stake Potential profit: ${profit:.2f}")
                print(f"Fractional Kelly Stake: ${kelly_stake:.2f}")
                print(f"Fractional Kelly Potential profit: ${kelly_profit:.2f}")
                print("---")

    return (total_stake, total_return, correct_bets, total_bets, confident_predictions,
            correct_confident_predictions, kelly_total_stake, kelly_total_return, kelly_bankroll)


def print_betting_results(total_fights, confident_predictions, correct_confident_predictions, total_bets, correct_bets,
                          total_stake, total_return, confidence_threshold, kelly_total_stake, kelly_total_return,
                          kelly_final_bankroll, initial_bankroll, kelly_fraction):
    confident_accuracy = correct_confident_predictions / confident_predictions if confident_predictions > 0 else 0
    net_profit = total_return - total_stake
    roi = (net_profit / total_stake) * 100 if total_stake > 0 else 0
    kelly_net_profit = kelly_final_bankroll - initial_bankroll
    kelly_roi = (kelly_net_profit / initial_bankroll) * 100

    print(f"\nConfident Prediction Accuracy (≥{confidence_threshold:.0%} confidence): {confident_accuracy:.2%}")
    print(f"\nBetting Results ({confidence_threshold:.0%} confidence threshold):")
    print(f"Total fights: {total_fights}")
    print(f"Fights predicted with ≥{confidence_threshold:.0%} confidence: {confident_predictions}")
    print(f"Correct predictions: {correct_confident_predictions}")
    print(f"Total bets: {total_bets}")
    print(f"Correct bets: {correct_bets}")

    print("\nFixed Stake Betting Results:")
    print(f"Total amount bet: ${total_stake:.2f}")
    print(f"Total return: ${total_return:.2f}")
    print(f"Net profit: ${net_profit:.2f}")
    print(f"ROI (based on total amount bet): {roi:.2f}%")

    print(f"\nFractional Kelly Criterion Betting Results (fraction: {kelly_fraction:.3f}):")
    print(f"Initial bankroll: ${initial_bankroll:.2f}")
    print(f"Final bankroll: ${kelly_final_bankroll:.2f}")
    print(f"Total amount bet: ${kelly_total_stake:.2f}")
    print(f"Total return: ${kelly_total_return:.2f}")
    print(f"Net profit: ${kelly_net_profit:.2f}")
    print(f"ROI (based on initial bankroll): {kelly_roi:.2f}%")


def print_overall_metrics(y_test, y_pred, y_pred_proba):
    overall_accuracy = accuracy_score(y_test, y_pred)
    overall_precision = precision_score(y_test, y_pred)
    overall_recall = recall_score(y_test, y_pred)
    overall_f1 = f1_score(y_test, y_pred)
    overall_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

    print(f"\nOverall Model Metrics (all predictions):")
    print(f"Accuracy: {overall_accuracy:.4f}")
    print(f"Precision: {overall_precision:.4f}")
    print(f"Recall: {overall_recall:.4f}")
    print(f"F1 Score: {overall_f1:.4f}")
    print(f"AUC: {overall_auc:.4f}")


def main():
    CONFIDENCE_THRESHOLD = 0.50
    INITIAL_BANKROLL = 10000
    KELLY_FRACTION = 0.125

    model_path = os.path.abspath('models/xgboost/model_0.6587_338_features.json')
    print(f"Attempting to load model from: {model_path}")
    model = load_model(model_path)

    test_data = pd.read_csv('data/train test data/test_data.csv')

    y_test = test_data['winner']
    X_test = test_data.drop(['winner'], axis=1)

    X_test = preprocess_data(X_test)

    expected_features = model.get_booster().feature_names
    X_test = X_test.reindex(columns=expected_features)

    y_pred_proba = model.predict_proba(X_test)

    print(f"\nExample predictions and bets ({CONFIDENCE_THRESHOLD:.0%} confidence threshold):")
    (total_stake, total_return, correct_bets, total_bets, confident_predictions,
     correct_confident_predictions, kelly_total_stake, kelly_total_return, kelly_final_bankroll) = evaluate_bets(
        y_test, y_pred_proba, test_data, CONFIDENCE_THRESHOLD, INITIAL_BANKROLL, KELLY_FRACTION)

    print_betting_results(len(test_data), confident_predictions, correct_confident_predictions, total_bets,
                          correct_bets, total_stake, total_return, CONFIDENCE_THRESHOLD,
                          kelly_total_stake, kelly_total_return, kelly_final_bankroll, INITIAL_BANKROLL, KELLY_FRACTION)

    y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
    print_overall_metrics(y_test, y_pred, y_pred_proba)


if __name__ == "__main__":
    main()

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
    full_kelly = max(0, (p - (q / b)))  # Ensure non-negative fraction
    return full_kelly * kelly_fraction  # Apply fractional Kelly


def evaluate_bets(y_test, y_pred_proba, test_data, confidence_threshold, initial_bankroll=10000, kelly_fraction=0.125, fixed_bet_fraction=0.001):
    current_bankroll = initial_bankroll
    total_return = 0
    correct_bets = 0
    total_bets = 0
    confident_predictions = 0
    correct_confident_predictions = 0
    kelly_bankroll = initial_bankroll

    confident_bets = []
    processed_fights = set()

    for i in range(len(test_data)):
        fight_id = frozenset([test_data.iloc[i]['fighter'], test_data.iloc[i]['fighter_b']])

        if fight_id in processed_fights:
            continue

        processed_fights.add(fight_id)

        true_winner = "Fighter A" if y_test.iloc[i] == 1 else "Fighter B"
        winning_probability = max(y_pred_proba[i])
        predicted_winner = "Fighter A" if y_pred_proba[i][1] > y_pred_proba[i][0] else "Fighter B"

        if winning_probability >= confidence_threshold:
            confident_predictions += 1
            odds = test_data.iloc[i]['current_fight_open_odds'] if predicted_winner == "Fighter A" else \
                test_data.iloc[i]['current_fight_open_odds_b']

            # Compounding Fixed Fraction Betting
            stake = current_bankroll * fixed_bet_fraction
            profit = calculate_profit(odds, stake)

            # Fractional Kelly Criterion Betting
            if odds > 0:
                b = odds / 100
            else:
                b = 100 / abs(odds)
            kelly_bet_size = calculate_kelly_fraction(winning_probability, b, kelly_fraction)
            kelly_stake = kelly_bankroll * kelly_bet_size
            kelly_profit = calculate_profit(odds, kelly_stake)

            total_bets += 1

            bet_result = {
                'Fight': i + 1,
                'Fighter A': test_data.iloc[i]['fighter'],
                'Fighter B': test_data.iloc[i]['fighter_b'],
                'Date': test_data.iloc[i]['current_fight_date'],
                'True Winner': true_winner,
                'Predicted Winner': predicted_winner,
                'Confidence': f"{winning_probability:.2%}",
                'Odds': odds,
                'Fixed Fraction Stake': f"${stake:.2f}",
                'Fixed Fraction Potential Profit': f"${profit:.2f}",
                'Fractional Kelly Stake': f"${kelly_stake:.2f}",
                'Fractional Kelly Potential Profit': f"${kelly_profit:.2f}"
            }

            confident_bets.append(bet_result)

            if predicted_winner == true_winner:
                current_bankroll += profit
                total_return += stake + profit
                kelly_bankroll += kelly_profit
                correct_bets += 1
                correct_confident_predictions += 1
            else:
                current_bankroll -= stake
                kelly_bankroll -= kelly_stake

    # Print all confident bets
    for bet in confident_bets:
        print(f"Fight {bet['Fight']}: {bet['Fighter A']} vs {bet['Fighter B']} on {bet['Date']}")
        print(f"True Winner: {bet['True Winner']}")
        print(f"Predicted Winner: {bet['Predicted Winner']}")
        print(f"Confidence: {bet['Confidence']}")
        print(f"Odds: {bet['Odds']}")
        print(f"Fixed Fraction Stake: {bet['Fixed Fraction Stake']}")
        print(f"Fixed Fraction Potential profit: {bet['Fixed Fraction Potential Profit']}")
        print(f"Fractional Kelly Stake: {bet['Fractional Kelly Stake']}")
        print(f"Fractional Kelly Potential profit: {bet['Fractional Kelly Potential Profit']}")
        print("---")

    return (current_bankroll, total_return, correct_bets, total_bets, confident_predictions,
            correct_confident_predictions, kelly_bankroll)


def print_betting_results(total_fights, confident_predictions, correct_confident_predictions, total_bets, correct_bets,
                          initial_bankroll, final_bankroll, total_return, confidence_threshold,
                          kelly_final_bankroll, kelly_fraction, fixed_bet_fraction):
    confident_accuracy = correct_confident_predictions / confident_predictions if confident_predictions > 0 else 0
    net_profit = final_bankroll - initial_bankroll
    roi = (net_profit / initial_bankroll) * 100
    kelly_net_profit = kelly_final_bankroll - initial_bankroll
    kelly_roi = (kelly_net_profit / initial_bankroll) * 100

    print("________________________________________________________________________")
    print(f"\nConfident Prediction Accuracy (≥{confidence_threshold:.0%} confidence): {confident_accuracy:.2%}")
    print(f"\nBetting Results ({confidence_threshold:.0%} confidence threshold):")
    print(f"Total fights: {total_fights}")
    print(f"Fights predicted with ≥{confidence_threshold:.0%} confidence: {confident_predictions}")
    print(f"Correct predictions: {correct_confident_predictions}")
    print(f"Total bets: {total_bets}")
    print(f"Correct bets: {correct_bets}")

    print("\nCompounding Fixed Fraction Betting Results:")
    print(f"Initial bankroll: ${initial_bankroll:.2f}")
    print(f"Final bankroll: ${final_bankroll:.2f}")
    print(f"Total return: ${total_return:.2f}")
    print(f"Net profit: ${net_profit:.2f}")
    print(f"ROI (based on initial bankroll): {roi:.2f}%")
    print(f"Fixed bet fraction: {fixed_bet_fraction:.3f}")

    print(f"\nFractional Kelly Criterion Betting Results (fraction: {kelly_fraction:.3f}):")
    print(f"Initial bankroll: ${initial_bankroll:.2f}")
    print(f"Final bankroll: ${kelly_final_bankroll:.2f}")
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


if __name__ == "__main__":
    CONFIDENCE_THRESHOLD = 0.65
    INITIAL_BANKROLL = 10000
    KELLY_FRACTION = 1
    FIXED_BET_FRACTION = 0.1

    model_path = os.path.abspath('models/xgboost/model_0.7308_338_features.json')
    model = load_model(model_path)

    test_data = pd.read_csv('data/train test data/test_data.csv')

    y_test = test_data['winner']

    # Store the columns we want to drop for prediction but keep for display
    display_columns = ['current_fight_date', 'fighter', 'fighter_b']
    display_data = test_data[display_columns]

    # Drop the display columns and 'winner' for prediction
    X_test = test_data.drop(['winner'] + display_columns, axis=1)

    X_test = preprocess_data(X_test)

    expected_features = model.get_booster().feature_names
    X_test = X_test.reindex(columns=expected_features)

    y_pred_proba = model.predict_proba(X_test)

    # Add back the display columns for result printing
    test_data_with_display = pd.concat([X_test, display_data], axis=1)

    print(f"\nExample predictions and bets ({CONFIDENCE_THRESHOLD:.0%} confidence threshold):")
    (final_bankroll, total_return, correct_bets, total_bets, confident_predictions,
     correct_confident_predictions, kelly_final_bankroll) = evaluate_bets(
        y_test, y_pred_proba, test_data_with_display, CONFIDENCE_THRESHOLD, INITIAL_BANKROLL,
        KELLY_FRACTION, FIXED_BET_FRACTION)

    print_betting_results(len(test_data), confident_predictions, correct_confident_predictions, total_bets,
                          correct_bets, INITIAL_BANKROLL, final_bankroll, total_return, CONFIDENCE_THRESHOLD,
                          kelly_final_bankroll, KELLY_FRACTION, FIXED_BET_FRACTION)

    y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
    print_overall_metrics(y_test, y_pred, y_pred_proba)

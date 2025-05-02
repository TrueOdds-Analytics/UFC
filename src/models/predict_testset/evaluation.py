"""
Evaluation functions for MMA betting strategies
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from rich.console import Console
from rich.table import Table

from betting_utils import (
    calculate_profit, calculate_kelly_fraction, print_fight_results,
    calculate_average_odds
)

def evaluate_bets(y_test, y_pred_proba_list, test_data, confidence_threshold, initial_bankroll=10000,
                  kelly_fraction=0.125, fixed_bet_fraction=0.001, default_bet=0.00, min_odds=-300,
                  max_underdog_odds=300, print_fights=True, max_bet_percentage=0.20, use_ensemble=True,
                  odds_type='average'):
    """
    Evaluate betting performance while ensuring consistent predictions regardless of event order.
    This implementation selects the fighter with the highest confidence from the ensemble prediction.

    Args:
        y_test: True outcome values
        y_pred_proba_list: List of prediction probability arrays
        test_data: DataFrame with test data
        confidence_threshold: Minimum confidence to place a bet
        initial_bankroll: Starting bankroll amount
        kelly_fraction: Fraction of Kelly to use (0.5 = half Kelly)
        fixed_bet_fraction: Fraction of bankroll for fixed bets
        default_bet: Minimum bet size as fraction of bankroll
        min_odds: Minimum odds to place a bet (-300 = 1.33 decimal)
        max_underdog_odds: Maximum underdog odds to bet
        print_fights: Whether to print detailed fight results
        max_bet_percentage: Maximum percentage of bankroll to bet
        use_ensemble: Whether to use ensemble predictions
        odds_type: Which odds to use ('open', 'close', 'average')

    Returns:
        Tuple of betting performance metrics
    """
    # Store the original data and predictions by fight ID to ensure consistency
    fight_data = {}

    # First pass: build a mapping of fights to their predictions and outcomes
    for i in range(len(test_data)):
        row = test_data.iloc[i]
        # Create a unique fight identifier using date and fighter names
        fight_id = (row['current_fight_date'], frozenset([row['fighter_a'], row['fighter_b']]))

        # Calculate prediction for this fight
        if use_ensemble:
            # Still average the model predictions for the ensemble
            y_pred_proba_avg = np.mean([y_pred_proba[i] for y_pred_proba in y_pred_proba_list], axis=0)

            # Track how many models agree with the ensemble
            models_agreeing = sum([1 for y_pred_proba in y_pred_proba_list if
                                   (y_pred_proba[i][1] > y_pred_proba[i][0]) == (
                                           y_pred_proba_avg[1] > y_pred_proba_avg[0])])
        else:
            y_pred_proba_avg = y_pred_proba_list[0][i]
            models_agreeing = 1

        # Store all relevant data for this fight
        fight_data[fight_id] = {
            'row': row,
            'prediction': y_pred_proba_avg,
            'true_outcome': y_test.iloc[i],
            'models_agreeing': models_agreeing
        }

    # Initialize tracking variables
    fixed_bankroll = initial_bankroll
    kelly_bankroll = initial_bankroll
    fixed_total_volume = 0
    kelly_total_volume = 0
    fixed_correct_bets = 0
    kelly_correct_bets = 0
    fixed_total_bets = 0
    kelly_total_bets = 0
    confident_predictions = 0
    correct_confident_predictions = 0
    confident_bets = []

    # Sort fight dates for chronological processing
    all_dates = sorted(set(row['current_fight_date'] for row in [data['row'] for data in fight_data.values()]))

    # Initialize bankroll tracking dictionaries
    daily_fixed_bankrolls = {}
    daily_kelly_bankrolls = {}
    daily_fixed_profits = {}
    daily_kelly_profits = {}

    available_fixed_bankroll = fixed_bankroll
    available_kelly_bankroll = kelly_bankroll

    # Process fights chronologically by date
    for current_date in all_dates:
        # Reset daily profit tracking
        daily_fixed_profits[current_date] = 0
        daily_kelly_profits[current_date] = 0

        # Get all fights for the current date
        date_fights = [(fight_id, fight_info) for fight_id, fight_info in fight_data.items()
                       if fight_id[0] == current_date]

        # Process each fight on this date
        for fight_id, fight_info in date_fights:
            row = fight_info['row']
            y_pred_proba_avg = fight_info['prediction']
            true_outcome = fight_info['true_outcome']
            models_agreeing = fight_info['models_agreeing']

            # Determine predicted winner and true winner
            true_winner = row['fighter_a'] if true_outcome == 1 else row['fighter_b']

            # Find the fighter prediction with higher confidence
            if y_pred_proba_avg[0] > y_pred_proba_avg[1]:
                # Fighter B has higher probability
                predicted_winner = row['fighter_b']
                winning_probability = y_pred_proba_avg[0]
            else:
                # Fighter A has higher probability
                predicted_winner = row['fighter_a']
                winning_probability = y_pred_proba_avg[1]

            # Track confident predictions
            confident_predictions += 1
            if predicted_winner == true_winner:
                correct_confident_predictions += 1

            # Only place bets if confidence meets threshold and enough models agree
            if winning_probability >= confidence_threshold and models_agreeing >= (5 if use_ensemble else 1):
                # Get the appropriate odds based on prediction
                if predicted_winner == row['fighter_a']:
                    open_odds = row['current_fight_open_odds']
                    close_odds = row['current_fight_closing_odds']
                else:
                    open_odds = row['current_fight_open_odds_b']
                    close_odds = row['current_fight_closing_odds_b']

                # Determine which odds to use
                if odds_type == 'open':
                    odds = open_odds
                elif odds_type == 'close':
                    odds = close_odds
                else:  # 'average'
                    odds = calculate_average_odds(open_odds, close_odds)

                # Skip if odds are outside acceptable range
                if odds < min_odds or odds > max_underdog_odds:
                    continue

                # Track available bankroll before bet
                fixed_available_bankroll_before_bet = available_fixed_bankroll
                kelly_available_bankroll_before_bet = available_kelly_bankroll

                # Calculate bet sizes
                fixed_max_bet = fixed_available_bankroll_before_bet * max_bet_percentage
                kelly_max_bet = kelly_available_bankroll_before_bet * max_bet_percentage

                fixed_stake = min(fixed_available_bankroll_before_bet * fixed_bet_fraction,
                                  fixed_available_bankroll_before_bet, fixed_max_bet)

                # Kelly calculation
                b = odds / 100 if odds > 0 else 100 / abs(odds)
                full_kelly_fraction = calculate_kelly_fraction(winning_probability, b)
                adjusted_kelly_fraction = full_kelly_fraction * kelly_fraction
                kelly_stake = kelly_available_bankroll_before_bet * adjusted_kelly_fraction
                kelly_stake = min(kelly_stake, kelly_available_bankroll_before_bet, kelly_max_bet)

                # Use default bet if Kelly is too small
                if kelly_stake <= kelly_available_bankroll_before_bet * default_bet:
                    kelly_stake = min(kelly_available_bankroll_before_bet * default_bet,
                                      kelly_available_bankroll_before_bet, kelly_max_bet)

                # Store bet information
                bet_result = {
                    'Fight': confident_predictions,
                    'Fighter A': row['fighter_a'],
                    'Fighter B': row['fighter_b'],
                    'Date': current_date,
                    'True Winner': true_winner,
                    'Predicted Winner': predicted_winner,
                    'Confidence': f"{winning_probability:.2%}",
                    'Odds': odds,
                    'Models Agreeing': models_agreeing
                }

                # Process fixed stake bet
                if fixed_stake > 0:
                    fixed_total_bets += 1
                    available_fixed_bankroll -= fixed_stake
                    fixed_profit = calculate_profit(odds, fixed_stake)
                    fixed_total_volume += fixed_stake

                    bet_result.update({
                        'Fixed Fraction Starting Bankroll': f"${fixed_bankroll:.2f}",
                        'Fixed Fraction Available Bankroll': f"${fixed_available_bankroll_before_bet:.2f}",
                        'Fixed Fraction Stake': f"${fixed_stake:.2f}",
                        'Fixed Fraction Potential Profit': f"${fixed_profit:.2f}",
                    })

                    if predicted_winner == true_winner:
                        daily_fixed_profits[current_date] += fixed_profit
                        fixed_correct_bets += 1
                        bet_result['Fixed Fraction Profit'] = fixed_profit
                    else:
                        daily_fixed_profits[current_date] -= fixed_stake
                        bet_result['Fixed Fraction Profit'] = -fixed_stake

                    bet_result[
                        'Fixed Fraction Bankroll After'] = f"${(fixed_bankroll + daily_fixed_profits[current_date]):.2f}"
                    bet_result['Fixed Fraction ROI'] = (bet_result[
                                                            'Fixed Fraction Profit'] / fixed_available_bankroll_before_bet) * 100

                # Process Kelly bet
                if kelly_stake > 0:
                    kelly_total_bets += 1
                    available_kelly_bankroll -= kelly_stake
                    kelly_profit = calculate_profit(odds, kelly_stake)
                    kelly_total_volume += kelly_stake

                    bet_result.update({
                        'Kelly Starting Bankroll': f"${kelly_bankroll:.2f}",
                        'Kelly Available Bankroll': f"${kelly_available_bankroll_before_bet:.2f}",
                        'Kelly Stake': f"${kelly_stake:.2f}",
                        'Kelly Potential Profit': f"${kelly_profit:.2f}",
                    })

                    if predicted_winner == true_winner:
                        daily_kelly_profits[current_date] += kelly_profit
                        kelly_correct_bets += 1
                        bet_result['Kelly Profit'] = kelly_profit
                    else:
                        daily_kelly_profits[current_date] -= kelly_stake
                        bet_result['Kelly Profit'] = -kelly_stake

                    bet_result['Kelly Bankroll After'] = f"${(kelly_bankroll + daily_kelly_profits[current_date]):.2f}"
                    bet_result['Kelly ROI'] = (bet_result['Kelly Profit'] / kelly_available_bankroll_before_bet) * 100

                confident_bets.append(bet_result)

        # Update bankroll at the end of the day
        daily_fixed_bankrolls[current_date] = fixed_bankroll + daily_fixed_profits[current_date]
        daily_kelly_bankrolls[current_date] = kelly_bankroll + daily_kelly_profits[current_date]

        # Update running bankroll for next day
        fixed_bankroll = daily_fixed_bankrolls[current_date]
        kelly_bankroll = daily_kelly_bankrolls[current_date]
        available_fixed_bankroll = fixed_bankroll
        available_kelly_bankroll = kelly_bankroll

    # Print fight results if requested
    if print_fights:
        print_fight_results(confident_bets)

    return (fixed_bankroll, fixed_total_volume, fixed_correct_bets, fixed_total_bets,
            kelly_bankroll, kelly_total_volume, kelly_correct_bets, kelly_total_bets,
            confident_predictions, correct_confident_predictions,
            daily_fixed_bankrolls, daily_kelly_bankrolls)


def print_overall_metrics(y_test, y_pred, y_pred_proba):
    """
    Print overall model performance metrics

    Args:
        y_test: True outcome values
        y_pred: Predicted class values
        y_pred_proba: Prediction probabilities
    """
    console = Console()
    table = Table(title="Overall Model Metrics (all predictions)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")

    try:
        overall_accuracy = accuracy_score(y_test, y_pred)
        table.add_row("Accuracy", f"{overall_accuracy:.4f}")
    except Exception as e:
        table.add_row("Accuracy", "Not available")
        print(f"Warning: Could not calculate accuracy - {str(e)}")

    try:
        overall_precision = precision_score(y_test, y_pred)
        table.add_row("Precision", f"{overall_precision:.4f}")
    except Exception as e:
        table.add_row("Precision", "Not available")
        print(f"Warning: Could not calculate precision - {str(e)}")

    try:
        overall_recall = recall_score(y_test, y_pred)
        table.add_row("Recall", f"{overall_recall:.4f}")
    except Exception as e:
        table.add_row("Recall", "Not available")
        print(f"Warning: Could not calculate recall - {str(e)}")

    try:
        overall_f1 = f1_score(y_test, y_pred)
        table.add_row("F1 Score", f"{overall_f1:.4f}")
    except Exception as e:
        table.add_row("F1 Score", "Not available")
        print(f"Warning: Could not calculate F1 score - {str(e)}")

    try:
        # Check if both classes are present before calculating AUC
        if len(np.unique(y_test)) > 1:
            overall_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            table.add_row("AUC", f"{overall_auc:.4f}")
        else:
            table.add_row("AUC", "Not available (only one class in test data)")
            print("Warning: ROC AUC score is not defined when only one class is present in y_true")
    except Exception as e:
        table.add_row("AUC", "Not available")
        print(f"Warning: Could not calculate AUC - {str(e)}")

    console.print(table)

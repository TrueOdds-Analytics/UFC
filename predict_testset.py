import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import numpy as np
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich.console import Group, Console
from rich.panel import Panel
import datetime
from io import StringIO
import sys
from sklearn.calibration import CalibratedClassifierCV
import pickle
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt


class CategoryEncoder:
    """Ensures consistent categorical encoding across different datasets"""

    def __init__(self):
        self.category_mappings = {}
        self.initialized = False

    def fit(self, data):
        """Learn category mappings from reference data"""
        category_columns = [col for col in data.columns if col.endswith(('fight_1', 'fight_2', 'fight_3'))]

        for col in category_columns:
            # Create mapping from unique values to integer codes
            unique_values = data[col].dropna().unique()
            self.category_mappings[col] = {val: i for i, val in enumerate(unique_values)}

        self.initialized = True
        return self

    def transform(self, data):
        """Apply consistent categorical mappings"""
        if not self.initialized:
            raise ValueError("CategoryEncoder must be fit before transform")

        # Create a copy to avoid modifying the original
        data_copy = data.copy()

        # Process each categorical column
        for col, mapping in self.category_mappings.items():
            if col in data_copy.columns:
                # Apply mapping, with -1 for unknown values
                data_copy[col] = data_copy[col].map(mapping).fillna(-1).astype('int32')

        return data_copy

    def fit_transform(self, data):
        """Fit and transform in one step"""
        self.fit(data)
        return self.transform(data)

    def save(self, filepath):
        """Save encoder to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.category_mappings, f)

    @classmethod
    def load(cls, filepath):
        """Load encoder from disk"""
        encoder = cls()
        with open(filepath, 'rb') as f:
            encoder.category_mappings = pickle.load(f)
        encoder.initialized = True
        return encoder


def preprocess_data(data, encoder=None, fit=False):
    """
    Preprocess data with consistent categorical encodings

    Args:
        data: DataFrame to preprocess
        encoder: Optional CategoryEncoder instance
        fit: Whether to fit the encoder on this data

    Returns:
        Preprocessed DataFrame and encoder
    """
    # Create a new encoder if none provided
    if encoder is None:
        encoder = CategoryEncoder()

    # Apply categorical encoding
    if fit:
        # Fit and transform (for training/validation data)
        processed_data = encoder.fit_transform(data)
    else:
        # Only transform (for test/prediction data)
        processed_data = encoder.transform(data)

    return processed_data, encoder


def load_model(model_path, model_type='xgboost'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(enable_categorical=True)
            model.load_model(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def calculate_profit(odds, stake):
    if odds < 0:
        return (100 / abs(odds)) * stake
    else:
        return (odds / 100) * stake


def calculate_kelly_fraction(p, b):
    q = 1 - p
    return max(0, (p - (q / b)))


def print_fight_results(confident_bets):
    console = Console(width=160)
    for bet in confident_bets:
        fighter_a = bet['Fighter A'].title()
        fighter_b = bet['Fighter B'].title()
        date_obj = datetime.datetime.strptime(bet['Date'], '%Y-%m-%d')
        formatted_date = date_obj.strftime('%B %d, %Y')

        fixed_available_bankroll_before_bet = float(bet.get('Fixed Fraction Available Bankroll', '0').replace('$', ''))
        fixed_stake = float(bet.get('Fixed Fraction Stake', '0').replace('$', ''))
        fixed_stake_percentage = (
                                         fixed_stake / fixed_available_bankroll_before_bet) * 100 if fixed_available_bankroll_before_bet > 0 else 0

        fixed_panel = Panel(
            f"Starting Bankroll: {bet.get('Fixed Fraction Starting Bankroll', 'N/A')}\n"
            f"Available Bankroll: {bet.get('Fixed Fraction Available Bankroll', 'N/A')}\n"
            f"Stake: {bet.get('Fixed Fraction Stake', 'N/A')}\n"
            f"Stake Percentage: {fixed_stake_percentage:.2f}%\n"
            f"Potential Profit: {bet.get('Fixed Fraction Potential Profit', 'N/A')}\n"
            f"Bankroll After Bet: {bet.get('Fixed Fraction Bankroll After', 'N/A')}\n"
            f"Profit: ${bet.get('Fixed Fraction Profit', 0):.2f}\n"
            f"ROI (of available bankroll): {bet.get('Fixed Fraction ROI', 0):.2f}%",
            title="Fixed Fraction",
            expand=True,
            width=42
        )

        kelly_available_bankroll_before_bet = float(bet.get('Kelly Available Bankroll', '0').replace('$', ''))
        kelly_stake = float(bet.get('Kelly Stake', '0').replace('$', ''))
        kelly_stake_percentage = (
                                         kelly_stake / kelly_available_bankroll_before_bet) * 100 if kelly_available_bankroll_before_bet > 0 else 0

        kelly_panel = Panel(
            f"Starting Bankroll: {bet.get('Kelly Starting Bankroll', 'N/A')}\n"
            f"Available Bankroll: {bet.get('Kelly Available Bankroll', 'N/A')}\n"
            f"Stake: {bet.get('Kelly Stake', 'N/A')}\n"
            f"Stake Percentage: {kelly_stake_percentage:.2f}%\n"
            f"Potential Profit: {bet.get('Kelly Potential Profit', 'N/A')}\n"
            f"Bankroll After Bet: {bet.get('Kelly Bankroll After', 'N/A')}\n"
            f"Profit: ${bet.get('Kelly Profit', 0):.2f}\n"
            f"ROI (of available bankroll): {bet.get('Kelly ROI', 0):.2f}%",
            title="Kelly",
            expand=True,
            width=42
        )

        fight_info = Group(
            Text(f"True Winner: {bet['True Winner'].title()}", style="green"),
            Text(f"Predicted Winner: {bet['Predicted Winner'].title()}", style="blue"),
            Text(f"Confidence: {bet['Confidence']}", style="yellow"),
            Text(f"Models Agreeing: {bet['Models Agreeing']}/5", style="cyan")
        )

        main_panel = Panel(
            Group(
                Panel(fight_info, title="Fight Information"),
                Columns([fixed_panel, kelly_panel], equal=False, expand=False, align="left")
            ),
            title=f"Fight {bet['Fight']}: {fighter_a} vs {fighter_b} on {formatted_date}",
            subtitle=f"Odds: {bet['Odds']}",
            width=89
        )

        console.print(main_panel, style="magenta")
        console.print()


def calculate_average_odds(open_odds, close_odds):
    def american_to_decimal(odds):
        return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1

    avg_decimal = (american_to_decimal(open_odds) + american_to_decimal(close_odds)) / 2

    if avg_decimal > 2:
        return round((avg_decimal - 1) * 100)
    else:
        return round(-100 / (avg_decimal - 1))


# Modified parts of the code to use highest confidence instead of averaging

# Modified parts of the code to use highest confidence fighter selection
# while maintaining ensemble averaging

def evaluate_bets(y_test, y_pred_proba_list, test_data, confidence_threshold, initial_bankroll=10000,
                  kelly_fraction=0.125, fixed_bet_fraction=0.001, default_bet=0.00, min_odds=-300,
                  max_underdog_odds=200, print_fights=True, max_bet_percentage=0.20, use_ensemble=True,
                  odds_type='average'):
    """
    Evaluate betting performance while ensuring consistent predictions regardless of event order.
    This implementation selects the fighter with the highest confidence from the ensemble prediction.
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

            # MODIFIED: Find the fighter prediction with higher confidence
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


def calculate_daily_roi(daily_bankrolls, initial_bankroll):
    daily_roi = {}
    previous_bankroll = initial_bankroll

    for date, bankroll in sorted(daily_bankrolls.items()):
        daily_profit = bankroll - previous_bankroll
        daily_roi[date] = (daily_profit / previous_bankroll) * 100
        previous_bankroll = bankroll

    return daily_roi


def print_daily_roi(daily_fixed_roi, daily_kelly_roi):
    console = Console()
    console.print("\nDaily ROI:")
    table = Table(title="Daily Return on Investment")
    table.add_column("Date", style="cyan")
    table.add_column("Fixed Fraction ROI", justify="right", style="magenta")
    table.add_column("Kelly ROI", justify="right", style="green")

    for date in sorted(daily_fixed_roi.keys()):
        fixed_roi = f"{daily_fixed_roi[date]:.2f}%"
        kelly_roi = f"{daily_kelly_roi[date]:.2f}%"
        table.add_row(date, fixed_roi, kelly_roi)

    console.print(table)


def calculate_monthly_roi(daily_bankrolls, initial_bankroll, kelly):
    monthly_roi = {}
    monthly_profit = {}
    current_month = None
    current_bankroll = initial_bankroll
    month_start_bankroll = initial_bankroll
    total_profit = 0
    if kelly:
        print("\nDetailed Kelly ROI Calculation:")
    else:
        print("\nDetailed Fixed Fraction ROI Calculation:")
    print(f"{'Month':<10}{'Profit':<15}{'ROI':<10}{'Start Bankroll':<20}{'End Bankroll':<20}")
    print("-" * 80)

    sorted_dates = sorted(daily_bankrolls.keys())
    for date in sorted_dates:
        bankroll = daily_bankrolls[date]
        month = date[:7]  # Extract YYYY-MM

        if month != current_month:
            if current_month is not None:
                profit = current_bankroll - month_start_bankroll
                monthly_profit[current_month] = profit
                total_profit += profit
                roi = (profit / initial_bankroll) * 100  # ROI based on initial bankroll
                monthly_roi[current_month] = roi
                print(
                    f"{current_month:<10}${profit:<14.2f}{roi:<10.2f}${month_start_bankroll:<19.2f}${current_bankroll:<19.2f}")

            current_month = month
            month_start_bankroll = current_bankroll

        current_bankroll = bankroll

    # Handle the last month
    if current_month is not None:
        profit = current_bankroll - month_start_bankroll
        monthly_profit[current_month] = profit
        total_profit += profit
        roi = (profit / initial_bankroll) * 100  # ROI based on initial bankroll
        monthly_roi[current_month] = roi
        print(
            f"{current_month:<10}${profit:<14.2f}{roi:<10.2f}${month_start_bankroll:<19.2f}${current_bankroll:<19.2f}")

    total_roi = (total_profit / initial_bankroll) * 100
    sum_monthly_roi = sum(monthly_roi.values())

    print("-" * 80)
    print(f"{'Total':<10}${total_profit:<14.2f}{total_roi:<10.2f}")
    print(f"\nSum of monthly ROIs: {sum_monthly_roi:.2f}%")
    print(f"Total ROI: {total_roi:.2f}%")
    print(f"Difference: {total_roi - sum_monthly_roi:.2f}%")

    print("\nDebug Information:")
    print(f"Number of events in dataset: {len(sorted_dates)}")
    print(f"First date: {sorted_dates[0]}, Last date: {sorted_dates[-1]}")
    print(f"Initial bankroll: ${initial_bankroll:.2f}, Final bankroll: ${current_bankroll:.2f}")

    return monthly_roi, monthly_profit, total_roi


def print_betting_results(total_fights, confident_predictions, correct_confident_predictions,
                          fixed_total_bets, fixed_correct_bets, initial_bankroll, fixed_final_bankroll,
                          fixed_total_volume, confidence_threshold, kelly_final_bankroll, kelly_total_volume,
                          kelly_correct_bets, kelly_total_bets, kelly_fraction, fixed_bet_fraction,
                          earliest_fight_date, fixed_monthly_profit, kelly_monthly_profit):
    confident_accuracy = correct_confident_predictions / confident_predictions if confident_predictions > 0 else 0
    fixed_accuracy = fixed_correct_bets / fixed_total_bets if fixed_total_bets > 0 else 0
    kelly_accuracy = kelly_correct_bets / kelly_total_bets if kelly_total_bets > 0 else 0

    fixed_net_profit = sum(fixed_monthly_profit.values())
    fixed_roi = (fixed_net_profit / initial_bankroll) * 100

    kelly_net_profit = sum(kelly_monthly_profit.values())
    kelly_roi = (kelly_net_profit / initial_bankroll) * 100

    avg_fixed_bet_size = fixed_total_volume / fixed_total_bets if fixed_total_bets > 0 else 0
    avg_kelly_bet_size = kelly_total_volume / kelly_total_bets if kelly_total_bets > 0 else 0

    fixed_scale = (avg_fixed_bet_size / fixed_net_profit) * 100 if fixed_net_profit != 0 else 0
    kelly_scale = (avg_kelly_bet_size / kelly_net_profit) * 100 if kelly_net_profit != 0 else 0

    earliest_date = datetime.datetime.strptime(earliest_fight_date, '%Y-%m-%d')
    today = datetime.datetime.now()
    months_diff = (today.year - earliest_date.year) * 12 + today.month - earliest_date.month

    console = Console()

    console.print(Panel(f"Best confidence threshold: {confidence_threshold:.4f}\n"
                        f"Best Kelly ROI: {kelly_roi:.2f}%",
                        title="Optimal Parameters"))

    table = Table(title=f"Betting Results ({confidence_threshold:.0%} confidence threshold)")
    table.add_column("Metric", style="cyan")
    table.add_column("Fixed Fraction", justify="right", style="magenta")
    table.add_column("Kelly", justify="right", style="green")

    table.add_row("Total fights", str(total_fights), str(total_fights))
    table.add_row("Confident predictions", str(confident_predictions), str(confident_predictions))
    table.add_row("Correct predictions", str(correct_confident_predictions), str(correct_confident_predictions))
    table.add_row("Total bets", str(fixed_total_bets), str(kelly_total_bets))
    table.add_row("Correct bets", str(fixed_correct_bets), str(kelly_correct_bets))
    table.add_row("Betting Accuracy", f"{fixed_accuracy:.2%}", f"{kelly_accuracy:.2%}")
    table.add_row("Confident Prediction Accuracy", f"{confident_accuracy:.2%}", f"{confident_accuracy:.2%}")

    console.print(table)

    fixed_panel = Panel(
        f"Initial bankroll: ${initial_bankroll:.2f}\n"
        f"Final bankroll: ${fixed_final_bankroll:.2f}\n"
        f"Total volume: ${fixed_total_volume:.2f}\n"
        f"Net profit: ${fixed_net_profit:.2f}\n"
        f"ROI: {fixed_roi:.2f}%\n"
        f"Fixed bet fraction: {fixed_bet_fraction:.3f}\n"
        f"Average bet size: ${avg_fixed_bet_size:.2f}\n"
        f"Risk: {fixed_scale:.2f}%",
        title="Fixed Fraction Betting Results"
    )

    kelly_panel = Panel(
        f"Initial bankroll: ${initial_bankroll:.2f}\n"
        f"Final bankroll: ${kelly_final_bankroll:.2f}\n"
        f"Total volume: ${kelly_total_volume:.2f}\n"
        f"Net profit: ${kelly_net_profit:.2f}\n"
        f"ROI: {kelly_roi:.2f}%\n"
        f"Kelly fraction: {kelly_fraction:.3f}\n"
        f"Average bet size: ${avg_kelly_bet_size:.2f}\n"
        f"Risk: {kelly_scale:.2f}%",
        title="Kelly Criterion Betting Results"
    )

    console.print(Columns([fixed_panel, kelly_panel]))


def print_overall_metrics(y_test, y_pred, y_pred_proba):
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


def create_and_save_calibration_curves(y_test, y_pred_proba_list, use_ensemble=True, output_dir='calibration_plots',
                                       calibration_type='uncalibrated', model_names=None):
    """
    Create and save calibration curve plots for the model(s)

    Args:
        y_test: True labels
        y_pred_proba_list: List of prediction probabilities from models
        use_ensemble: Whether to use ensemble predictions
        output_dir: Directory to save the plot files
        calibration_type: Type of calibration ('uncalibrated', 'isotonic', 'sigmoid')
        model_names: List of model names for plots

    Returns:
        Dictionary mapping plot types to file paths
    """
    # Create the base output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectory for this calibration type
    calibration_dir = os.path.join(output_dir, calibration_type)
    os.makedirs(calibration_dir, exist_ok=True)

    plot_files = {}

    # Use default model names if none provided
    if model_names is None:
        model_names = [f"Model_{i + 1}" for i in range(len(y_pred_proba_list))]

    # First, create individual model plots if using ensemble
    if use_ensemble and len(y_pred_proba_list) > 1:
        plt.figure(figsize=(10, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

        for i, y_pred_proba in enumerate(y_pred_proba_list):
            prob_true, prob_pred = calibration_curve(y_test, y_pred_proba[:, 1], n_bins=10)
            plt.plot(prob_pred, prob_true, 's-', label=f'{model_names[i]}')

        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title(f'Individual Models Calibration Curves ({calibration_type.capitalize()})')
        plt.legend(loc='best')
        plt.grid(True)

        filename = os.path.join(calibration_dir, f'individual_models_calibration_{calibration_type}.png')
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()
        plot_files['individual_models'] = filename

    # Now create the ensemble or single model plot
    plt.figure(figsize=(12, 8))
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfectly calibrated')

    if use_ensemble:
        # Average predictions across all models
        y_pred_proba_avg = np.mean([y_pred_proba for y_pred_proba in y_pred_proba_list], axis=0)
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba_avg[:, 1], n_bins=10)
        plt.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8, label='Ensemble model')

        # Add annotations for each point
        for i, (x, y) in enumerate(zip(prob_pred, prob_true)):
            plt.annotate(f'({x:.2f}, {y:.2f})',
                         (x, y),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         fontsize=9)

        title = f'Ensemble Model Calibration Curve ({calibration_type.capitalize()})'
        filename = os.path.join(calibration_dir, f'ensemble_calibration_curve_{calibration_type}.png')
    else:
        # Single model
        y_pred_proba = y_pred_proba_list[0]
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba[:, 1], n_bins=10)
        plt.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8, label=model_names[0])

        # Add annotations for each point
        for i, (x, y) in enumerate(zip(prob_pred, prob_true)):
            plt.annotate(f'({x:.2f}, {y:.2f})',
                         (x, y),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         fontsize=9)

        title = f'{model_names[0]} Calibration Curve ({calibration_type.capitalize()})'
        filename = os.path.join(calibration_dir, f'{model_names[0]}_calibration_curve_{calibration_type}.png')

    plt.xlabel('Mean predicted probability', fontsize=12)
    plt.ylabel('Fraction of positives (true probability)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True)

    # Add calibration interpretation text box
    textstr = 'Interpretation:\n' + \
              'Points above the diagonal: Model is under-confident\n' + \
              'Points below the diagonal: Model is over-confident\n' + \
              'Points on the diagonal: Perfect calibration'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gcf().text(0.5, 0.02, textstr, fontsize=12, bbox=props,
                   ha='center', va='center')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for text box
    plt.savefig(filename, dpi=100)
    plt.close()
    plot_files['main_model'] = filename

    return plot_files


def create_reliability_diagram(y_test, y_pred_proba_list, use_ensemble=True, output_dir='calibration_plots',
                               calibration_type='uncalibrated', model_names=None):
    """
    Create a reliability diagram with histogram showing distribution of predictions

    Args:
        y_test: True labels
        y_pred_proba_list: List of prediction probabilities from models
        use_ensemble: Whether to use ensemble predictions
        output_dir: Directory to save the plot file
        calibration_type: Type of calibration ('uncalibrated', 'isotonic', 'sigmoid')
        model_names: List of model names for plots

    Returns:
        Path to the saved plot file
    """
    # Create the base output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectory for this calibration type
    calibration_dir = os.path.join(output_dir, calibration_type)
    os.makedirs(calibration_dir, exist_ok=True)

    # Use default model names if none provided
    if model_names is None:
        model_names = [f"Model_{i + 1}" for i in range(len(y_pred_proba_list))]

    # Get prediction probabilities
    if use_ensemble:
        y_pred_proba = np.mean([proba for proba in y_pred_proba_list], axis=0)
        title = f"Ensemble Model Reliability Diagram ({calibration_type.capitalize()})"
        filename = os.path.join(calibration_dir, f'ensemble_reliability_diagram_{calibration_type}.png')
    else:
        y_pred_proba = y_pred_proba_list[0]
        title = f"{model_names[0]} Reliability Diagram ({calibration_type.capitalize()})"
        filename = os.path.join(calibration_dir, f'{model_names[0]}_reliability_diagram_{calibration_type}.png')

    # Extract positive class probabilities
    y_pred_prob_pos = y_pred_proba[:, 1]

    # Create figure with two subplots: histogram and calibration curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10),
                                   gridspec_kw={'height_ratios': [1, 3]})

    # Top plot: histogram of prediction distribution
    ax1.hist(y_pred_prob_pos, bins=20, range=(0, 1), histtype='step',
             lw=2, color='blue', density=True)
    ax1.set_ylabel('Density')
    ax1.set_xlim([0, 1])
    ax1.set_title('Distribution of Predicted Probabilities')

    # Bottom plot: reliability diagram
    prob_true, prob_pred = calibration_curve(y_test, y_pred_prob_pos, n_bins=10)

    ax2.plot(prob_pred, prob_true, 's-', label='Calibration curve', color='blue', linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

    # Add calibration gap areas
    for i in range(len(prob_pred)):
        if prob_true[i] > prob_pred[i]:  # Under-confident
            ax2.fill_between([prob_pred[i], prob_pred[i]], [prob_pred[i], prob_true[i]],
                             alpha=0.2, color='green')
        elif prob_true[i] < prob_pred[i]:  # Over-confident
            ax2.fill_between([prob_pred[i], prob_pred[i]], [prob_pred[i], prob_true[i]],
                             alpha=0.2, color='red')

    # Annotate points
    for i, (x, y) in enumerate(zip(prob_pred, prob_true)):
        ax2.annotate(f'({x:.2f}, {y:.2f})', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center')

    ax2.set_xlabel('Mean predicted probability')
    ax2.set_ylabel('Fraction of positives (true probability)')
    ax2.set_title('Reliability Diagram (Calibration Curve)')
    ax2.legend(loc='best')
    ax2.grid(True)

    # Add interpretation text box
    textstr = 'Interpretation:\n' + \
              'Green areas: Model is under-confident (true probability > predicted)\n' + \
              'Red areas: Model is over-confident (predicted > true probability)\n' + \
              'For Kelly betting: Under-confidence leads to smaller bets than optimal\n' + \
              'Over-confidence leads to larger bets than optimal'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.5, 0.02, textstr, fontsize=10, bbox=props,
             ha='center', va='center')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for text box
    plt.savefig(filename, dpi=100)
    plt.close()

    return filename


def main(manual_threshold, use_calibration=True,
         initial_bankroll=10000, kelly_fraction=1.0, fixed_bet_fraction=0.1,
         max_bet_percentage=0.25, min_odds=-300, use_ensemble=True, odds_type='average'):
    # Redirect stdout to capture output
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    # Set constants
    INITIAL_BANKROLL = initial_bankroll
    KELLY_FRACTION = kelly_fraction
    FIXED_BET_FRACTION = fixed_bet_fraction
    MAX_BET_PERCENTAGE = max_bet_percentage

    # Load and preprocess data
    # Load data
    val_data = pd.read_csv('data/train test data/val_data.csv')
    test_data = pd.read_csv('data/train test data/test_data.csv')

    # Define columns to display
    display_columns = ['current_fight_date', 'fighter_a', 'fighter_b']

    # Separate target variables
    y_val, y_test = val_data['winner'], test_data['winner']

    # Initialize our category encoder
    encoder = CategoryEncoder()

    # First fit the encoder on validation data to learn consistent categorical mappings
    X_val, encoder = preprocess_data(
        val_data.drop(['winner'] + display_columns, axis=1),
        encoder=encoder,
        fit=True  # Important: fit the encoder on validation data
    )

    # Now use the same encoder to transform test data (without fitting again)
    X_test, _ = preprocess_data(
        test_data.drop(['winner'] + display_columns, axis=1),
        encoder=encoder,
        fit=False  # Don't fit on test data, just apply the mappings
    )

    # Ensure the encoder directory exists
    os.makedirs('models/encoders', exist_ok=True)

    # Save the encoder for future use
    encoder.save('models/encoders/category_encoder.pkl')

    # Concatenate features with display columns
    test_data_with_display = pd.concat([X_test, test_data[display_columns], y_test], axis=1)

    # Shuffle the test data
    test_data_with_display = test_data_with_display.sample(frac=1, random_state=42).reset_index(drop=True)

    # Separate shuffled data into features and target variable
    X_test = test_data_with_display.drop(display_columns + ['winner'], axis=1)
    y_test = test_data_with_display['winner']

    model_files = [
        'model_0.6940_auc_diff_0.0094.json',
        'model_0.7071_auc_diff_0.0073.json',
        'model_0.7071_auc_diff_0.0076.json',
        'model_0.7071_auc_diff_0.0078.json',
        'model_0.7071_auc_diff_0.0089.json'
    ]

    # Extract model names for use in plots
    model_names = [os.path.splitext(model_file)[0] for model_file in model_files]

    models = []
    calibrated_models = []

    if use_ensemble:
        for model_file in model_files:
            model_path = os.path.abspath(f'models/xgboost/jan2024-dec2025/dynamicmatchup sorted/{model_file}')
            models.append(load_model(model_path, 'xgboost'))
    else:
        model_path = os.path.abspath(f'models/xgboost/jan2024-dec2025/dynamicmatchup sorted/{model_files[0]}')
        models.append(load_model(model_path, 'xgboost'))
        model_names = [model_names[0]]  # Keep only the first model name if not using ensemble

    # Ensure consistent feature ordering
    expected_features = models[0].get_booster().feature_names
    X_val = X_val.reindex(columns=expected_features)
    X_test = X_test.reindex(columns=expected_features)

    # Calibrate models once using the validation data
    calibration_type = 'uncalibrated'  # Default
    if use_calibration:
        calibration_type = 'isotonic'  # Using isotonic calibration
        for model in models:
            calibrated_model = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
            calibrated_model.fit(X_val, y_val)
            calibrated_models.append(calibrated_model)
        # Use calibrated models for predictions
        models_to_use = calibrated_models
    else:
        # Use uncalibrated models
        models_to_use = models

    # Generate predictions row by row to ensure determinism
    y_pred_proba_list = []

    # Initialize a prediction list for each model
    for _ in range(len(models_to_use)):
        y_pred_proba_list.append([])

    # Process each row individually
    for i in range(len(X_test)):
        # Extract a single row as a DataFrame (maintaining 2D structure)
        row_features = X_test.iloc[[i]]

        # Get predictions from each model for this row
        for j, model in enumerate(models_to_use):
            y_pred_proba = model.predict_proba(row_features)
            y_pred_proba_list[j].append(y_pred_proba[0])  # Store just the probabilities

    # Convert lists to numpy arrays
    for j in range(len(y_pred_proba_list)):
        y_pred_proba_list[j] = np.array(y_pred_proba_list[j])

    # Check if there are enough samples for evaluation
    if len(test_data) == 0:
        console = Console()
        console.print("[bold red]Error: No test data available for evaluation[/bold red]")
        sys.stdout = old_stdout
        return

    # Evaluate bets
    bet_results = evaluate_bets(
        y_test, y_pred_proba_list, test_data_with_display, manual_threshold,
        INITIAL_BANKROLL, KELLY_FRACTION, FIXED_BET_FRACTION,
        default_bet=0.00, print_fights=True, max_bet_percentage=MAX_BET_PERCENTAGE,
        min_odds=min_odds, use_ensemble=use_ensemble, odds_type=odds_type, max_underdog_odds=200
    )

    (fixed_final_bankroll, fixed_total_volume, fixed_correct_bets, fixed_total_bets,
     kelly_final_bankroll, kelly_total_volume, kelly_correct_bets, kelly_total_bets,
     confident_predictions, correct_confident_predictions,
     daily_fixed_bankrolls, daily_kelly_bankrolls) = bet_results

    # Calculate ROI if we have daily results
    if daily_fixed_bankrolls and daily_kelly_bankrolls:
        earliest_fight_date = test_data['current_fight_date'].min()
        daily_fixed_roi = calculate_daily_roi(daily_fixed_bankrolls, INITIAL_BANKROLL)
        daily_kelly_roi = calculate_daily_roi(daily_kelly_bankrolls, INITIAL_BANKROLL)

        if daily_fixed_roi and daily_kelly_roi:
            print_daily_roi(daily_fixed_roi, daily_kelly_roi)

        fixed_monthly_roi, fixed_monthly_profit, fixed_total_roi = calculate_monthly_roi(
            daily_fixed_bankrolls, INITIAL_BANKROLL, False
        )
        kelly_monthly_roi, kelly_monthly_profit, kelly_total_roi = calculate_monthly_roi(
            daily_kelly_bankrolls, INITIAL_BANKROLL, True
        )

        # Print monthly results if available
        if fixed_monthly_roi and kelly_monthly_roi:
            console = Console()
            console.print("\nMonthly ROI (based on monthly performance, calibrated):")
            table = Table()
            table.add_column("Month", style="cyan")
            table.add_column("Fixed Fraction ROI", justify="right", style="magenta")
            table.add_column("Kelly ROI", justify="right", style="green")

            for month in sorted(fixed_monthly_roi.keys()):
                fixed_monthly = f"{fixed_monthly_roi[month]:.2f}%"
                kelly_monthly = f"{kelly_monthly_roi[month]:.2f}%"
                table.add_row(month, fixed_monthly, kelly_monthly)

            table.add_row("Total", f"{fixed_total_roi:.2f}%", f"{kelly_total_roi:.2f}%")
            console.print(table)

        print_betting_results(
            len(test_data), confident_predictions, correct_confident_predictions,
            fixed_total_bets, fixed_correct_bets, INITIAL_BANKROLL, fixed_final_bankroll,
            fixed_total_volume, manual_threshold, kelly_final_bankroll, kelly_total_volume,
            kelly_correct_bets, kelly_total_bets, KELLY_FRACTION, FIXED_BET_FRACTION,
            earliest_fight_date, fixed_monthly_profit, kelly_monthly_profit
        )

    # MODIFIED: Calculate and print overall metrics using ensemble predictions but selecting the class with higher confidence
    if use_ensemble:
        # Average predictions across all models
        y_pred_proba_avg = np.mean([y_pred_proba_list[j] for j in range(len(y_pred_proba_list))], axis=0)

        # For each sample, predict the class with highest confidence
        y_pred = np.zeros(len(y_test))
        for i in range(len(y_test)):
            y_pred[i] = 1 if y_pred_proba_avg[i][1] > y_pred_proba_avg[i][0] else 0
    else:
        # If not using ensemble, just use the single model
        y_pred_proba_avg = y_pred_proba_list[0]
        y_pred = np.array([1 if proba[1] > proba[0] else 0 for proba in y_pred_proba_avg])

    print_overall_metrics(y_test, y_pred, y_pred_proba_avg)
    try:
        console = Console()
        console.print(f"\n[bold cyan]Generating {calibration_type.capitalize()} Calibration Plots[/bold cyan]")

        # Create standard calibration curves with model names
        calibration_files = create_and_save_calibration_curves(
            y_test,
            y_pred_proba_list,
            use_ensemble,
            calibration_type=calibration_type,
            model_names=model_names
        )

        # Create reliability diagram with histogram
        reliability_file = create_reliability_diagram(
            y_test,
            y_pred_proba_list,
            use_ensemble,
            calibration_type=calibration_type,
            model_names=model_names
        )

        print(f"\n[{calibration_type.capitalize()} Calibration Plots Generated]")
        print(f"Main calibration curve: {calibration_files['main_model']}")

        if use_ensemble and 'individual_models' in calibration_files:
            print(f"Individual models comparison: {calibration_files['individual_models']}")

        print(f"Reliability diagram: {reliability_file}")

        print("\nInterpreting Calibration for Kelly Betting:")
        print("- Perfect calibration means optimal Kelly bet sizing")
        print("- Under-confidence (points above diagonal) leads to smaller bets than optimal")
        print("- Over-confidence (points below diagonal) leads to larger bets than optimal")
        print("- For maximum profit with Kelly criterion, calibration is critical")

        # Add analysis of calibration results
        if use_ensemble:
            y_pred_proba_avg = np.mean([y_pred_proba for y_pred_proba in y_pred_proba_list], axis=0)
            prob_true, prob_pred = calibration_curve(y_test, y_pred_proba_avg[:, 1], n_bins=10)
            model_label = "Ensemble model"
        else:
            y_pred_proba = y_pred_proba_list[0]
            prob_true, prob_pred = calibration_curve(y_test, y_pred_proba[:, 1], n_bins=10)
            model_label = model_names[0]

        # Calculate average calibration error
        calibration_error = np.mean(np.abs(prob_true - prob_pred))
        print(f"\nAverage calibration error for {model_label}: {calibration_error:.4f}")

        # Determine if model is generally over or under confident
        if np.mean(prob_true - prob_pred) > 0:
            print(f"Model tendency: Under-confident (true probabilities > predicted)")
            print("Betting implication: Bets are smaller than optimal")
        elif np.mean(prob_true - prob_pred) < 0:
            print(f"Model tendency: Over-confident (predicted > true probabilities)")
            print("Betting implication: Bets are larger than optimal")
        else:
            print(f"Model tendency: Well calibrated overall")
            print("Betting implication: Optimal bet sizing")

    except Exception as e:
        print(f"\n[Warning] Error generating calibration plots: {str(e)}")
        print("Continuing with analysis without calibration plots.")

    # Restore stdout and print final output
    sys.stdout = old_stdout
    output = mystdout.getvalue()
    console = Console(width=93)
    main_panel = Panel(
        output,
        title=f"Past Fight Testing (Odds Type: {odds_type.capitalize()}, Calibration: {calibration_type.capitalize()})",
        border_style="bold magenta",
        expand=True,
    )
    console.print(main_panel)


if __name__ == "__main__":
    main(manual_threshold=0.5,
         use_calibration=True,
         initial_bankroll=10000,
         kelly_fraction=0.5,
         fixed_bet_fraction=0.1,
         max_bet_percentage=0.1,
         min_odds=-300,
         use_ensemble=False,
         odds_type='close')  # Options: 'open', 'close', 'average'

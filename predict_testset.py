import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import numpy as np
import multiprocessing as mp
from functools import partial
from sklearn.utils.validation import check_is_fitted
from tqdm import tqdm
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich.console import Group
import datetime
from io import StringIO
import sys
from rich.console import Console
from rich.panel import Panel
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV


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


def load_model(model_path, model_type='xgboost'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(enable_categorical=True)
            model.load_model(model_path)
        elif model_type == 'lightgbm':
            model = lgb.Booster(model_file=model_path)
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


def calculate_kelly_fraction(p, b, kelly_fraction):
    q = 1 - p
    full_kelly = max(0, (p - (q / b)))
    return full_kelly * kelly_fraction


def print_fight_results(confident_bets):
    console = Console(width=160)
    for bet in confident_bets:
        fighter_a = bet['Fighter A'].title()
        fighter_b = bet['Fighter B'].title()
        date_obj = datetime.datetime.strptime(bet['Date'], '%Y-%m-%d')
        formatted_date = date_obj.strftime('%B %d, %Y')

        # Calculate Fixed Fraction stake as a percentage of starting bankroll
        fixed_starting_bankroll = float(bet.get('Fixed Fraction Starting Bankroll', '0').replace('$', ''))
        fixed_stake = float(bet.get('Fixed Fraction Stake', '0').replace('$', ''))
        fixed_stake_percentage = (fixed_stake / fixed_starting_bankroll) * 100 if fixed_starting_bankroll > 0 else 0

        fixed_panel = Panel(
            f"Starting Bankroll: {bet.get('Fixed Fraction Starting Bankroll', 'N/A')}\n"
            f"Available Bankroll: {bet.get('Fixed Fraction Available Bankroll', 'N/A')}\n"
            f"Stake: {bet.get('Fixed Fraction Stake', 'N/A')}\n"
            f"Stake Percentage: {fixed_stake_percentage:.2f}%\n"
            f"Potential Profit: {bet.get('Fixed Fraction Potential Profit', 'N/A')}\n"
            f"Bankroll After: {bet.get('Fixed Fraction Bankroll After', 'N/A')}\n"
            f"Profit: ${bet.get('Fixed Fraction Profit', 0):.2f}\n"
            f"ROI (of daily bankroll): {bet.get('Fixed Fraction ROI', 0):.2f}%",
            title="Fixed Fraction",
            expand=True,
            width=42
        )

        # Calculate Kelly stake as a percentage of starting bankroll
        kelly_starting_bankroll = float(bet.get('Kelly Starting Bankroll', '0').replace('$', ''))
        kelly_stake = float(bet.get('Kelly Stake', '0').replace('$', ''))
        kelly_stake_percentage = (kelly_stake / kelly_starting_bankroll) * 100 if kelly_starting_bankroll > 0 else 0

        kelly_panel = Panel(
            f"Starting Bankroll: {bet.get('Kelly Starting Bankroll', 'N/A')}\n"
            f"Available Bankroll: {bet.get('Kelly Available Bankroll', 'N/A')}\n"
            f"Stake: {bet.get('Kelly Stake', 'N/A')}\n"
            f"Stake Percentage: {kelly_stake_percentage:.2f}%\n"
            f"Potential Profit: {bet.get('Kelly Potential Profit', 'N/A')}\n"
            f"Bankroll After: {bet.get('Kelly Bankroll After', 'N/A')}\n"
            f"Profit: ${bet.get('Kelly Profit', 0):.2f}\n"
            f"ROI (of daily bankroll): {bet.get('Kelly ROI', 0):.2f}%",
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


def evaluate_bets(y_test, y_pred_proba_list, test_data, confidence_threshold, initial_bankroll=10000,
                  kelly_fraction=0.125, fixed_bet_fraction=0.001, default_bet=0.00, min_odds=-300, print_fights=True,
                  max_bet_percentage=0.20, use_ensemble=True):
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
    processed_fights = set()

    test_data = test_data.sort_values(by=['current_fight_date', 'fighter', 'fighter_b'], ascending=[True, True, True])
    test_data = test_data.reset_index(drop=True)

    daily_fixed_bankrolls = {}
    daily_kelly_bankrolls = {}
    daily_fixed_stakes = {}
    daily_kelly_stakes = {}
    daily_fixed_profits = {}
    daily_kelly_profits = {}

    current_date = None

    for i in range(len(test_data)):
        row = test_data.iloc[i]
        fight_id = frozenset([row['fighter'], row['fighter_b']])
        fight_date = row['current_fight_date']

        if fight_id in processed_fights:
            continue

        processed_fights.add(fight_id)

        if fight_date != current_date:
            if current_date is not None:
                fixed_bankroll += daily_fixed_profits.get(current_date, 0)
                kelly_bankroll += daily_kelly_profits.get(current_date, 0)
                daily_fixed_bankrolls[current_date] = fixed_bankroll
                daily_kelly_bankrolls[current_date] = kelly_bankroll
            current_date = fight_date
            daily_fixed_stakes[current_date] = 0
            daily_kelly_stakes[current_date] = 0
            daily_fixed_profits[current_date] = 0
            daily_kelly_profits[current_date] = 0

        true_winner = row['fighter'] if y_test.iloc[i] == 1 else row['fighter_b']

        if use_ensemble:
            y_pred_proba_avg = np.mean([y_pred_proba[i] for y_pred_proba in y_pred_proba_list], axis=0)
        else:
            y_pred_proba_avg = y_pred_proba_list[0][i]

        winning_probability = max(y_pred_proba_avg)
        predicted_winner = row['fighter'] if y_pred_proba_avg[1] > y_pred_proba_avg[0] else row['fighter_b']

        if use_ensemble:
            models_agreeing = sum([1 for y_pred_proba in y_pred_proba_list if
                                   (y_pred_proba[i][1] > y_pred_proba[i][0]) == (
                                               y_pred_proba_avg[1] > y_pred_proba_avg[0])])
        else:
            models_agreeing = 1

        confident_predictions += 1
        if predicted_winner == true_winner:
            correct_confident_predictions += 1

        if winning_probability >= confidence_threshold and models_agreeing >= (5 if use_ensemble else 1):
            odds = row['current_fight_open_odds'] if predicted_winner == row['fighter'] else row[
                'current_fight_open_odds_b']

            if odds < min_odds:
                continue

            available_fixed_bankroll = fixed_bankroll - daily_fixed_stakes[current_date]
            available_kelly_bankroll = kelly_bankroll - daily_kelly_stakes[current_date]

            fixed_max_bet = fixed_bankroll * max_bet_percentage
            kelly_max_bet = kelly_bankroll * max_bet_percentage

            fixed_stake = min(fixed_bankroll * fixed_bet_fraction, available_fixed_bankroll, fixed_max_bet)

            b = odds / 100 if odds > 0 else 100 / abs(odds)
            kelly_bet_size = calculate_kelly_fraction(winning_probability, b, kelly_fraction)
            kelly_stake = kelly_bankroll * kelly_bet_size  # Calculate based on starting bankroll
            kelly_stake = min(kelly_stake, available_kelly_bankroll, kelly_max_bet)  # Apply limits

            if kelly_stake <= kelly_bankroll * default_bet:
                kelly_stake = min(kelly_bankroll * default_bet, available_kelly_bankroll, kelly_max_bet)

            bet_result = {
                'Fight': i + 1,
                'Fighter A': row['fighter'],
                'Fighter B': row['fighter_b'],
                'Date': fight_date,
                'True Winner': true_winner,
                'Predicted Winner': predicted_winner,
                'Confidence': f"{winning_probability:.2%}",
                'Odds': odds,
                'Models Agreeing': models_agreeing
            }

            if fixed_stake > 0:
                fixed_total_bets += 1
                daily_fixed_stakes[current_date] += fixed_stake
                fixed_profit = calculate_profit(odds, fixed_stake)
                fixed_total_volume += fixed_stake

                bet_result.update({
                    'Fixed Fraction Starting Bankroll': f"${fixed_bankroll:.2f}",
                    'Fixed Fraction Available Bankroll': f"${available_fixed_bankroll:.2f}",
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
                bet_result['Fixed Fraction ROI'] = (bet_result['Fixed Fraction Profit'] / fixed_bankroll) * 100

            if kelly_stake > 0:
                kelly_total_bets += 1
                daily_kelly_stakes[current_date] += kelly_stake
                kelly_profit = calculate_profit(odds, kelly_stake)
                kelly_total_volume += kelly_stake

                bet_result.update({
                    'Kelly Starting Bankroll': f"${kelly_bankroll:.2f}",
                    'Kelly Available Bankroll': f"${available_kelly_bankroll:.2f}",
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
                bet_result['Kelly ROI'] = (bet_result['Kelly Profit'] / kelly_bankroll) * 100

            confident_bets.append(bet_result)

    if current_date is not None:
        fixed_bankroll += daily_fixed_profits.get(current_date, 0)
        kelly_bankroll += daily_kelly_profits.get(current_date, 0)
        daily_fixed_bankrolls[current_date] = fixed_bankroll
        daily_kelly_bankrolls[current_date] = kelly_bankroll

    if print_fights:
        print_fight_results(confident_bets)

    return (fixed_bankroll, fixed_total_volume, fixed_correct_bets, fixed_total_bets,
            kelly_bankroll, kelly_total_volume, kelly_correct_bets, kelly_total_bets,
            confident_predictions, correct_confident_predictions,
            daily_fixed_bankrolls, daily_kelly_bankrolls)


def calculate_monthly_roi(daily_bankrolls, initial_bankroll):
    monthly_roi = {}
    monthly_profit = {}
    current_month = None
    current_bankroll = initial_bankroll
    month_start_bankroll = initial_bankroll
    total_profit = 0

    print("\nDetailed ROI Calculation:")
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
    print(f"Number of days in dataset: {len(sorted_dates)}")
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
    overall_accuracy = accuracy_score(y_test, y_pred)
    overall_precision = precision_score(y_test, y_pred)
    overall_recall = recall_score(y_test, y_pred)
    overall_f1 = f1_score(y_test, y_pred)
    overall_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

    console = Console()
    table = Table(title="Overall Model Metrics (all predictions)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")

    table.add_row("Accuracy", f"{overall_accuracy:.4f}")
    table.add_row("Precision", f"{overall_precision:.4f}")
    table.add_row("Recall", f"{overall_recall:.4f}")
    table.add_row("F1 Score", f"{overall_f1:.4f}")
    table.add_row("AUC", f"{overall_auc:.4f}")

    console.print(table)


def evaluate_threshold(threshold, y_test, y_pred_proba_list, test_data_with_display, INITIAL_BANKROLL, KELLY_FRACTION,
                       FIXED_BET_FRACTION, MAX_BET_PERCENTAGE, use_ensemble):
    (final_bankroll, total_volume, correct_bets, total_bets, kelly_final_bankroll, kelly_total_volume,
     kelly_correct_bets, kelly_total_bets, confident_predictions, correct_confident_predictions,
     daily_fixed_bankrolls, daily_kelly_bankrolls) = evaluate_bets(
        y_test, y_pred_proba_list, test_data_with_display, threshold, INITIAL_BANKROLL,
        KELLY_FRACTION, FIXED_BET_FRACTION, default_bet=0.00, print_fights=False,
        max_bet_percentage=MAX_BET_PERCENTAGE, use_ensemble=use_ensemble)

    kelly_roi = ((kelly_final_bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL) * 100

    return (threshold, kelly_roi, final_bankroll, total_volume, correct_bets, total_bets, confident_predictions,
            correct_confident_predictions, kelly_final_bankroll, kelly_total_volume)


class LGBMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ['classes_'])
        raw_preds = self.model.predict(X, raw_score=True)
        proba = 1 / (1 + np.exp(-raw_preds))
        return np.column_stack((1 - proba, proba))

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def main(optimize_threshold=True, manual_threshold=None, use_calibration=True,
         initial_bankroll=10000, kelly_fraction=1, fixed_bet_fraction=0.1,
         max_bet_percentage=0.25, min_odds=-300, use_ensemble=True):
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    INITIAL_BANKROLL = initial_bankroll
    KELLY_FRACTION = kelly_fraction
    FIXED_BET_FRACTION = fixed_bet_fraction
    MAX_BET_PERCENTAGE = max_bet_percentage

    val_data = pd.read_csv('data/train test data/val_data.csv')
    test_data = pd.read_csv('data/train test data/test_data.csv')

    display_columns = ['current_fight_date', 'fighter', 'fighter_b']
    y_val = val_data['winner']
    y_test = test_data['winner']
    X_val = val_data.drop(['winner'] + display_columns, axis=1)
    X_test = test_data.drop(['winner'] + display_columns, axis=1)

    X_val = preprocess_data(X_val)
    X_test = preprocess_data(X_test)

    display_data = test_data[display_columns]
    test_data_with_display = pd.concat([X_test, display_data], axis=1)

    model_files = [
        'model_0.6647_auc_diff_0.0446.json',
        'model_0.6647_auc_diff_0.0448.json',
        'model_0.6677_auc_diff_0.0406.json',
        'model_0.6677_auc_diff_0.0442.json',
        'model_0.6677_auc_diff_0.0465.json'
    ]

    # model_files = [
    #     'model_0.7007_auc_diff_0.0046.json',
    #     'model_0.7007_auc_diff_0.0058.json',
    #     'model_0.7039_auc_diff_0.0012.json',
    #     'model_0.7039_auc_diff_0.0027.json',
    #     'model_0.7039_auc_diff_0.0033.json'
    # ]

    models = []
    if use_ensemble:
        for model_file in model_files:
            model_path = os.path.abspath(f'models/xgboost/jan2024-july2024/125/{model_file}')
            model = load_model(model_path, 'xgboost')
            models.append(model)
    else:
        model_path = os.path.abspath(f'models/xgboost/jan2024-july2024/125/{model_files[0]}')
        model = load_model(model_path, 'xgboost')
        models.append(model)

    expected_features = models[0].get_booster().feature_names
    X_val = X_val.reindex(columns=expected_features)
    X_test = X_test.reindex(columns=expected_features)

    y_pred_proba_list = []
    for model in models:
        if use_calibration:
            calibrated_model = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')
            calibrated_model.fit(X_val, y_val)
            y_pred_proba = calibrated_model.predict_proba(X_test)
        else:
            y_pred_proba = model.predict_proba(X_test)
        y_pred_proba_list.append(y_pred_proba)

    if optimize_threshold:
        print("Evaluating thresholds...")
        thresholds = np.arange(0.5, 0.75, 0.0001)

        evaluate_threshold_partial = partial(evaluate_threshold,
                                             y_test=y_test,
                                             y_pred_proba_list=y_pred_proba_list,
                                             test_data_with_display=test_data_with_display,
                                             INITIAL_BANKROLL=INITIAL_BANKROLL,
                                             KELLY_FRACTION=KELLY_FRACTION,
                                             FIXED_BET_FRACTION=FIXED_BET_FRACTION,
                                             MAX_BET_PERCENTAGE=MAX_BET_PERCENTAGE,
                                             use_ensemble=use_ensemble)

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = list(tqdm(pool.imap(evaluate_threshold_partial, thresholds), total=len(thresholds)))

        best_result = max(results, key=lambda x: x[1])
        best_threshold, best_kelly_roi, *_ = best_result
    else:
        if manual_threshold is None:
            raise ValueError("If optimize_threshold is False, you must provide a manual_threshold value.")
        best_threshold = manual_threshold

    bet_results = evaluate_bets(y_test, y_pred_proba_list, test_data_with_display, best_threshold, INITIAL_BANKROLL,
                                KELLY_FRACTION, FIXED_BET_FRACTION, default_bet=0.00, print_fights=True,
                                max_bet_percentage=MAX_BET_PERCENTAGE, min_odds=min_odds, use_ensemble=use_ensemble)

    (fixed_final_bankroll, fixed_total_volume, fixed_correct_bets, fixed_total_bets,
     kelly_final_bankroll, kelly_total_volume, kelly_correct_bets, kelly_total_bets,
     confident_predictions, correct_confident_predictions,
     daily_fixed_bankrolls, daily_kelly_bankrolls) = bet_results

    earliest_fight_date = test_data['current_fight_date'].min()

    fixed_monthly_roi, fixed_monthly_profit, fixed_total_roi = calculate_monthly_roi(daily_fixed_bankrolls,
                                                                                     INITIAL_BANKROLL)
    kelly_monthly_roi, kelly_monthly_profit, kelly_total_roi = calculate_monthly_roi(daily_kelly_bankrolls,
                                                                                     INITIAL_BANKROLL)

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

    print_betting_results(len(test_data), confident_predictions, correct_confident_predictions,
                          fixed_total_bets, fixed_correct_bets, INITIAL_BANKROLL, fixed_final_bankroll,
                          fixed_total_volume, best_threshold, kelly_final_bankroll, kelly_total_volume,
                          kelly_correct_bets, kelly_total_bets, KELLY_FRACTION, FIXED_BET_FRACTION,
                          earliest_fight_date, fixed_monthly_profit, kelly_monthly_profit)

    y_pred_avg = np.mean([y_pred_proba[:, 1] for y_pred_proba in y_pred_proba_list], axis=0)
    y_pred = (y_pred_avg > 0.5).astype(int)
    print_overall_metrics(y_test, y_pred, np.column_stack((1 - y_pred_avg, y_pred_avg)))

    sys.stdout = old_stdout
    output = mystdout.getvalue()
    console = Console(width=93)
    main_panel = Panel(
        output,
        title="Past Fight Testing",
        border_style="bold magenta",
        expand=True,
    )
    console.print(main_panel)


if __name__ == "__main__":
    main(optimize_threshold=False, manual_threshold=0.50,
         use_calibration=True, initial_bankroll=10000, kelly_fraction=1,
         fixed_bet_fraction=0.1, max_bet_percentage=0.20, min_odds=-500,
         use_ensemble=True)

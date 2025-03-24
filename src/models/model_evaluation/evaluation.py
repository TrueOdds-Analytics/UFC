"""
Model evaluation functions for MMA betting models
"""
import os
import sys
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from betting_utils import (
    calculate_profit, calculate_kelly_fraction, calculate_average_odds
)
from data_utils import load_model
from calibrators import calibrate_predictions_with_range_method
from visualization import (
    create_and_save_calibration_curves, create_reliability_diagram,
    create_range_based_reliability_diagram
)
from metrics import (
    write_metrics_to_csv, analyze_calibration_quality
)


def evaluate_bets(y_test, y_pred_proba_list, test_data, confidence_threshold, initial_bankroll=10000,
                  kelly_fraction=0.125, fixed_bet_fraction=0.001, default_bet=0.00, min_odds=-300,
                  max_underdog_odds=200, print_fights=True, max_bet_percentage=0.20, use_ensemble=True,
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
    if print_fights and len(confident_bets) > 0:
        from betting_utils import print_fight_results
        print_fight_results(confident_bets)

    return (fixed_bankroll, fixed_total_volume, fixed_correct_bets, fixed_total_bets,
            kelly_bankroll, kelly_total_volume, kelly_correct_bets, kelly_total_bets,
            confident_predictions, correct_confident_predictions,
            daily_fixed_bankrolls, daily_kelly_bankrolls)


def evaluate_single_model(model_file, model_path, X_val, y_val, X_test, y_test, test_data_with_display,
                          initial_bankroll, kelly_fraction, fixed_bet_fraction, max_bet_percentage,
                          min_odds, odds_type, use_calibration=True, calibration_type='isotonic',
                          max_underdog_odds=200):
    """
    Evaluate a single model and return its metrics using a fixed confidence threshold of 0.5

    Args:
        model_file: Name of the model file
        model_path: Path to the model directory
        X_val, y_val: Validation data features and targets
        X_test, y_test: Test data features and targets
        test_data_with_display: Test data with display columns
        initial_bankroll: Initial bankroll for betting simulation
        kelly_fraction: Kelly fraction for betting
        fixed_bet_fraction: Fixed fraction for betting
        max_bet_percentage: Maximum bet percentage
        min_odds: Minimum odds to consider
        odds_type: Type of odds to use ('open', 'close', 'average')
        use_calibration: Whether to use calibration
        calibration_type: Type of calibration ('isotonic', 'range_based', 'uncalibrated')
        max_underdog_odds: Maximum underdog odds to bet on

    Returns:
        Dictionary containing model metrics
    """
    # Load the model
    full_model_path = os.path.join(model_path, model_file)
    model = load_model(full_model_path, 'xgboost')

    # Apply calibration if needed
    if use_calibration:
        if calibration_type == 'range_based':
            # Define custom ranges for different probability regions
            calibration_ranges = [0.25, 0.45, 0.65, 0.85]

            # Apply range-based calibration
            model_list = [model]  # Put in list format for the function
            y_pred_proba_list, _ = calibrate_predictions_with_range_method(
                model_list, X_val, y_val, X_test, ranges=calibration_ranges
            )
            y_pred_proba = y_pred_proba_list[0]  # Get the first (and only) model's predictions
        else:
            # Use standard scikit-learn calibration (isotonic by default)
            calibration_type = 'isotonic'  # Default to isotonic if not range_based
            calibrated_model = CalibratedClassifierCV(model, cv='prefit', method=calibration_type)
            calibrated_model.fit(X_val, y_val)
            y_pred_proba = calibrated_model.predict_proba(X_test)
    else:
        calibration_type = 'uncalibrated'
        y_pred_proba = model.predict_proba(X_test)

    # Put in list format for evaluate_bets
    y_pred_proba_list = [y_pred_proba]

    # Binary predictions
    y_pred = np.array([1 if proba[1] > proba[0] else 0 for proba in y_pred_proba])

    # Fixed threshold of 0.5
    threshold = 0.5

    # Evaluate betting performance with fixed threshold
    bet_results = evaluate_bets(
        y_test, y_pred_proba_list, test_data_with_display, threshold,
        initial_bankroll, kelly_fraction, fixed_bet_fraction,
        default_bet=0.00, print_fights=False, max_bet_percentage=max_bet_percentage,
        min_odds=min_odds, use_ensemble=False, odds_type=odds_type,
        max_underdog_odds=max_underdog_odds
    )

    # Unpack results
    (fixed_final_bankroll, fixed_total_volume, fixed_correct_bets, fixed_total_bets,
     kelly_final_bankroll, kelly_total_volume, kelly_correct_bets, kelly_total_bets,
     confident_predictions, correct_confident_predictions,
     daily_fixed_bankrolls, daily_kelly_bankrolls) = bet_results

    # Calculate ROI
    fixed_roi = ((fixed_final_bankroll - initial_bankroll) / initial_bankroll) * 100
    kelly_roi = ((kelly_final_bankroll - initial_bankroll) / initial_bankroll) * 100

    # Calculate betting accuracy
    fixed_accuracy = fixed_correct_bets / fixed_total_bets if fixed_total_bets > 0 else 0
    kelly_accuracy = kelly_correct_bets / kelly_total_bets if kelly_total_bets > 0 else 0

    # Calculate overall model accuracy
    overall_accuracy = accuracy_score(y_test, y_pred)

    # Calculate calibration error
    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba[:, 1], n_bins=10)
    calibration_error = np.mean(np.abs(prob_true - prob_pred))

    # Calculate AUC
    auc = roc_auc_score(y_test, y_pred_proba[:, 1]) if len(np.unique(y_test)) > 1 else 0

    # Determine if model is generally over or under confident
    calibration_bias = np.mean(prob_true - prob_pred)
    if calibration_bias > 0.01:
        calibration_tendency = "Under-confident"
    elif calibration_bias < -0.01:
        calibration_tendency = "Over-confident"
    else:
        calibration_tendency = "Well-calibrated"

    # Return metrics as dictionary
    metrics = {
        'model_name': os.path.splitext(model_file)[0],
        'calibration_type': calibration_type,
        'calibration_tendency': calibration_tendency,
        'calibration_bias': calibration_bias,
        'confidence_threshold': threshold,  # Fixed threshold of 0.5
        'overall_accuracy': overall_accuracy,
        'auc': auc,
        'calibration_error': calibration_error,
        'fixed_roi': fixed_roi,
        'kelly_roi': kelly_roi,
        'fixed_accuracy': fixed_accuracy,
        'kelly_accuracy': kelly_accuracy,
        'fixed_total_bets': fixed_total_bets,
        'kelly_total_bets': kelly_total_bets,
        'fixed_final_bankroll': fixed_final_bankroll,
        'kelly_final_bankroll': kelly_final_bankroll,
        'confident_predictions': confident_predictions,
        'correct_confident_predictions': correct_confident_predictions
    }

    return metrics


def evaluate_model_by_name(model_name, model_directory, val_data_path, test_data_path,
                           encoder_path, display_columns, output_dir,
                           use_calibration=True, calibration_type='isotonic', initial_bankroll=10000,
                           kelly_fraction=0.5, fixed_bet_fraction=0.1, max_bet_percentage=0.1,
                           min_odds=-300, odds_type='close', max_underdog_odds=200):
    """
    Evaluate a single model by name

    Args:
        model_name: Name of the model file (with or without .json extension)
        model_directory: Path to directory containing model files
        val_data_path: Path to validation data
        test_data_path: Path to test data
        encoder_path: Path to encoder file
        display_columns: List of columns to keep for display
        output_dir: Directory for output files
        use_calibration: Whether to use calibration
        calibration_type: Type of calibration ('isotonic', 'range_based', 'uncalibrated')
        initial_bankroll: Initial bankroll for betting simulation
        kelly_fraction: Kelly fraction for betting
        fixed_bet_fraction: Fixed fraction for betting
        max_bet_percentage: Maximum bet percentage
        min_odds: Minimum odds to consider
        odds_type: Type of odds to use ('open', 'close', 'average')
        max_underdog_odds: Maximum underdog odds to bet on

    Returns:
        Dictionary containing model metrics
    """
    # Add .json extension if not provided
    if not model_name.endswith('.json'):
        model_name = f"{model_name}.json"

    # Check if model exists
    model_path = os.path.join(model_directory, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Redirect stdout to capture output
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    # Load and preprocess data
    from data_utils import prepare_datasets
    X_val, y_val, X_test, y_test, test_data_with_display, encoder = prepare_datasets(
        val_data_path, test_data_path, encoder_path, display_columns
    )

    # Load model to get expected features
    first_model = load_model(model_path, 'xgboost')
    expected_features = first_model.get_booster().feature_names

    # Ensure consistent feature ordering
    X_val = X_val.reindex(columns=expected_features)
    X_test = X_test.reindex(columns=expected_features)

    # Evaluate the model
    print(f"Evaluating model: {model_name}")
    metrics = evaluate_single_model(
        model_name, model_directory, X_val, y_val, X_test, y_test,
        test_data_with_display, initial_bankroll, kelly_fraction,
        fixed_bet_fraction, max_bet_percentage, min_odds, odds_type,
        use_calibration, calibration_type, max_underdog_odds
    )

    # Display detailed results
    console = Console()
    console.print(f"\n[bold cyan]Model Evaluation: {metrics['model_name']}[/bold cyan]")

    # Create results table
    table = Table(title=f"Model Performance Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="magenta")

    # Add metrics to table
    table.add_row("Model Name", metrics['model_name'])
    table.add_row("Calibration Type", metrics['calibration_type'])
    table.add_row("Confidence Threshold", f"{metrics['confidence_threshold']:.2f}")
    table.add_row("Kelly ROI", f"{metrics['kelly_roi']:.2f}%")
    table.add_row("Fixed ROI", f"{metrics['fixed_roi']:.2f}%")
    table.add_row("Calibration Error", f"{metrics['calibration_error']:.4f}")
    table.add_row("Calibration Tendency", metrics['calibration_tendency'])
    table.add_row("Calibration Bias", f"{metrics['calibration_bias']:.4f}")
    table.add_row("Overall Accuracy", f"{metrics['overall_accuracy']:.4f}")
    table.add_row("AUC", f"{metrics['auc']:.4f}")
    table.add_row("Kelly Betting Accuracy", f"{metrics['kelly_accuracy']:.4f}")
    table.add_row("Fixed Betting Accuracy", f"{metrics['fixed_accuracy']:.4f}")
    table.add_row("Kelly Total Bets", str(metrics['kelly_total_bets']))
    table.add_row("Fixed Total Bets", str(metrics['fixed_total_bets']))
    table.add_row("Initial Bankroll", f"${initial_bankroll:.2f}")
    table.add_row("Kelly Final Bankroll", f"${metrics['kelly_final_bankroll']:.2f}")
    table.add_row("Fixed Final Bankroll", f"${metrics['fixed_final_bankroll']:.2f}")

    console.print(table)

    # Generate calibration plots
    try:
        # Load model and generate predictions
        model = load_model(model_path, 'xgboost')

        # Apply appropriate calibration
        if use_calibration:
            if calibration_type == 'range_based':
                # Define ranges for different probability regions
                calibration_ranges = [0.25, 0.45, 0.65, 0.85]  # Creates 5 regions

                # Apply range-based calibration
                model_list = [model]
                y_pred_proba_list, _ = calibrate_predictions_with_range_method(
                    model_list, X_val, y_val, X_test, ranges=calibration_ranges
                )

                # Generate calibration plot
                console.print(f"\n[bold cyan]Generating Range-Based Calibration Plots[/bold cyan]")

                # Create reliability diagram comparing original vs range-based calibration
                reliability_file, _ = create_range_based_reliability_diagram(
                    y_test, y_pred_proba_list, use_ensemble=False,
                    ranges=calibration_ranges,
                    output_dir=output_dir,
                    model_names=[metrics['model_name']]
                )

                print(f"\n[Range-Based Calibration Plots Generated]")
                print(f"Reliability diagram: {reliability_file}")

            else:
                # Use standard scikit-learn calibration
                calibration_type = 'isotonic'  # Default
                calibrated_model = CalibratedClassifierCV(model, cv='prefit', method=calibration_type)
                calibrated_model.fit(X_val, y_val)

                y_pred_proba = calibrated_model.predict_proba(X_test)
                y_pred_proba_list = [y_pred_proba]

                console.print(f"\n[bold cyan]Generating {calibration_type.capitalize()} Calibration Plots[/bold cyan]")

                # Create calibration curves
                calibration_files = create_and_save_calibration_curves(
                    y_test,
                    y_pred_proba_list,
                    use_ensemble=False,
                    output_dir=output_dir,
                    calibration_type=calibration_type,
                    model_names=[metrics['model_name']]
                )

                # Create reliability diagram
                reliability_file = create_reliability_diagram(
                    y_test,
                    y_pred_proba_list,
                    use_ensemble=False,
                    output_dir=output_dir,
                    calibration_type=calibration_type,
                    model_names=[metrics['model_name']]
                )

                print(f"\n[{calibration_type.capitalize()} Calibration Plots Generated]")
                print(f"Main calibration curve: {calibration_files['main_model']}")
                print(f"Reliability diagram: {reliability_file}")
        else:
            # No calibration
            calibration_type = 'uncalibrated'
            y_pred_proba = model.predict_proba(X_test)
            y_pred_proba_list = [y_pred_proba]

            console.print(f"\n[bold cyan]Generating Uncalibrated Calibration Plots[/bold cyan]")

            # Create calibration curves
            calibration_files = create_and_save_calibration_curves(
                y_test,
                y_pred_proba_list,
                use_ensemble=False,
                output_dir=output_dir,
                calibration_type='uncalibrated',
                model_names=[metrics['model_name']]
            )

            # Create reliability diagram
            reliability_file = create_reliability_diagram(
                y_test,
                y_pred_proba_list,
                use_ensemble=False,
                output_dir=output_dir,
                calibration_type='uncalibrated',
                model_names=[metrics['model_name']]
            )

            print(f"\n[Uncalibrated Calibration Plots Generated]")
            print(f"Main calibration curve: {calibration_files['main_model']}")
            print(f"Reliability diagram: {reliability_file}")

    except Exception as e:
        print(f"\n[Warning] Error generating calibration plots: {str(e)}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_output = f"{metrics['model_name']}_evaluation.csv"
    metrics_df.to_csv(metrics_output, index=False)
    print(f"Metrics saved to {metrics_output}")

    # Restore stdout and print output
    sys.stdout = old_stdout
    output = mystdout.getvalue()
    console = Console(width=100)
    main_panel = Panel(
        output,
        title=f"Model Evaluation: {metrics['model_name']} (Odds: {odds_type.capitalize()}, Calibration: {calibration_type.capitalize()})",
        border_style="bold magenta",
        expand=True,
    )
    console.print(main_panel)

    return metrics


def evaluate_all_models(model_directory, val_data_path, test_data_path, encoder_path,
                        display_columns, output_dir, use_calibration=True,
                        calibration_type='isotonic', initial_bankroll=10000,
                        kelly_fraction=0.5, fixed_bet_fraction=0.1, max_bet_percentage=0.1,
                        min_odds=-300, odds_type='close', max_underdog_odds=200,
                        output_file='model_comparison.csv'):
    """
    Evaluate all models in a directory and save metrics to CSV

    Args:
        model_directory: Path to directory containing model files
        val_data_path: Path to validation data
        test_data_path: Path to test data
        encoder_path: Path to encoder file
        display_columns: List of columns to keep for display
        output_dir: Directory for output files
        use_calibration: Whether to use calibration
        calibration_type: Type of calibration ('isotonic', 'range_based', 'uncalibrated')
        initial_bankroll: Initial bankroll for betting simulation
        kelly_fraction: Kelly fraction for betting
        fixed_bet_fraction: Fixed fraction for betting
        max_bet_percentage: Maximum bet percentage
        min_odds: Minimum odds to consider
        odds_type: Type of odds to use ('open', 'close', 'average')
        max_underdog_odds: Maximum underdog odds to bet on
        output_file: Path to output CSV file

    Returns:
        DataFrame with metrics for all models
    """
    # Redirect stdout to capture output
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    # Get all model files
    from model_evaluation.data_utils import get_models_from_directory
    model_files = get_models_from_directory(model_directory)

    if not model_files:
        print("No models to evaluate. Exiting.")
        sys.stdout = old_stdout
        return pd.DataFrame()

    # Load and preprocess data
    from data_utils import prepare_datasets
    X_val, y_val, X_test, y_test, test_data_with_display, encoder = prepare_datasets(
        val_data_path, test_data_path, encoder_path, display_columns
    )

    # List to collect all metrics
    all_metrics = []

    # Process each model
    for i, model_file in enumerate(model_files):
        print(f"Evaluating model {i + 1}/{len(model_files)}: {model_file}")

        try:
            # Load the first model to get expected features
            if i == 0:
                first_model_path = os.path.join(model_directory, model_file)
                first_model = load_model(first_model_path, 'xgboost')
                expected_features = first_model.get_booster().feature_names

                # Ensure consistent feature ordering
                X_val = X_val.reindex(columns=expected_features)
                X_test = X_test.reindex(columns=expected_features)

            # Evaluate the current model
            metrics = evaluate_single_model(
                model_file, model_directory, X_val, y_val, X_test, y_test,
                test_data_with_display, initial_bankroll, kelly_fraction,
                fixed_bet_fraction, max_bet_percentage, min_odds, odds_type,
                use_calibration, calibration_type, max_underdog_odds
            )

            all_metrics.append(metrics)
            print(f"Model {model_file}")
            print(f"  Threshold: {metrics['confidence_threshold']:.2f}")
            print(f"  Kelly ROI: {metrics['kelly_roi']:.2f}%")
            print(f"  Fixed ROI: {metrics['fixed_roi']:.2f}%")
            print(f"  Calibration Error: {metrics['calibration_error']:.4f} ({metrics['calibration_tendency']})")
            print(f"  Betting Accuracy: {metrics['kelly_accuracy']:.4f}")
            print(f"  Overall Accuracy: {metrics['overall_accuracy']:.4f}")
            print(f"  AUC: {metrics['auc']:.4f}")

        except Exception as e:
            print(f"Error evaluating model {model_file}: {str(e)}")

    # Write metrics to CSV
    metrics_df = write_metrics_to_csv(all_metrics, output_file)

    # Display summary of top models
    console = Console()
    console.print("\n[bold cyan]Model Comparison Summary[/bold cyan]")

    table = Table(title=f"Top 10 Models by Kelly ROI (Calibration: {calibration_type})")
    table.add_column("Model", style="cyan")
    table.add_column("Kelly ROI", justify="right", style="green")
    table.add_column("Threshold", justify="right", style="blue")
    table.add_column("Cal. Error", justify="right", style="yellow")
    table.add_column("Tendency", justify="right", style="magenta")
    table.add_column("Accuracy", justify="right", style="blue")
    table.add_column("AUC", justify="right", style="red")

    # Display top 10 models by Kelly ROI
    max_models = min(10, len(metrics_df))
    for _, row in metrics_df.head(max_models).iterrows():
        table.add_row(
            row['model_name'],
            f"{row['kelly_roi']:.2f}%",
            f"{row['confidence_threshold']:.2f}",
            f"{row['calibration_error']:.4f}",
            row['calibration_tendency'],
            f"{row['overall_accuracy']:.4f}",
            f"{row['auc']:.4f}"
        )

    console.print(table)

    # Restore stdout and print final output
    sys.stdout = old_stdout
    output = mystdout.getvalue()
    console = Console(width=120)
    main_panel = Panel(
        output,
        title=f"Model Comparison (Odds Type: {odds_type.capitalize()}, Calibration: {calibration_type.capitalize()})",
        border_style="bold magenta",
        expand=True,
    )
    console.print(main_panel)

    return metrics_df
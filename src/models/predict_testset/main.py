"""
Main execution module for MMA betting analysis
"""
import os
import sys
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import DATA_PATHS, DISPLAY_COLUMNS, MODEL_FILES, RANGE_CALIBRATION_RANGES, OUTPUT_DIR
from data_utils import load_model, prepare_datasets
from evaluation import evaluate_bets, print_overall_metrics
from betting_utils import (
    calculate_daily_roi, print_daily_roi, calculate_monthly_roi,
    print_betting_results
)
from visualization import (
    create_and_save_calibration_curves, create_reliability_diagram,
    create_range_based_reliability_diagram
)
from calibrators import calibrate_predictions_with_range_method


def main(manual_threshold, use_calibration=True, calibration_type='isotonic',
         initial_bankroll=10000, kelly_fraction=0.5, fixed_bet_fraction=0.1,
         max_bet_percentage=0.25, min_odds=-300, max_underdog_odds=200,
         use_ensemble=True, odds_type='average'):
    """
    Main function for MMA betting analysis

    Args:
        manual_threshold: Confidence threshold for placing bets
        use_calibration: Whether to apply probability calibration
        calibration_type: Type of calibration ('range_based', 'isotonic', or False)
        initial_bankroll: Starting bankroll amount
        kelly_fraction: Fraction of Kelly criterion to use
        fixed_bet_fraction: Fraction of bankroll for fixed betting
        max_bet_percentage: Maximum percentage of bankroll to bet
        min_odds: Minimum odds to place a bet
        use_ensemble: Whether to use ensemble of models
        odds_type: Which odds to use ('open', 'close', 'average')
    """
    # Redirect stdout to capture output
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    # Set constants
    INITIAL_BANKROLL = initial_bankroll
    KELLY_FRACTION = kelly_fraction
    FIXED_BET_FRACTION = fixed_bet_fraction
    MAX_BET_PERCENTAGE = max_bet_percentage

    # Prepare datasets
    X_val, y_val, X_test, y_test, test_data_with_display, encoder = prepare_datasets(
        DATA_PATHS['val_data'],
        DATA_PATHS['test_data'],
        DATA_PATHS['encoder_path'],
        DISPLAY_COLUMNS
    )

    # Extract model names for use in plots
    model_names = [os.path.splitext(model_file)[0] for model_file in MODEL_FILES]

    models = []

    # Load models
    if use_ensemble:
        for model_file in MODEL_FILES:
            model_path = os.path.abspath(f"{DATA_PATHS['model_base_path']}/{model_file}")
            models.append(load_model(model_path, 'xgboost'))
    else:
        model_path = os.path.abspath(f"{DATA_PATHS['model_base_path']}/{MODEL_FILES[0]}")
        models.append(load_model(model_path, 'xgboost'))
        model_names = [model_names[0]]  # Keep only the first model name if not using ensemble

    # Ensure consistent feature ordering
    expected_features = models[0].get_booster().feature_names
    X_val = X_val.reindex(columns=expected_features)
    X_test = X_test.reindex(columns=expected_features)

    # Apply calibration based on selected type
    if use_calibration:
        if calibration_type == 'range_based':
            print(f"Applying range-based calibration...")
            # Apply range-based calibration with custom ranges
            y_pred_proba_list, calibrators = calibrate_predictions_with_range_method(
                models, X_val, y_val, X_test, ranges=RANGE_CALIBRATION_RANGES
            )
        else:
            # Use standard calibration methods
            calibration_type = 'isotonic'  # Default to isotonic if not range_based
            print(f"Applying {calibration_type} calibration...")

            calibrated_models = []
            for model in models:
                calibrated_model = CalibratedClassifierCV(model, cv='prefit', method=calibration_type)
                calibrated_model.fit(X_val, y_val)
                calibrated_models.append(calibrated_model)

            # Initialize prediction lists
            y_pred_proba_list = []
            for _ in range(len(calibrated_models)):
                y_pred_proba_list.append([])

            # Generate predictions row by row
            for i in range(len(X_test)):
                row_features = X_test.iloc[[i]]
                for j, model in enumerate(calibrated_models):
                    y_pred_proba = model.predict_proba(row_features)
                    y_pred_proba_list[j].append(y_pred_proba[0])

            # Convert lists to numpy arrays
            for j in range(len(y_pred_proba_list)):
                y_pred_proba_list[j] = np.array(y_pred_proba_list[j])
    else:
        # No calibration
        calibration_type = 'uncalibrated'
        print("Using uncalibrated models...")

        # Initialize prediction lists
        y_pred_proba_list = []
        for _ in range(len(models)):
            y_pred_proba_list.append([])

        # Generate predictions row by row
        for i in range(len(X_test)):
            row_features = X_test.iloc[[i]]
            for j, model in enumerate(models):
                y_pred_proba = model.predict_proba(row_features)
                y_pred_proba_list[j].append(y_pred_proba[0])

        # Convert lists to numpy arrays
        for j in range(len(y_pred_proba_list)):
            y_pred_proba_list[j] = np.array(y_pred_proba_list[j])

    # Check if there are enough samples for evaluation
    if len(test_data_with_display) == 0:
        console = Console()
        console.print("[bold red]Error: No test data available for evaluation[/bold red]")
        sys.stdout = old_stdout
        return

    # Evaluate bets
    bet_results = evaluate_bets(
        y_test, y_pred_proba_list, test_data_with_display, manual_threshold,
        INITIAL_BANKROLL, KELLY_FRACTION, FIXED_BET_FRACTION,
        default_bet=0.00, print_fights=True, max_bet_percentage=MAX_BET_PERCENTAGE,
        min_odds=min_odds, use_ensemble=use_ensemble, odds_type=odds_type, max_underdog_odds=max_underdog_odds
    )

    (fixed_final_bankroll, fixed_total_volume, fixed_correct_bets, fixed_total_bets,
     kelly_final_bankroll, kelly_total_volume, kelly_correct_bets, kelly_total_bets,
     confident_predictions, correct_confident_predictions,
     daily_fixed_bankrolls, daily_kelly_bankrolls) = bet_results

    # Calculate ROI if we have daily results
    if daily_fixed_bankrolls and daily_kelly_bankrolls:
        earliest_fight_date = test_data_with_display['current_fight_date'].min()
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
            console.print("\nMonthly ROI (based on monthly performance):")
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
            len(test_data_with_display), confident_predictions, correct_confident_predictions,
            fixed_total_bets, fixed_correct_bets, INITIAL_BANKROLL, fixed_final_bankroll,
            fixed_total_volume, manual_threshold, kelly_final_bankroll, kelly_total_volume,
            kelly_correct_bets, kelly_total_bets, KELLY_FRACTION, FIXED_BET_FRACTION,
            earliest_fight_date, fixed_monthly_profit, kelly_monthly_profit
        )

    # Calculate overall metrics
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

    # Print overall metrics
    print_overall_metrics(y_test, y_pred, y_pred_proba_avg)

    # Generate calibration plots
    try:
        console = Console()
        console.print(f"\n[bold cyan]Generating {calibration_type.capitalize()} Calibration Plots[/bold cyan]")

        # For range-based calibration, create special comparison plot
        if calibration_type == 'range_based':
            # Create reliability diagram comparing original vs range-based calibration
            reliability_file, _ = create_range_based_reliability_diagram(
                y_test, y_pred_proba_list, use_ensemble,
                ranges=RANGE_CALIBRATION_RANGES,
                output_dir=OUTPUT_DIR,
                model_names=model_names
            )

            print(f"\n[Range-Based Calibration Plots Generated]")
            print(f"Reliability diagram: {reliability_file}")
        else:
            # Create standard calibration curves
            calibration_files = create_and_save_calibration_curves(
                y_test,
                y_pred_proba_list,
                use_ensemble,
                output_dir=OUTPUT_DIR,
                calibration_type=calibration_type,
                model_names=model_names
            )

            # Create reliability diagram with histogram
            reliability_file = create_reliability_diagram(
                y_test,
                y_pred_proba_list,
                use_ensemble,
                output_dir=OUTPUT_DIR,
                calibration_type=calibration_type,
                model_names=model_names
            )

            print(f"\n[{calibration_type.capitalize()} Calibration Plots Generated]")
            print(f"Main calibration curve: {calibration_files['main_model']}")

            if use_ensemble and 'individual_models' in calibration_files:
                print(f"Individual models comparison: {calibration_files['individual_models']}")

            print(f"Reliability diagram: {reliability_file}")

        # Add calibration interpretation for betting
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
        bias = np.mean(prob_true - prob_pred)
        if bias > 0:
            print(f"Model tendency: Under-confident (true probabilities > predicted)")
            print("Betting implication: Bets are smaller than optimal")
        elif bias < 0:
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
        title=f"MMA Betting Analysis (Odds: {odds_type.capitalize()}, Calibration: {calibration_type.capitalize()})",
        border_style="bold magenta",
        expand=True,
    )
    console.print(main_panel)
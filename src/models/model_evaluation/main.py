"""
Main execution module for MMA betting model evaluation
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


from config import (
    DATA_PATHS, DISPLAY_COLUMNS, MODEL_FILES, RANGE_CALIBRATION_RANGES,
    OUTPUT_DIR, ensure_directories
)
from data_utils import load_model, prepare_datasets
from evaluation import evaluate_bets, evaluate_model_by_name, evaluate_all_models
from metrics import print_overall_metrics, print_betting_results, analyze_calibration_quality
from betting_utils import (
    calculate_daily_roi, print_daily_roi, calculate_monthly_roi
)
from visualization import (
    create_and_save_calibration_curves, create_reliability_diagram,
    create_range_based_reliability_diagram
)
from calibrators import calibrate_predictions_with_range_method


def run_single_model_evaluation(model_name, model_directory, val_data_path, test_data_path,
                                encoder_path, display_columns, output_dir,
                                use_calibration=True, calibration_type='isotonic',
                                initial_bankroll=10000, kelly_fraction=0.5, fixed_bet_fraction=0.1,
                                max_bet_percentage=0.1, min_odds=-300, odds_type='close',
                                max_underdog_odds=200):
    """
    Run evaluation for a single model by name

    Args:
        model_name: Name of the model file
        model_directory: Directory containing the model
        val_data_path: Path to validation data
        test_data_path: Path to test data
        encoder_path: Path to encoder file
        display_columns: Columns to display in output
        output_dir: Directory for output files
        use_calibration: Whether to calibrate model probabilities
        calibration_type: Type of calibration ('isotonic', 'range_based', or None)
        initial_bankroll: Starting bankroll for betting simulation
        kelly_fraction: Kelly criterion fraction to use
        fixed_bet_fraction: Fixed fraction betting percentage
        max_bet_percentage: Maximum bet size as percentage of bankroll
        min_odds: Minimum odds to consider for betting
        odds_type: Type of odds to use ('open', 'close', 'average')
        max_underdog_odds: Maximum underdog odds to bet on

    Returns:
        Dictionary of model metrics
    """
    # Ensure directories exist
    ensure_directories()

    # Run the evaluation
    metrics = evaluate_model_by_name(
        model_name=model_name,
        model_directory=model_directory,
        val_data_path=val_data_path,
        test_data_path=test_data_path,
        encoder_path=encoder_path,
        display_columns=display_columns,
        output_dir=output_dir,
        use_calibration=use_calibration,
        calibration_type=calibration_type,
        initial_bankroll=initial_bankroll,
        kelly_fraction=kelly_fraction,
        fixed_bet_fraction=fixed_bet_fraction,
        max_bet_percentage=max_bet_percentage,
        min_odds=min_odds,
        odds_type=odds_type,
        max_underdog_odds=max_underdog_odds
    )

    return metrics


def run_all_models_evaluation(model_directory, val_data_path, test_data_path, encoder_path,
                              display_columns, output_dir, use_calibration=True,
                              calibration_type='isotonic', initial_bankroll=10000,
                              kelly_fraction=0.5, fixed_bet_fraction=0.1, max_bet_percentage=0.1,
                              min_odds=-300, odds_type='close', max_underdog_odds=200,
                              output_file='model_comparison.csv'):
    """
    Run evaluation for all models in a directory

    Args:
        model_directory: Directory containing model files
        val_data_path: Path to validation data
        test_data_path: Path to test data
        encoder_path: Path to encoder file
        display_columns: Columns to display in output
        output_dir: Directory for output files
        use_calibration: Whether to calibrate model probabilities
        calibration_type: Type of calibration ('isotonic', 'range_based', or None)
        initial_bankroll: Starting bankroll for betting simulation
        kelly_fraction: Kelly criterion fraction to use
        fixed_bet_fraction: Fixed fraction betting percentage
        max_bet_percentage: Maximum bet size as percentage of bankroll
        min_odds: Minimum odds to consider for betting
        odds_type: Type of odds to use ('open', 'close', 'average')
        max_underdog_odds: Maximum underdog odds to bet on
        output_file: Path to save comparison CSV

    Returns:
        DataFrame with metrics for all models
    """
    # Ensure directories exist
    ensure_directories()

    # Run the evaluation
    metrics_df = evaluate_all_models(
        model_directory=model_directory,
        val_data_path=val_data_path,
        test_data_path=test_data_path,
        encoder_path=encoder_path,
        display_columns=display_columns,
        output_dir=output_dir,
        use_calibration=use_calibration,
        calibration_type=calibration_type,
        initial_bankroll=initial_bankroll,
        kelly_fraction=kelly_fraction,
        fixed_bet_fraction=fixed_bet_fraction,
        max_bet_percentage=max_bet_percentage,
        min_odds=min_odds,
        odds_type=odds_type,
        max_underdog_odds=max_underdog_odds,
        output_file=output_file
    )

    return metrics_df


def main(manual_threshold=None, use_calibration=True, calibration_type='isotonic',
         initial_bankroll=10000, kelly_fraction=0.5, fixed_bet_fraction=0.1,
         max_bet_percentage=0.1, min_odds=-300, max_underdog_odds=200,
         use_ensemble=True, odds_type='close',
         evaluate_directory=False, model_directory=None, single_model_name=None):
    """
    Main function with options to evaluate all models in a directory or a single model by name.

    Args:
        manual_threshold: Confidence threshold for betting (if None, uses 0.5)
        use_calibration: Whether to calibrate model probabilities
        calibration_type: Type of calibration ('isotonic', 'range_based', or None)
        initial_bankroll: Initial bankroll for betting simulation
        kelly_fraction: Kelly criterion fraction
        fixed_bet_fraction: Fixed fraction betting percentage
        max_bet_percentage: Maximum bet size as percentage of bankroll
        min_odds: Minimum odds to place a bet
        max_underdog_odds: Maximum underdog odds to bet on
        use_ensemble: Whether to use ensemble of models
        odds_type: Type of odds to use ('open', 'close', 'average')
        evaluate_directory: Whether to evaluate all models in directory
        model_directory: Directory containing model files to evaluate
        single_model_name: Name of a specific model to evaluate (overrides evaluate_directory)
    """
    # Apply default threshold if not specified
    if manual_threshold is None:
        manual_threshold = 0.5

    # Use default model directory if not specified
    if model_directory is None:
        model_directory = DATA_PATHS['model_base_path']

    # Ensure directories exist
    ensure_directories()

    # Evaluate a single model by name if provided
    if single_model_name is not None:
        run_single_model_evaluation(
            model_name=single_model_name,
            model_directory=model_directory,
            val_data_path=DATA_PATHS['val_data'],
            test_data_path=DATA_PATHS['test_data'],
            encoder_path=DATA_PATHS['encoder_path'],
            display_columns=DISPLAY_COLUMNS,
            output_dir=OUTPUT_DIR,
            use_calibration=use_calibration,
            calibration_type=calibration_type,
            initial_bankroll=initial_bankroll,
            kelly_fraction=kelly_fraction,
            fixed_bet_fraction=fixed_bet_fraction,
            max_bet_percentage=max_bet_percentage,
            min_odds=min_odds,
            odds_type=odds_type,
            max_underdog_odds=max_underdog_odds
        )
        return

    # Evaluate all models in directory
    elif evaluate_directory:
        metrics_df = run_all_models_evaluation(
            model_directory=model_directory,
            val_data_path=DATA_PATHS['val_data'],
            test_data_path=DATA_PATHS['test_data'],
            encoder_path=DATA_PATHS['encoder_path'],
            display_columns=DISPLAY_COLUMNS,
            output_dir=OUTPUT_DIR,
            use_calibration=use_calibration,
            calibration_type=calibration_type,
            initial_bankroll=initial_bankroll,
            kelly_fraction=kelly_fraction,
            fixed_bet_fraction=fixed_bet_fraction,
            max_bet_percentage=max_bet_percentage,
            min_odds=min_odds,
            odds_type=odds_type,
            max_underdog_odds=max_underdog_odds,
            output_file='model_comparison.csv'
        )

        # Print summary of best model
        if not metrics_df.empty:
            best_model = metrics_df.iloc[0]
            print(f"\nBest model: {best_model['model_name']}")
            print(f"Kelly ROI: {best_model['kelly_roi']:.2f}%")
            print(f"Confidence threshold: {best_model['confidence_threshold']:.2f}")
            print(f"Calibration error: {best_model['calibration_error']:.4f}")
            print(f"Calibration tendency: {best_model['calibration_tendency']}")

    else:
        # Original functionality: evaluate an ensemble of models
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        # Set constants
        INITIAL_BANKROLL = initial_bankroll
        KELLY_FRACTION = kelly_fraction
        FIXED_BET_FRACTION = fixed_bet_fraction
        MAX_BET_PERCENTAGE = max_bet_percentage

        # Load and preprocess data
        X_val, y_val, X_test, y_test, test_data_with_display, encoder = prepare_datasets(
            DATA_PATHS['val_data'],
            DATA_PATHS['test_data'],
            DATA_PATHS['encoder_path'],
            DISPLAY_COLUMNS
        )

        # Determine which models to use (either ensemble or single)
        models = []
        model_names = []

        if use_ensemble:
            for model_file in MODEL_FILES:
                model_path = os.path.abspath(f"{model_directory}/{model_file}")
                models.append(load_model(model_path, 'xgboost'))
                model_names.append(os.path.splitext(model_file)[0])
        else:
            model_file = MODEL_FILES[0]  # Use the first model if not ensemble
            model_path = os.path.abspath(f"{model_directory}/{model_file}")
            models.append(load_model(model_path, 'xgboost'))
            model_names = [os.path.splitext(model_file)[0]]

        # Ensure consistent feature ordering
        expected_features = models[0].get_booster().feature_names
        X_val = X_val.reindex(columns=expected_features)
        X_test = X_test.reindex(columns=expected_features)

        # Apply calibration based on the selected type
        if use_calibration:
            if calibration_type == 'range_based':
                print(f"Applying range-based calibration...")
                # Apply range-based calibration
                y_pred_proba_list, calibrators = calibrate_predictions_with_range_method(
                    models, X_val, y_val, X_test, ranges=RANGE_CALIBRATION_RANGES
                )
            else:
                # Use standard scikit-learn calibration (isotonic by default)
                calibration_type = 'isotonic'  # Default to isotonic if not range_based
                print(f"Applying {calibration_type} calibration...")

                # Calibrate models using the validation data
                calibrated_models = []
                for model in models:
                    calibrated_model = CalibratedClassifierCV(model, cv='prefit', method=calibration_type)
                    calibrated_model.fit(X_val, y_val)
                    calibrated_models.append(calibrated_model)

                # Generate predictions row by row to ensure determinism
                y_pred_proba_list = []
                for _ in range(len(calibrated_models)):
                    y_pred_proba_list.append([])

                # Process each row individually
                for i in range(len(X_test)):
                    # Extract a single row as a DataFrame (maintaining 2D structure)
                    row_features = X_test.iloc[[i]]

                    # Get predictions from each model for this row
                    for j, model in enumerate(calibrated_models):
                        y_pred_proba = model.predict_proba(row_features)
                        y_pred_proba_list[j].append(y_pred_proba[0])  # Store just the probabilities

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

            # Process each row individually
            for i in range(len(X_test)):
                # Extract a single row as a DataFrame (maintaining 2D structure)
                row_features = X_test.iloc[[i]]

                # Get predictions from each model for this row
                for j, model in enumerate(models):
                    y_pred_proba = model.predict_proba(row_features)
                    y_pred_proba_list[j].append(y_pred_proba[0])  # Store just the probabilities

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
            min_odds=min_odds, use_ensemble=use_ensemble, odds_type=odds_type,
            max_underdog_odds=max_underdog_odds
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
                len(test_data_with_display), confident_predictions, correct_confident_predictions,
                fixed_total_bets, fixed_correct_bets, INITIAL_BANKROLL, fixed_final_bankroll,
                fixed_total_volume, manual_threshold, kelly_final_bankroll, kelly_total_volume,
                kelly_correct_bets, kelly_total_bets, KELLY_FRACTION, FIXED_BET_FRACTION,
                earliest_fight_date, fixed_monthly_profit, kelly_monthly_profit
            )

        # Calculate and print overall metrics using ensemble predictions
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
                # Create standard calibration curves with model names
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

            # Analyze and print calibration quality information
            analyze_calibration_quality(y_test, y_pred_proba_avg,
                                        "Ensemble model" if use_ensemble else model_names[0])

        except Exception as e:
            print(f"\n[Warning] Error generating calibration plots: {str(e)}")
            print("Continuing with analysis without calibration plots.")

        # Restore stdout and print final output
        sys.stdout = old_stdout
        output = mystdout.getvalue()
        console = Console(width=100)
        main_panel = Panel(
            output,
            title=f"MMA Betting Analysis (Odds: {odds_type.capitalize()}, Calibration: {calibration_type.capitalize()})",
            border_style="bold magenta",
            expand=True,
        )
        console.print(main_panel)
"""
Metrics calculation and reporting functions
"""
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import calibration_curve
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns


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


def print_betting_results(total_fights, confident_predictions, correct_confident_predictions,
                          fixed_total_bets, fixed_correct_bets, initial_bankroll, fixed_final_bankroll,
                          fixed_total_volume, confidence_threshold, kelly_final_bankroll, kelly_total_volume,
                          kelly_correct_bets, kelly_total_bets, kelly_fraction, fixed_bet_fraction,
                          earliest_fight_date, fixed_monthly_profit, kelly_monthly_profit):
    """
    Print comprehensive betting results summary

    Args:
        Various betting performance metrics
    """
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

    console.print(Panel(f"Confidence threshold: {confidence_threshold:.4f}\n"
                        f"Kelly ROI: {kelly_roi:.2f}%",
                        title="Parameters"))

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


def write_metrics_to_csv(metrics_list, output_file='model_comparison.csv'):
    """
    Write model metrics to a CSV file

    Args:
        metrics_list: List of dictionaries containing model metrics
        output_file: Path to output CSV file

    Returns:
        DataFrame containing metrics sorted by Kelly ROI
    """
    # Convert to DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Sort by Kelly ROI (descending)
    metrics_df = metrics_df.sort_values('kelly_roi', ascending=False)

    # Write to CSV
    metrics_df.to_csv(output_file, index=False)
    print(f"Metrics written to {output_file}")

    return metrics_df


def analyze_calibration_quality(y_test, y_pred_proba, model_label="Model"):
    """
    Analyze and print information about model calibration quality

    Args:
        y_test: True labels
        y_pred_proba: Predicted probabilities (2D array with class probabilities)
        model_label: Label for the model in output

    Returns:
        Dictionary with calibration metrics
    """
    # Calculate calibration curve
    if len(y_pred_proba.shape) > 1:
        # Extract positive class probabilities
        y_pred_prob_pos = y_pred_proba[:, 1]
    else:
        # Assume already extracted
        y_pred_prob_pos = y_pred_proba

    prob_true, prob_pred = calibration_curve(y_test, y_pred_prob_pos, n_bins=10)

    # Calculate calibration error
    calibration_error = np.mean(np.abs(prob_true - prob_pred))

    # Calculate calibration bias
    bias = np.mean(prob_true - prob_pred)

    # Determine calibration tendency
    if bias > 0.01:
        tendency = "Under-confident"
        implication = "Bets are smaller than optimal"
    elif bias < -0.01:
        tendency = "Over-confident"
        implication = "Bets are larger than optimal"
    else:
        tendency = "Well-calibrated"
        implication = "Optimal bet sizing"

    # Print analysis
    print(f"\nAverage calibration error for {model_label}: {calibration_error:.4f}")
    print(f"Model tendency: {tendency} (bias: {bias:.4f})")
    print(f"Betting implication: {implication}")

    return {
        "calibration_error": calibration_error,
        "calibration_bias": bias,
        "calibration_tendency": tendency,
        "betting_implication": implication
    }
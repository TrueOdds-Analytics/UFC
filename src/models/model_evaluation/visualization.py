"""
Visualization functions for MMA betting model evaluation
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from calibrators import RangeBasedCalibrator


def create_and_save_calibration_curves(y_test, y_pred_proba_list, use_ensemble=True,
                                       output_dir='outputs/calibration_plots',
                                       calibration_type='uncalibrated', model_names=None):
    """
    Create and save calibration curve plots for the model(s)

    Args:
        y_test: True labels
        y_pred_proba_list: List of prediction probabilities from models
        use_ensemble: Whether to use ensemble predictions
        output_dir: Directory to save the plot files
        calibration_type: Type of calibration ('uncalibrated', 'isotonic', 'range_based')
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

            # Calculate calibration error
            cal_error = np.mean(np.abs(prob_true - prob_pred))

            plt.plot(prob_pred, prob_true, 's-', label=f'{model_names[i]} (Error: {cal_error:.4f})')

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

        # Calculate calibration error
        cal_error = np.mean(np.abs(prob_true - prob_pred))

        plt.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8,
                 label=f'Ensemble model (Calibration Error: {cal_error:.4f})')

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

        # Calculate calibration error
        cal_error = np.mean(np.abs(prob_true - prob_pred))

        plt.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8,
                 label=f'{model_names[0]} (Calibration Error: {cal_error:.4f})')

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
              'Points on the diagonal: Perfect calibration\n' + \
              f'Calibration Error: {cal_error:.4f} (Lower is better)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.gcf().text(0.5, 0.02, textstr, fontsize=12, bbox=props,
                   ha='center', va='center')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for text box
    plt.savefig(filename, dpi=100)
    plt.close()
    plot_files['main_model'] = filename

    return plot_files


def create_reliability_diagram(y_test, y_pred_proba_list, use_ensemble=True, output_dir='outputs/calibration_plots',
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

    # Calculate calibration error
    cal_error = np.mean(np.abs(prob_true - prob_pred))

    # Calculate direction (over or under confidence)
    bias = np.mean(prob_true - prob_pred)
    bias_type = "Under-confident" if bias > 0 else "Over-confident" if bias < 0 else "Well-calibrated"

    ax2.plot(prob_pred, prob_true, 's-', label=f'Calibration curve (Error: {cal_error:.4f})',
             color='blue', linewidth=2)
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
    ax2.set_title(f'Reliability Diagram (Calibration Curve) - {bias_type}')
    ax2.legend(loc='best')
    ax2.grid(True)

    # Add interpretation text box
    textstr = 'Interpretation:\n' + \
              'Green areas: Model is under-confident (true probability > predicted)\n' + \
              'Red areas: Model is over-confident (predicted > true probability)\n' + \
              'For Kelly betting: Under-confidence leads to smaller bets than optimal\n' + \
              'Over-confidence leads to larger bets than optimal\n\n' + \
              f'Calibration Error: {cal_error:.4f} (Lower is better)\n' + \
              f'Bias: {bias:.4f} ({bias_type})'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.5, 0.02, textstr, fontsize=10, bbox=props,
             ha='center', va='center')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)  # Make room for text box
    plt.savefig(filename, dpi=100)
    plt.close()

    return filename


def create_range_based_reliability_diagram(y_test, y_pred_proba_list, use_ensemble=True,
                                           output_dir='outputs/calibration_plots', ranges=None,
                                           model_names=None):
    """
    Create a reliability diagram comparing original vs range-based calibration

    Args:
        y_test: True labels
        y_pred_proba_list: List of prediction probabilities from models
        use_ensemble: Whether to use ensemble predictions
        output_dir: Directory to save the plot file
        ranges: List of probability thresholds for range-based calibration
        model_names: List of model names for plots

    Returns:
        Tuple of (filename, calibrator)
    """
    # Create the base output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    calibration_dir = os.path.join(output_dir, 'range_based')
    os.makedirs(calibration_dir, exist_ok=True)

    # Use default model names if none provided
    if model_names is None:
        model_names = [f"Model_{i + 1}" for i in range(len(y_pred_proba_list))]

    # Use default ranges if none provided
    if ranges is None:
        ranges = [0.25, 0.5, 0.75]

    # Get prediction probabilities
    if use_ensemble:
        y_pred_proba = np.mean([proba for proba in y_pred_proba_list], axis=0)
        title = f"Ensemble Model with Range-Based Calibration"
        filename = os.path.join(calibration_dir, f'ensemble_range_calibration.png')
    else:
        y_pred_proba = y_pred_proba_list[0]
        title = f"{model_names[0]} with Range-Based Calibration"
        filename = os.path.join(calibration_dir, f'{model_names[0]}_range_calibration.png')

    # Extract positive class probabilities
    y_pred_prob_pos = y_pred_proba[:, 1]

    # Create a range-based calibrator
    calibrator = RangeBasedCalibrator(ranges=ranges, method='isotonic')

    # Split data into calibration and evaluation sets (70/30 split)
    n_samples = len(y_test)
    n_cal = int(n_samples * 0.3)

    # Use a portion for fitting the calibrator
    calibrator.fit(y_pred_prob_pos[:n_cal], y_test[:n_cal])

    # Apply to all data
    calibrated_probs = calibrator.transform(y_pred_prob_pos)

    # Create figure with two subplots: histogram and calibration curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12),
                                   gridspec_kw={'height_ratios': [1, 3]})

    # Top plot: histogram of prediction distribution
    ax1.hist(y_pred_prob_pos, bins=20, range=(0, 1), histtype='step',
             lw=2, color='blue', density=True, label='Original')
    ax1.hist(calibrated_probs, bins=20, range=(0, 1), histtype='step',
             lw=2, color='red', density=True, label='Calibrated')

    ax1.set_ylabel('Density')
    ax1.set_xlim([0, 1])
    ax1.set_title('Distribution of Predicted Probabilities')
    ax1.legend(loc='best')

    # Bottom plot: reliability diagram
    # Original calibration curve
    prob_true_orig, prob_pred_orig = calibration_curve(y_test, y_pred_prob_pos, n_bins=10)
    cal_error_orig = np.mean(np.abs(prob_true_orig - prob_pred_orig))

    # Range-calibrated curve
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, calibrated_probs, n_bins=10)
    cal_error_cal = np.mean(np.abs(prob_true_cal - prob_pred_cal))

    # Calculate direction (over or under confidence)
    orig_bias = np.mean(prob_true_orig - prob_pred_orig)
    cal_bias = np.mean(prob_true_cal - prob_pred_cal)

    orig_bias_type = "Under-confident" if orig_bias > 0 else "Over-confident" if orig_bias < 0 else "Well-calibrated"
    cal_bias_type = "Under-confident" if cal_bias > 0 else "Over-confident" if cal_bias < 0 else "Well-calibrated"

    # Plot reference diagonal
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

    # Plot original curve
    ax2.plot(prob_pred_orig, prob_true_orig, 's-', color='blue',
             label=f'Original (Error: {cal_error_orig:.4f}, {orig_bias_type})')

    # Plot calibrated curve
    ax2.plot(prob_pred_cal, prob_true_cal, 'o-', color='red',
             label=f'Range Calibrated (Error: {cal_error_cal:.4f}, {cal_bias_type})')

    # Mark range boundaries
    for threshold in ranges:
        ax2.axvline(x=threshold, color='gray', linestyle=':', alpha=0.7)

    # Add annotations for original points
    for i, (x, y) in enumerate(zip(prob_pred_orig, prob_true_orig)):
        ax2.annotate(f'({x:.2f}, {y:.2f})', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=8, color='blue')

    # Add annotations for calibrated points
    for i, (x, y) in enumerate(zip(prob_pred_cal, prob_true_cal)):
        ax2.annotate(f'({x:.2f}, {y:.2f})', (x, y), textcoords="offset points",
                     xytext=(0, -15), ha='center', fontsize=8, color='red')

    ax2.set_xlabel('Mean predicted probability')
    ax2.set_ylabel('Fraction of positives (true probability)')
    ax2.set_title('Reliability Diagram (Original vs Range-Based Calibration)')
    ax2.legend(loc='best')
    ax2.grid(True)

    # Calculate improvement percentage
    improvement = ((cal_error_orig - cal_error_cal) / cal_error_orig) * 100

    # Add interpretation text box
    textstr = 'Interpretation:\n' + \
              'Points above diagonal: Model is under-confident (true probability > predicted)\n' + \
              'Points below diagonal: Model is over-confident (predicted > true probability)\n' + \
              'For Kelly betting: Under-confidence leads to smaller bets than optimal\n' + \
              'Over-confidence leads to larger bets than optimal\n\n' + \
              f'Original Calibration Error: {cal_error_orig:.4f} ({orig_bias_type})\n' + \
              f'Range-Based Calibration Error: {cal_error_cal:.4f} ({cal_bias_type})\n' + \
              f'Improvement: {improvement:.1f}%'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fig.text(0.5, 0.02, textstr, fontsize=10, bbox=props,
             ha='center', va='center')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)  # Make room for text box
    plt.savefig(filename, dpi=100)
    plt.close()

    return filename, calibrator
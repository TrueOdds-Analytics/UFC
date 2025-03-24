"""
Calibration classes for model probability calibration
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class RangeBasedCalibrator:
    """
    A custom calibrator that applies different calibration strategies to
    different probability ranges for improved calibration.
    """

    def __init__(self, ranges=None, method='isotonic', out_of_bounds='clip'):
        """
        Initialize the range-based calibrator.

        Parameters:
        -----------
        ranges : list of float, default=[0.33, 0.67]
            The probability thresholds that divide the regions.
            Default creates three regions: low (<0.33), medium (0.33-0.67), high (>0.67)

        method : str, default='isotonic'
            Calibration method to use for each region:
            - 'isotonic': IsotonicRegression (non-parametric, monotonic)
            - 'sigmoid': Platt scaling (logistic regression)

        out_of_bounds : str, default='clip'
            How to handle predictions outside the range of training data
        """
        self.ranges = [0.33, 0.67] if ranges is None else ranges
        self.method = method
        self.out_of_bounds = out_of_bounds
        self.calibrators = {}
        self.min_probs = {}
        self.max_probs = {}

    def _get_region(self, prob):
        """Determine which region a probability belongs to"""
        if prob < self.ranges[0]:
            return 'low'
        elif prob > self.ranges[-1]:
            return 'high'
        else:
            for i in range(len(self.ranges) - 1):
                if self.ranges[i] <= prob <= self.ranges[i + 1]:
                    return f'mid_{i}'
            return 'mid_0'  # Default if no match (shouldn't happen)

    def fit(self, probs, y_true):
        """
        Fit calibration models for each probability region.

        Parameters:
        -----------
        probs : array-like of shape (n_samples,)
            Predicted probabilities for the positive class

        y_true : array-like of shape (n_samples,)
            True binary labels

        Returns:
        --------
        self
        """
        all_regions = ['low'] + [f'mid_{i}' for i in range(len(self.ranges) - 1)] + ['high']

        # Create masks for each region
        region_masks = {}
        region_masks['low'] = probs < self.ranges[0]
        region_masks['high'] = probs > self.ranges[-1]

        for i in range(len(self.ranges) - 1):
            region_masks[f'mid_{i}'] = (probs >= self.ranges[i]) & (probs <= self.ranges[i + 1])

        # Fit calibrators for each region
        for region in all_regions:
            mask = region_masks[region]

            # Skip regions with too few samples
            if np.sum(mask) < 10:  # Minimum number of samples needed for reliable calibration
                print(f"Warning: Not enough samples in {region} region. Using adjacent region.")
                continue

            region_probs = probs[mask]
            region_y = y_true[mask]

            # Store min and max probs for handling out-of-bounds
            self.min_probs[region] = np.min(region_probs) if len(region_probs) > 0 else 0
            self.max_probs[region] = np.max(region_probs) if len(region_probs) > 0 else 1

            if self.method == 'isotonic':
                self.calibrators[region] = IsotonicRegression(out_of_bounds=self.out_of_bounds)
                self.calibrators[region].fit(region_probs, region_y)
            elif self.method == 'sigmoid':
                # Simple sigmoid calibration
                lr = LogisticRegression(C=1.0)
                lr.fit(region_probs.reshape(-1, 1), region_y)
                self.calibrators[region] = lr
            else:
                raise ValueError(f"Unknown calibration method: {self.method}")

        return self

    def transform(self, probs):
        """
        Apply calibration transformation to new probabilities.

        Parameters:
        -----------
        probs : array-like of shape (n_samples,)
            Predicted probabilities to calibrate

        Returns:
        --------
        calibrated_probs : array-like of shape (n_samples,)
            Calibrated probabilities
        """
        calibrated_probs = np.zeros_like(probs)

        for i, prob in enumerate(probs):
            region = self._get_region(prob)

            # If calibrator for this region doesn't exist, use the closest one
            if region not in self.calibrators:
                # Find closest region with a calibrator
                available_regions = list(self.calibrators.keys())
                if len(available_regions) == 0:
                    # No calibrators available, return original
                    calibrated_probs[i] = prob
                    continue

                # Define region midpoints for finding closest
                region_values = {
                    'low': self.ranges[0] / 2,
                    'high': (1 + self.ranges[-1]) / 2
                }

                for j in range(len(self.ranges) - 1):
                    region_values[f'mid_{j}'] = (self.ranges[j] + self.ranges[j + 1]) / 2

                # Find closest region
                current_region_value = region_values.get(region, 0.5)
                closest_region = min(
                    available_regions,
                    key=lambda r: abs(region_values.get(r, 0.5) - current_region_value)
                )
                region = closest_region

            # Apply calibration for the region
            if self.method == 'isotonic':
                # IsotonicRegression handles out_of_bounds internally
                calibrated_probs[i] = self.calibrators[region].predict([prob])[0]
            elif self.method == 'sigmoid':
                # For sigmoid, we need to handle out-of-bounds ourselves
                if self.out_of_bounds == 'clip':
                    if prob < self.min_probs[region]:
                        prob = self.min_probs[region]
                    elif prob > self.max_probs[region]:
                        prob = self.max_probs[region]
                # Apply logistic regression
                calibrated_probs[i] = self.calibrators[region].predict_proba([[prob]])[0, 1]

        return calibrated_probs

    def fit_transform(self, probs, y_true):
        """
        Fit calibration models and then transform the input probabilities.

        Parameters:
        -----------
        probs : array-like of shape (n_samples,)
            Predicted probabilities for the positive class

        y_true : array-like of shape (n_samples,)
            True binary labels

        Returns:
        --------
        calibrated_probs : array-like of shape (n_samples,)
            Calibrated probabilities
        """
        return self.fit(probs, y_true).transform(probs)

    def plot_calibration(self, probs, y_true, n_bins=10, title='Range-Based Calibration', figsize=(10, 8)):
        """
        Plot calibration curve before and after range-based calibration.

        Parameters:
        -----------
        probs : array-like
            Original uncalibrated probabilities

        y_true : array-like
            True binary labels

        n_bins : int, default=10
            Number of bins for calibration curve

        title : str, default='Range-Based Calibration'
            Plot title

        figsize : tuple, default=(10, 8)
            Figure size

        Returns:
        --------
        fig : matplotlib.Figure
            The figure object
        """
        calibrated_probs = self.transform(probs)

        # Compute calibration curves
        prob_true_orig, prob_pred_orig = calibration_curve(y_true, probs, n_bins=n_bins)
        prob_true_cal, prob_pred_cal = calibration_curve(y_true, calibrated_probs, n_bins=n_bins)

        # Calculate calibration errors
        cal_error_orig = np.mean(np.abs(prob_true_orig - prob_pred_orig))
        cal_error_cal = np.mean(np.abs(prob_true_cal - prob_pred_cal))

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Plot reference line for perfect calibration
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')

        # Plot original calibration curve
        ax.plot(prob_pred_orig, prob_true_orig, 's-', label=f'Original (Error: {cal_error_orig:.4f})')

        # Plot calibrated curve
        ax.plot(prob_pred_cal, prob_true_cal, 'o-', label=f'Range-Calibrated (Error: {cal_error_cal:.4f})')

        # Mark range boundaries
        for threshold in self.ranges:
            ax.axvline(x=threshold, color='gray', linestyle=':', alpha=0.7)

        # Add annotations
        for i, (x, y) in enumerate(zip(prob_pred_cal, prob_true_cal)):
            ax.annotate(f'({x:.2f}, {y:.2f})', (x, y), textcoords="offset points",
                        xytext=(0, 5), ha='center', fontsize=9)

        # Set labels and title
        ax.set_xlabel('Mean predicted probability')
        ax.set_ylabel('Fraction of positives')
        ax.set_title(title)

        # Add legend and grid
        ax.legend(loc='best')
        ax.grid(True)

        # Add interpretation text
        textstr = 'Interpretation:\n' + \
                  'Points above the diagonal: Model is under-confident\n' + \
                  'Points below the diagonal: Model is over-confident\n' + \
                  'Points on the diagonal: Perfect calibration\n\n' + \
                  f'Original Calibration Error: {cal_error_orig:.4f}\n' + \
                  f'Range-Based Calibration Error: {cal_error_cal:.4f}\n' + \
                  f'Improvement: {(cal_error_orig - cal_error_cal) / cal_error_orig:.1%}'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', bbox=props)

        plt.tight_layout()
        return fig


def calibrate_predictions_with_range_method(models, X_val, y_val, X_test, ranges=None):
    """
    Apply range-based calibration to model predictions

    Parameters:
    -----------
    models : list of trained models
        Models to calibrate

    X_val : array-like
        Validation features for fitting calibrators

    y_val : array-like
        Validation target values

    X_test : array-like
        Test features for generating predictions

    ranges : list of float
        Probability thresholds that divide calibration regions

    Returns:
    --------
    y_pred_proba_list : list of calibrated prediction probability arrays
    calibrators : list of fitted RangeBasedCalibrator objects
    """
    # Use default ranges if none provided
    if ranges is None:
        ranges = [0.25, 0.45, 0.65, 0.85]  # Creates 5 regions

    y_pred_proba_list = []
    calibrators = []

    # Process each model
    for model in models:
        # Get uncalibrated probabilities from validation data
        val_probs = model.predict_proba(X_val)[:, 1]

        # Create and fit range-based calibrator
        calibrator = RangeBasedCalibrator(ranges=ranges, method='isotonic')
        calibrator.fit(val_probs, y_val)
        calibrators.append(calibrator)

        # Generate test predictions
        test_probs_raw = model.predict_proba(X_test)

        # Apply calibration to class 1 probabilities
        test_probs_class1 = test_probs_raw[:, 1]
        calibrated_probs_class1 = calibrator.transform(test_probs_class1)

        # Reconstruct full probability array
        calibrated_probs = np.zeros_like(test_probs_raw)
        calibrated_probs[:, 1] = calibrated_probs_class1
        calibrated_probs[:, 0] = 1 - calibrated_probs_class1

        y_pred_proba_list.append(calibrated_probs)

    return y_pred_proba_list, calibrators
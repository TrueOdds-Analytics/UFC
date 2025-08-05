"""
MMA Betting Model Evaluator - Refactored and Consolidated Version

This module consolidates the betting evaluation system into a cleaner structure.
"""

import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from rich.console import Console
from rich.table import Table
from rich.panel import Panel


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BettingConfig:
    """Centralized configuration for betting evaluation"""
    # Data paths
    val_data_path: str = '../../../data/train_test/val_data.csv'
    test_data_path: str = '../../../data/train_test/test_data.csv'
    encoder_path: str = '../../../saved_models/encoders/category_encoder.pkl'
    model_base_path: str = '../../../saved_models/xgboost/jan2024-dec2025/no odds/'
    output_dir: str = '../../outputs/calibration_plots'

    # Betting parameters
    initial_bankroll: float = 10000
    confidence_threshold: float = 0.5
    kelly_fraction: float = 0.5
    fixed_bet_fraction: float = 0.1
    max_bet_percentage: float = 0.1
    min_odds: int = -300
    max_underdog_odds: int = 200

    # Model settings
    use_ensemble: bool = True
    use_calibration: bool = True
    calibration_type: str = 'isotonic'  # 'isotonic', 'range_based', or None

    # Odds settings
    odds_type: str = 'close'  # 'open', 'close', 'average'

    # Display settings
    display_columns: List[str] = field(default_factory=lambda: ['current_fight_date', 'fighter_a', 'fighter_b'])

    # Model files for ensemble
    model_files: List[str] = field(default_factory=lambda: [
        'model_0.7009_auc_diff_0.0072.json',
        'model_0.7025_auc_diff_0.0221.json',
        'model_0.7009_auc_diff_0.0005.json',
        'model_0.7057_auc_diff_0.0020.json',
        'model_0.7080_auc_diff_0.0037.json'
    ])

    def ensure_directories(self):
        """Ensure all necessary directories exist"""
        os.makedirs(os.path.dirname(self.encoder_path), exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


# ============================================================================
# Data Processing
# ============================================================================

class CategoryEncoder:
    """Handles categorical feature encoding for consistent preprocessing"""

    def __init__(self):
        self.category_mappings = {}
        self.initialized = False

    def fit(self, data: pd.DataFrame) -> 'CategoryEncoder':
        """Learn category mappings from data"""
        category_columns = [col for col in data.columns
                            if col.endswith(('fight_1', 'fight_2', 'fight_3'))]

        for col in category_columns:
            unique_values = data[col].dropna().unique()
            self.category_mappings[col] = {val: i for i, val in enumerate(unique_values)}

        self.initialized = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply categorical mappings to data"""
        if not self.initialized:
            raise ValueError("CategoryEncoder must be fit before transform")

        data_copy = data.copy()

        for col, mapping in self.category_mappings.items():
            if col in data_copy.columns:
                data_copy[col] = data_copy[col].map(mapping).fillna(-1).astype('int32')

        return data_copy

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(data).transform(data)

    def save(self, filepath: str):
        """Save encoder to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.category_mappings, f)

    @classmethod
    def load(cls, filepath: str) -> 'CategoryEncoder':
        """Load encoder from disk"""
        encoder = cls()
        with open(filepath, 'rb') as f:
            encoder.category_mappings = pickle.load(f)
        encoder.initialized = True
        return encoder


# ============================================================================
# Betting Calculations
# ============================================================================

class BettingCalculator:
    """Handles all betting-related calculations"""

    @staticmethod
    def calculate_profit(odds: float, stake: float) -> float:
        """Calculate potential profit based on American odds"""
        if odds < 0:
            return (100 / abs(odds)) * stake
        else:
            return (odds / 100) * stake

    @staticmethod
    def calculate_kelly_fraction(p: float, b: float) -> float:
        """
        Calculate Kelly criterion optimal bet size
        p: Probability of winning
        b: Decimal odds-1 (payout per unit wagered)
        """
        q = 1 - p
        return max(0, (p - (q / b)))

    @staticmethod
    def calculate_average_odds(open_odds: float, close_odds: float) -> float:
        """Calculate average between opening and closing odds"""

        def american_to_decimal(odds):
            return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1

        avg_decimal = (american_to_decimal(open_odds) + american_to_decimal(close_odds)) / 2

        if avg_decimal > 2:
            return round((avg_decimal - 1) * 100)
        else:
            return round(-100 / (avg_decimal - 1))


# ============================================================================
# Model Calibration
# ============================================================================

class RangeBasedCalibrator:
    """Custom calibrator for different probability ranges"""

    def __init__(self, ranges: List[float] = None, method: str = 'isotonic'):
        self.ranges = ranges or [0.33, 0.67]
        self.method = method
        self.calibrators = {}
        self.min_probs = {}
        self.max_probs = {}

    def _get_region(self, prob: float) -> str:
        """Determine which region a probability belongs to"""
        if prob < self.ranges[0]:
            return 'low'
        elif prob > self.ranges[-1]:
            return 'high'
        else:
            for i in range(len(self.ranges) - 1):
                if self.ranges[i] <= prob <= self.ranges[i + 1]:
                    return f'mid_{i}'
            return 'mid_0'

    def fit(self, probs: np.ndarray, y_true: np.ndarray) -> 'RangeBasedCalibrator':
        """Fit calibration models for each probability region"""
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

            if np.sum(mask) < 10:  # Skip regions with too few samples
                continue

            region_probs = probs[mask]
            region_y = y_true[mask]

            self.min_probs[region] = np.min(region_probs) if len(region_probs) > 0 else 0
            self.max_probs[region] = np.max(region_probs) if len(region_probs) > 0 else 1

            if self.method == 'isotonic':
                self.calibrators[region] = IsotonicRegression(out_of_bounds='clip')
                self.calibrators[region].fit(region_probs, region_y)
            elif self.method == 'sigmoid':
                lr = LogisticRegression(C=1.0)
                lr.fit(region_probs.reshape(-1, 1), region_y)
                self.calibrators[region] = lr

        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Apply calibration transformation"""
        calibrated_probs = np.zeros_like(probs)

        for i, prob in enumerate(probs):
            region = self._get_region(prob)

            if region not in self.calibrators:
                calibrated_probs[i] = prob
                continue

            if self.method == 'isotonic':
                calibrated_probs[i] = self.calibrators[region].predict([prob])[0]
            elif self.method == 'sigmoid':
                calibrated_probs[i] = self.calibrators[region].predict_proba([[prob]])[0, 1]

        return calibrated_probs


# ============================================================================
# Main Evaluator Class
# ============================================================================

class MMABettingEvaluator:
    """Main class for evaluating MMA betting models"""

    def __init__(self, config: BettingConfig = None):
        self.config = config or BettingConfig()
        self.config.ensure_directories()
        self.console = Console()
        self.encoder = None
        self.models = []
        self.betting_calc = BettingCalculator()

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and preprocess training and test data"""
        # Load data
        val_data = pd.read_csv(self.config.val_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Separate features and targets
        y_val = val_data['winner']
        y_test = test_data['winner']

        # Handle encoder
        if os.path.exists(self.config.encoder_path):
            self.encoder = CategoryEncoder.load(self.config.encoder_path)
            X_val = self.encoder.transform(val_data.drop(['winner'] + self.config.display_columns, axis=1))
        else:
            self.encoder = CategoryEncoder()
            X_val = self.encoder.fit_transform(val_data.drop(['winner'] + self.config.display_columns, axis=1))
            self.encoder.save(self.config.encoder_path)

        X_test = self.encoder.transform(test_data.drop(['winner'] + self.config.display_columns, axis=1))

        # Prepare test data with display columns
        test_data_with_display = pd.concat([X_test, test_data[self.config.display_columns], y_test], axis=1)
        X_test = test_data_with_display.drop(self.config.display_columns + ['winner'], axis=1)
        y_test = test_data_with_display['winner']

        return X_val, y_val, X_test, y_test, test_data_with_display

    def load_models(self, X_val: pd.DataFrame) -> List[Any]:
        """Load models for evaluation"""
        models = []

        if self.config.use_ensemble:
            for model_file in self.config.model_files:
                model_path = os.path.join(self.config.model_base_path, model_file)
                models.append(self._load_single_model(model_path))
        else:
            model_path = os.path.join(self.config.model_base_path, self.config.model_files[0])
            models.append(self._load_single_model(model_path))

        # Ensure consistent feature ordering
        expected_features = models[0].get_booster().feature_names
        X_val = X_val.reindex(columns=expected_features)

        self.models = models
        return models

    def _load_single_model(self, model_path: str) -> xgb.XGBClassifier:
        """Load a single XGBoost model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = xgb.XGBClassifier(enable_categorical=True)
        model.load_model(model_path)
        return model

    def generate_predictions(self, X_val: pd.DataFrame, y_val: pd.Series,
                             X_test: pd.DataFrame) -> List[np.ndarray]:
        """Generate predictions with optional calibration"""
        if self.config.use_calibration:
            if self.config.calibration_type == 'range_based':
                return self._generate_range_calibrated_predictions(X_val, y_val, X_test)
            else:
                return self._generate_isotonic_calibrated_predictions(X_val, y_val, X_test)
        else:
            return self._generate_uncalibrated_predictions(X_test)

    def _generate_uncalibrated_predictions(self, X_test: pd.DataFrame) -> List[np.ndarray]:
        """Generate uncalibrated predictions"""
        predictions = []
        for model in self.models:
            pred_proba = model.predict_proba(X_test)
            predictions.append(pred_proba)
        return predictions

    def _generate_isotonic_calibrated_predictions(self, X_val: pd.DataFrame, y_val: pd.Series,
                                                  X_test: pd.DataFrame) -> List[np.ndarray]:
        """Generate isotonic calibrated predictions"""
        predictions = []
        for model in self.models:
            calibrated_model = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
            calibrated_model.fit(X_val, y_val)
            pred_proba = calibrated_model.predict_proba(X_test)
            predictions.append(pred_proba)
        return predictions

    def _generate_range_calibrated_predictions(self, X_val: pd.DataFrame, y_val: pd.Series,
                                               X_test: pd.DataFrame) -> List[np.ndarray]:
        """Generate range-based calibrated predictions"""
        predictions = []
        ranges = [0.25, 0.45, 0.65, 0.85]

        for model in self.models:
            val_probs = model.predict_proba(X_val)[:, 1]
            test_probs = model.predict_proba(X_test)

            calibrator = RangeBasedCalibrator(ranges=ranges)
            calibrator.fit(val_probs, y_val)

            calibrated_probs = np.zeros_like(test_probs)
            calibrated_probs[:, 1] = calibrator.transform(test_probs[:, 1])
            calibrated_probs[:, 0] = 1 - calibrated_probs[:, 1]

            predictions.append(calibrated_probs)

        return predictions

    def evaluate_bets(self, y_test: pd.Series, predictions: List[np.ndarray],
                      test_data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate betting performance"""
        # Initialize tracking variables
        fixed_bankroll = self.config.initial_bankroll
        kelly_bankroll = self.config.initial_bankroll
        fixed_correct_bets = 0
        kelly_correct_bets = 0
        fixed_total_bets = 0
        kelly_total_bets = 0
        confident_predictions = 0
        correct_confident_predictions = 0

        daily_fixed_bankrolls = {}
        daily_kelly_bankrolls = {}

        # Process fights chronologically
        for date in sorted(test_data['current_fight_date'].unique()):
            date_fights = test_data[test_data['current_fight_date'] == date]

            for idx in date_fights.index:
                # Get predictions for this fight
                if self.config.use_ensemble:
                    y_pred_proba_avg = np.mean([pred[idx] for pred in predictions], axis=0)
                else:
                    y_pred_proba_avg = predictions[0][idx]

                # Determine predicted winner
                predicted_winner_idx = np.argmax(y_pred_proba_avg)
                winning_probability = y_pred_proba_avg[predicted_winner_idx]

                confident_predictions += 1
                if predicted_winner_idx == y_test.loc[idx]:
                    correct_confident_predictions += 1

                # Only bet if confidence meets threshold
                if winning_probability >= self.config.confidence_threshold:
                    # Get odds and calculate bet sizes
                    row = test_data.loc[idx]

                    # Determine odds based on predicted winner
                    if predicted_winner_idx == 1:
                        odds = self._get_odds(row, 'a')
                    else:
                        odds = self._get_odds(row, 'b')

                    # Skip if odds outside range
                    if odds < self.config.min_odds or odds > self.config.max_underdog_odds:
                        continue

                    # Calculate fixed bet
                    fixed_stake = min(
                        fixed_bankroll * self.config.fixed_bet_fraction,
                        fixed_bankroll * self.config.max_bet_percentage
                    )

                    # Calculate Kelly bet
                    b = odds / 100 if odds > 0 else 100 / abs(odds)
                    kelly_fraction = self.betting_calc.calculate_kelly_fraction(winning_probability, b)
                    kelly_stake = min(
                        kelly_bankroll * kelly_fraction * self.config.kelly_fraction,
                        kelly_bankroll * self.config.max_bet_percentage
                    )

                    # Process bets
                    if fixed_stake > 0:
                        fixed_total_bets += 1
                        profit = self.betting_calc.calculate_profit(odds, fixed_stake)

                        if predicted_winner_idx == y_test.loc[idx]:
                            fixed_bankroll += profit
                            fixed_correct_bets += 1
                        else:
                            fixed_bankroll -= fixed_stake

                    if kelly_stake > 0:
                        kelly_total_bets += 1
                        profit = self.betting_calc.calculate_profit(odds, kelly_stake)

                        if predicted_winner_idx == y_test.loc[idx]:
                            kelly_bankroll += profit
                            kelly_correct_bets += 1
                        else:
                            kelly_bankroll -= kelly_stake

            # Track daily bankrolls
            daily_fixed_bankrolls[date] = fixed_bankroll
            daily_kelly_bankrolls[date] = kelly_bankroll

        return {
            'fixed_final_bankroll': fixed_bankroll,
            'kelly_final_bankroll': kelly_bankroll,
            'fixed_correct_bets': fixed_correct_bets,
            'kelly_correct_bets': kelly_correct_bets,
            'fixed_total_bets': fixed_total_bets,
            'kelly_total_bets': kelly_total_bets,
            'confident_predictions': confident_predictions,
            'correct_confident_predictions': correct_confident_predictions,
            'daily_fixed_bankrolls': daily_fixed_bankrolls,
            'daily_kelly_bankrolls': daily_kelly_bankrolls
        }

    def _get_odds(self, row: pd.Series, fighter: str) -> float:
        """Get odds for a fighter based on configuration"""
        if fighter == 'a':
            open_odds = row['current_fight_open_odds']
            close_odds = row['current_fight_closing_odds']
        else:
            open_odds = row['current_fight_open_odds_b']
            close_odds = row['current_fight_closing_odds_b']

        if self.config.odds_type == 'open':
            return open_odds
        elif self.config.odds_type == 'close':
            return close_odds
        else:  # average
            return self.betting_calc.calculate_average_odds(open_odds, close_odds)

    def calculate_metrics(self, y_test: pd.Series, predictions: List[np.ndarray],
                          betting_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        # Get ensemble predictions
        if self.config.use_ensemble:
            y_pred_proba = np.mean(predictions, axis=0)
        else:
            y_pred_proba = predictions[0]

        y_pred = np.argmax(y_pred_proba, axis=1)

        # Calculate classification metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
        }

        # Add AUC if both classes present
        if len(np.unique(y_test)) > 1:
            metrics['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            metrics['auc'] = None

        # Calculate calibration metrics
        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba[:, 1], n_bins=10)
        metrics['calibration_error'] = np.mean(np.abs(prob_true - prob_pred))
        metrics['calibration_bias'] = np.mean(prob_true - prob_pred)

        # Determine calibration tendency
        if metrics['calibration_bias'] > 0.01:
            metrics['calibration_tendency'] = "Under-confident"
        elif metrics['calibration_bias'] < -0.01:
            metrics['calibration_tendency'] = "Over-confident"
        else:
            metrics['calibration_tendency'] = "Well-calibrated"

        # Add betting metrics
        initial = self.config.initial_bankroll
        metrics['fixed_roi'] = ((betting_results['fixed_final_bankroll'] - initial) / initial) * 100
        metrics['kelly_roi'] = ((betting_results['kelly_final_bankroll'] - initial) / initial) * 100

        if betting_results['fixed_total_bets'] > 0:
            metrics['fixed_accuracy'] = betting_results['fixed_correct_bets'] / betting_results['fixed_total_bets']
        else:
            metrics['fixed_accuracy'] = 0

        if betting_results['kelly_total_bets'] > 0:
            metrics['kelly_accuracy'] = betting_results['kelly_correct_bets'] / betting_results['kelly_total_bets']
        else:
            metrics['kelly_accuracy'] = 0

        metrics.update(betting_results)

        return metrics

    def display_results(self, metrics: Dict[str, Any]):
        """Display evaluation results in a formatted table"""
        table = Table(title="MMA Betting Model Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="magenta")

        # Classification metrics
        table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
        table.add_row("Precision", f"{metrics['precision']:.4f}")
        table.add_row("Recall", f"{metrics['recall']:.4f}")
        table.add_row("F1 Score", f"{metrics['f1_score']:.4f}")

        if metrics['auc'] is not None:
            table.add_row("AUC", f"{metrics['auc']:.4f}")

        # Calibration metrics
        table.add_row("Calibration Error", f"{metrics['calibration_error']:.4f}")
        table.add_row("Calibration Tendency", metrics['calibration_tendency'])

        # Betting metrics
        table.add_row("", "")  # Separator
        table.add_row("Initial Bankroll", f"${self.config.initial_bankroll:.2f}")
        table.add_row("", "")  # Separator

        table.add_row("Fixed Final Bankroll", f"${metrics['fixed_final_bankroll']:.2f}")
        table.add_row("Fixed ROI", f"{metrics['fixed_roi']:.2f}%")
        table.add_row("Fixed Betting Accuracy", f"{metrics['fixed_accuracy']:.2%}")
        table.add_row("Fixed Total Bets", str(metrics['fixed_total_bets']))

        table.add_row("", "")  # Separator

        table.add_row("Kelly Final Bankroll", f"${metrics['kelly_final_bankroll']:.2f}")
        table.add_row("Kelly ROI", f"{metrics['kelly_roi']:.2f}%")
        table.add_row("Kelly Betting Accuracy", f"{metrics['kelly_accuracy']:.2%}")
        table.add_row("Kelly Total Bets", str(metrics['kelly_total_bets']))

        self.console.print(table)

    def create_calibration_plot(self, y_test: pd.Series, predictions: List[np.ndarray],
                                save_path: str = None):
        """Create and save calibration curve plot"""
        if self.config.use_ensemble:
            y_pred_proba = np.mean(predictions, axis=0)
            title = "Ensemble Model Calibration Curve"
        else:
            y_pred_proba = predictions[0]
            title = "Model Calibration Curve"

        prob_true, prob_pred = calibration_curve(y_test, y_pred_proba[:, 1], n_bins=10)
        cal_error = np.mean(np.abs(prob_true - prob_pred))

        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        plt.plot(prob_pred, prob_true, 'o-', label=f'Model (Error: {cal_error:.4f})')

        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title(f'{title} ({self.config.calibration_type})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.config.output_dir, 'calibration_curve.png'),
                        dpi=100, bbox_inches='tight')
        plt.close()

    def run(self) -> Dict[str, Any]:
        """Run the complete evaluation pipeline"""
        self.console.print("[bold cyan]Starting MMA Betting Model Evaluation[/bold cyan]")

        # Load and preprocess data
        self.console.print("Loading data...")
        X_val, y_val, X_test, y_test, test_data_with_display = self.load_data()

        # Load models
        self.console.print("Loading models...")
        self.load_models(X_val)

        # Ensure consistent feature ordering
        expected_features = self.models[0].get_booster().feature_names
        X_val = X_val.reindex(columns=expected_features)
        X_test = X_test.reindex(columns=expected_features)

        # Generate predictions
        self.console.print(f"Generating predictions with {self.config.calibration_type} calibration...")
        predictions = self.generate_predictions(X_val, y_val, X_test)

        # Evaluate betting performance
        self.console.print("Evaluating betting strategies...")
        betting_results = self.evaluate_bets(y_test, predictions, test_data_with_display)

        # Calculate comprehensive metrics
        metrics = self.calculate_metrics(y_test, predictions, betting_results)

        # Display results
        self.display_results(metrics)

        # Create calibration plot
        self.console.print("Creating calibration plot...")
        self.create_calibration_plot(y_test, predictions)

        self.console.print("[bold green]Evaluation complete![/bold green]")

        return metrics


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the betting evaluator"""
    # Create configuration (modify as needed)
    config = BettingConfig(
        use_ensemble=False,
        use_calibration=True,
        calibration_type='isotonic',
        initial_bankroll=10000,
        kelly_fraction=0.5,
        fixed_bet_fraction=0.1,
        confidence_threshold=0.5,
        odds_type='close',
        model_files=['model_0.6279_auc_diff_0.1924.json']
    )

    # Create and run evaluator
    evaluator = MMABettingEvaluator(config)
    results = evaluator.run()

    return results


if __name__ == "__main__":
    main()

"""
Unified MMA Betting Model Evaluator
Combines batch evaluation with robust betting simulation for consistent results
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
from pathlib import Path
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
    model_base_path: str = '../../../saved_models/xgboost/jan2024-dec2025/no odds 125 new/'
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
# Robust Betting Evaluator (from paste-3.txt logic)
# ============================================================================

class RobustBettingEvaluator:
    """Robust betting evaluation with consistent fight handling"""

    def __init__(self, config: BettingConfig):
        self.config = config
        self.betting_calc = BettingCalculator()

    def evaluate_bets(self, y_test: pd.Series, predictions: List[np.ndarray],
                      test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate betting performance with robust fight order handling
        This is the superior implementation from paste-3.txt
        """
        # Store fight data by unique identifier to ensure consistency
        fight_data = {}

        # Build mapping of fights to predictions
        for i in range(len(test_data)):
            row = test_data.iloc[i]
            # Use frozenset to ensure consistent fight identification regardless of fighter order
            fight_id = (row['current_fight_date'], frozenset([row['fighter_a'], row['fighter_b']]))

            # Calculate prediction
            if self.config.use_ensemble:
                y_pred_proba_avg = np.mean([pred[i] for pred in predictions], axis=0)
                models_agreeing = sum([1 for pred in predictions if
                                       (pred[i][1] > pred[i][0]) ==
                                       (y_pred_proba_avg[1] > y_pred_proba_avg[0])])
            else:
                y_pred_proba_avg = predictions[0][i]
                models_agreeing = 1

            fight_data[fight_id] = {
                'row': row,
                'prediction': y_pred_proba_avg,
                'true_outcome': y_test.iloc[i],
                'models_agreeing': models_agreeing,
                'index': i
            }

        # Initialize tracking variables
        fixed_bankroll = self.config.initial_bankroll
        kelly_bankroll = self.config.initial_bankroll
        fixed_total_volume = 0
        kelly_total_volume = 0
        fixed_correct_bets = 0
        kelly_correct_bets = 0
        fixed_total_bets = 0
        kelly_total_bets = 0
        confident_predictions = 0
        correct_confident_predictions = 0

        # Bankroll tracking
        daily_fixed_bankrolls = {}
        daily_kelly_bankrolls = {}
        daily_fixed_profits = {}
        daily_kelly_profits = {}

        available_fixed_bankroll = fixed_bankroll
        available_kelly_bankroll = kelly_bankroll

        # Sort dates for chronological processing
        all_dates = sorted(set(row['current_fight_date'] for row in
                               [data['row'] for data in fight_data.values()]))

        # Process fights chronologically
        for current_date in all_dates:
            daily_fixed_profits[current_date] = 0
            daily_kelly_profits[current_date] = 0

            # Get all fights for current date
            date_fights = [(fight_id, fight_info) for fight_id, fight_info in fight_data.items()
                           if fight_id[0] == current_date]

            # Process each fight
            for fight_id, fight_info in date_fights:
                row = fight_info['row']
                y_pred_proba_avg = fight_info['prediction']
                true_outcome = fight_info['true_outcome']
                models_agreeing = fight_info['models_agreeing']

                # Determine winners
                true_winner = row['fighter_a'] if true_outcome == 1 else row['fighter_b']

                if y_pred_proba_avg[0] > y_pred_proba_avg[1]:
                    predicted_winner = row['fighter_b']
                    winning_probability = y_pred_proba_avg[0]
                    predicted_winner_idx = 0
                else:
                    predicted_winner = row['fighter_a']
                    winning_probability = y_pred_proba_avg[1]
                    predicted_winner_idx = 1

                # Track confident predictions
                confident_predictions += 1
                if predicted_winner == true_winner:
                    correct_confident_predictions += 1

                # Place bets if confidence meets threshold
                min_models = 3 if self.config.use_ensemble else 1  # Adjusted for better consistency
                if winning_probability >= self.config.confidence_threshold and models_agreeing >= min_models:
                    # Get odds based on predicted winner
                    if predicted_winner == row['fighter_a']:
                        open_odds = row['current_fight_open_odds']
                        close_odds = row['current_fight_closing_odds']
                    else:
                        open_odds = row['current_fight_open_odds_b']
                        close_odds = row['current_fight_closing_odds_b']

                    # Determine which odds to use
                    if self.config.odds_type == 'open':
                        odds = open_odds
                    elif self.config.odds_type == 'close':
                        odds = close_odds
                    else:  # 'average'
                        odds = self.betting_calc.calculate_average_odds(open_odds, close_odds)

                    # Skip if odds outside range
                    if odds < self.config.min_odds or odds > self.config.max_underdog_odds:
                        continue

                    # Calculate stakes with available bankroll
                    fixed_available_before = available_fixed_bankroll
                    kelly_available_before = available_kelly_bankroll

                    # Fixed bet calculation
                    fixed_max_bet = fixed_available_before * self.config.max_bet_percentage
                    fixed_stake = min(
                        fixed_available_before * self.config.fixed_bet_fraction,
                        fixed_available_before,
                        fixed_max_bet
                    )

                    # Kelly bet calculation
                    b = odds / 100 if odds > 0 else 100 / abs(odds)
                    full_kelly = self.betting_calc.calculate_kelly_fraction(winning_probability, b)
                    adjusted_kelly = full_kelly * self.config.kelly_fraction
                    kelly_max_bet = kelly_available_before * self.config.max_bet_percentage
                    kelly_stake = min(
                        kelly_available_before * adjusted_kelly,
                        kelly_available_before,
                        kelly_max_bet
                    )

                    # Process fixed bet
                    if fixed_stake > 0:
                        fixed_total_bets += 1
                        available_fixed_bankroll -= fixed_stake
                        fixed_profit = self.betting_calc.calculate_profit(odds, fixed_stake)
                        fixed_total_volume += fixed_stake

                        if predicted_winner == true_winner:
                            daily_fixed_profits[current_date] += fixed_profit
                            fixed_correct_bets += 1
                        else:
                            daily_fixed_profits[current_date] -= fixed_stake

                    # Process Kelly bet
                    if kelly_stake > 0:
                        kelly_total_bets += 1
                        available_kelly_bankroll -= kelly_stake
                        kelly_profit = self.betting_calc.calculate_profit(odds, kelly_stake)
                        kelly_total_volume += kelly_stake

                        if predicted_winner == true_winner:
                            daily_kelly_profits[current_date] += kelly_profit
                            kelly_correct_bets += 1
                        else:
                            daily_kelly_profits[current_date] -= kelly_stake

            # Update bankrolls at end of day
            daily_fixed_bankrolls[current_date] = fixed_bankroll + daily_fixed_profits[current_date]
            daily_kelly_bankrolls[current_date] = kelly_bankroll + daily_kelly_profits[current_date]

            fixed_bankroll = daily_fixed_bankrolls[current_date]
            kelly_bankroll = daily_kelly_bankrolls[current_date]
            available_fixed_bankroll = fixed_bankroll
            available_kelly_bankroll = kelly_bankroll

        return {
            'fixed_final_bankroll': fixed_bankroll,
            'kelly_final_bankroll': kelly_bankroll,
            'fixed_correct_bets': fixed_correct_bets,
            'kelly_correct_bets': kelly_correct_bets,
            'fixed_total_bets': fixed_total_bets,
            'kelly_total_bets': kelly_total_bets,
            'fixed_total_volume': fixed_total_volume,
            'kelly_total_volume': kelly_total_volume,
            'confident_predictions': confident_predictions,
            'correct_confident_predictions': correct_confident_predictions,
            'daily_fixed_bankrolls': daily_fixed_bankrolls,
            'daily_kelly_bankrolls': daily_kelly_bankrolls
        }


# ============================================================================
# Main Evaluator Class (Updated with Robust Betting)
# ============================================================================

class MMABettingEvaluator:
    """Main class for evaluating MMA betting models with robust betting logic"""

    def __init__(self, config: BettingConfig = None, quiet: bool = False):
        self.config = config or BettingConfig()
        self.config.ensure_directories()
        self.console = Console()
        self.quiet = quiet
        self.encoder = None
        self.models = []
        self.robust_betting = RobustBettingEvaluator(self.config)

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
        if self.quiet:
            return

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

    def run(self) -> Dict[str, Any]:
        """Run the complete evaluation pipeline"""
        if not self.quiet:
            self.console.print("[bold cyan]Starting MMA Betting Model Evaluation[/bold cyan]")

        # Load and preprocess data
        if not self.quiet:
            self.console.print("Loading data...")
        X_val, y_val, X_test, y_test, test_data_with_display = self.load_data()

        # Load models
        if not self.quiet:
            self.console.print("Loading models...")
        self.load_models(X_val)

        # Ensure consistent feature ordering
        expected_features = self.models[0].get_booster().feature_names
        X_val = X_val.reindex(columns=expected_features)
        X_test = X_test.reindex(columns=expected_features)

        # Generate predictions
        if not self.quiet:
            self.console.print(f"Generating predictions with {self.config.calibration_type} calibration...")
        predictions = self.generate_predictions(X_val, y_val, X_test)

        # Use robust betting evaluation
        if not self.quiet:
            self.console.print("Evaluating betting strategies...")
        betting_results = self.robust_betting.evaluate_bets(y_test, predictions, test_data_with_display)

        # Calculate comprehensive metrics
        metrics = self.calculate_metrics(y_test, predictions, betting_results)

        # Display results
        self.display_results(metrics)

        if not self.quiet:
            self.console.print("[bold green]Evaluation complete![/bold green]")

        return metrics


# ============================================================================
# Batch Evaluator (Updated to use unified evaluator)
# ============================================================================

class BatchModelEvaluator:
    """Evaluate multiple models with consistent ROI calculation"""

    def __init__(self, base_config: BettingConfig = None):
        """
        Initialize batch evaluator with base configuration

        Args:
            base_config: Base configuration to use for all models
        """
        self.base_config = base_config or BettingConfig()
        self.console = Console()
        self.results = []

    def evaluate_single_model(self, model_name: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Evaluate a single model using the robust evaluator

        Args:
            model_name: Name of the model file to evaluate
            verbose: Whether to print detailed output

        Returns:
            Dictionary of evaluation metrics
        """
        # Create a config for this specific model
        config = BettingConfig(
            val_data_path=self.base_config.val_data_path,
            test_data_path=self.base_config.test_data_path,
            encoder_path=self.base_config.encoder_path,
            model_base_path=self.base_config.model_base_path,
            output_dir=self.base_config.output_dir,
            initial_bankroll=self.base_config.initial_bankroll,
            confidence_threshold=self.base_config.confidence_threshold,
            kelly_fraction=self.base_config.kelly_fraction,
            fixed_bet_fraction=self.base_config.fixed_bet_fraction,
            max_bet_percentage=self.base_config.max_bet_percentage,
            min_odds=self.base_config.min_odds,
            max_underdog_odds=self.base_config.max_underdog_odds,
            use_ensemble=False,  # Single model evaluation
            use_calibration=self.base_config.use_calibration,
            calibration_type=self.base_config.calibration_type,
            odds_type=self.base_config.odds_type,
            model_files=[model_name]  # Just this model
        )

        # Create evaluator with quiet mode
        evaluator = MMABettingEvaluator(config, quiet=True)

        try:
            metrics = evaluator.run()

            # Add model name and format for CSV
            clean_metrics = {
                'model_name': model_name,
                'confidence_threshold': config.confidence_threshold,
                'calibration_type': config.calibration_type,
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'auc': metrics.get('auc', 0) if metrics.get('auc') is not None else 0,
                'calibration_error': metrics.get('calibration_error', 0),
                'calibration_bias': metrics.get('calibration_bias', 0),
                'calibration_tendency': metrics.get('calibration_tendency', ''),
                'kelly_roi': metrics.get('kelly_roi', 0),
                'fixed_roi': metrics.get('fixed_roi', 0),
                'kelly_accuracy': metrics.get('kelly_accuracy', 0),
                'fixed_accuracy': metrics.get('fixed_accuracy', 0),
                'kelly_total_bets': metrics.get('kelly_total_bets', 0),
                'fixed_total_bets': metrics.get('fixed_total_bets', 0),
                'kelly_correct_bets': metrics.get('kelly_correct_bets', 0),
                'fixed_correct_bets': metrics.get('fixed_correct_bets', 0),
                'kelly_final_bankroll': metrics.get('kelly_final_bankroll', 0),
                'fixed_final_bankroll': metrics.get('fixed_final_bankroll', 0),
                'confident_predictions': metrics.get('confident_predictions', 0),
                'correct_confident_predictions': metrics.get('correct_confident_predictions', 0)
            }

            return clean_metrics

        except Exception as e:
            if verbose:
                self.console.print(f"[red]Error evaluating {model_name}: {str(e)}[/red]")
            return None

    def evaluate_directory(self, directory: str = None, output_filename: str = 'model_comparison.csv',
                           quiet: bool = True) -> pd.DataFrame:
        """
        Evaluate all models in a directory with consistent ROI calculation

        Args:
            directory: Directory containing model files (uses config default if None)
            output_filename: Name for the output CSV file
            quiet: If True, suppress progress output during evaluation

        Returns:
            DataFrame with comparison metrics for all models
        """
        directory = directory or self.base_config.model_base_path

        # Find all model files
        model_files = self._find_model_files(directory)

        if not model_files:
            self.console.print(f"[red]No model files found in {directory}[/red]")
            return pd.DataFrame()

        if not quiet:
            self.console.print(f"[cyan]Found {len(model_files)} models to evaluate[/cyan]")
            self.console.print(f"[cyan]Configuration: Calibration={self.base_config.calibration_type}, "
                               f"Kelly={self.base_config.kelly_fraction}, "
                               f"Threshold={self.base_config.confidence_threshold}[/cyan]\n")
        else:
            self.console.print(f"Evaluating {len(model_files)} models with robust betting logic...")

        # Clear previous results
        self.results = []

        # Evaluate each model
        for i, model_file in enumerate(model_files, 1):
            if not quiet:
                self.console.print(f"[dim]Evaluating model {i}/{len(model_files)}: {model_file}[/dim]")

            metrics = self.evaluate_single_model(model_file, verbose=False)
            if metrics:
                self.results.append(metrics)

                if not quiet:
                    # Show quick progress update
                    if metrics['kelly_roi'] > 0:
                        self.console.print(f"  ✓ Kelly ROI: {metrics['kelly_roi']:.2f}%")
                    else:
                        self.console.print(f"  ✗ Kelly ROI: {metrics['kelly_roi']:.2f}%")

        # Create comprehensive DataFrame
        if self.results:
            df = pd.DataFrame(self.results)

            # Sort by Kelly ROI (descending)
            df = df.sort_values('kelly_roi', ascending=False)

            # Add ranking column
            df.insert(0, 'rank', range(1, len(df) + 1))

            # Reorder columns for better readability
            column_order = [
                'rank', 'model_name',
                'kelly_roi', 'fixed_roi',
                'kelly_final_bankroll', 'fixed_final_bankroll',
                'kelly_accuracy', 'fixed_accuracy',
                'kelly_total_bets', 'fixed_total_bets',
                'kelly_correct_bets', 'fixed_correct_bets',
                'accuracy', 'precision', 'recall', 'f1_score', 'auc',
                'calibration_error', 'calibration_bias', 'calibration_tendency',
                'confident_predictions', 'correct_confident_predictions',
                'confidence_threshold', 'calibration_type'
            ]

            # Ensure all columns exist
            existing_columns = [col for col in column_order if col in df.columns]
            df = df[existing_columns]

            # Save full results to CSV
            output_file = os.path.join(self.base_config.output_dir, output_filename)
            df.to_csv(output_file, index=False)

            # Also save a summary CSV
            summary_columns = [
                'rank', 'model_name', 'kelly_roi', 'fixed_roi',
                'kelly_accuracy', 'accuracy', 'auc', 'calibration_error'
            ]
            summary_df = df[[col for col in summary_columns if col in df.columns]]
            summary_file = output_file.replace('.csv', '_summary.csv')
            summary_df.to_csv(summary_file, index=False)

            # Final output message
            self.console.print(f"\n[bold green]✓ Evaluation complete with robust betting logic![/bold green]")
            self.console.print(f"Results saved to: {output_file}")
            self.console.print(f"Summary saved to: {summary_file}")

            if not quiet:
                # Display summary statistics
                self._display_summary_stats(df)
                self._display_top_models(df)
            else:
                # Just show the best model when quiet
                best_model = df.iloc[0]
                self.console.print(
                    f"Best model: {best_model['model_name']} (Kelly ROI: {best_model['kelly_roi']:.2f}%)")

            return df
        else:
            self.console.print("[red]No models were successfully evaluated[/red]")
            return pd.DataFrame()

    def _find_model_files(self, directory: str) -> List[str]:
        """Find all model files in directory"""
        model_files = []

        for file in os.listdir(directory):
            if file.endswith('.json'):  # XGBoost model files
                model_files.append(file)

        return sorted(model_files)

    def _display_summary_stats(self, df: pd.DataFrame):
        """Display summary statistics"""
        self.console.print("\n[bold cyan]Summary Statistics:[/bold cyan]")
        self.console.print(f"Total models evaluated: {len(df)}")
        self.console.print(f"Models with positive Kelly ROI: {len(df[df['kelly_roi'] > 0])}")
        self.console.print(f"Average Kelly ROI: {df['kelly_roi'].mean():.2f}%")
        self.console.print(f"Best Kelly ROI: {df['kelly_roi'].max():.2f}%")
        self.console.print(f"Worst Kelly ROI: {df['kelly_roi'].min():.2f}%")

    def _display_top_models(self, df: pd.DataFrame, top_n: int = 10):
        """Display the top performing models"""
        table = Table(title=f"Top {min(top_n, len(df))} Models by Kelly ROI")

        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Model Name", style="white")
        table.add_column("Kelly ROI", justify="right", style="green")
        table.add_column("Fixed ROI", justify="right", style="yellow")
        table.add_column("Accuracy", justify="right", style="blue")
        table.add_column("AUC", justify="right", style="magenta")
        table.add_column("Cal. Error", justify="right", style="red")

        for _, row in df.head(top_n).iterrows():
            table.add_row(
                str(row['rank']),
                row['model_name'][:40],  # Truncate long names
                f"{row['kelly_roi']:.2f}%",
                f"{row['fixed_roi']:.2f}%",
                f"{row['accuracy']:.4f}",
                f"{row.get('auc', 0):.4f}" if row.get('auc') is not None else "N/A",
                f"{row['calibration_error']:.4f}"
            )

        self.console.print(table)


# ============================================================================
# Main Entry Points
# ============================================================================

def main():
    """Main entry point for single model evaluation"""
    # Create configuration
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


def batch_evaluate():
    """Entry point for batch evaluation"""
    console = Console()

    # Create base configuration
    base_config = BettingConfig(
        use_calibration=True,
        calibration_type='isotonic',
        initial_bankroll=10000,
        kelly_fraction=0.5,
        fixed_bet_fraction=0.1,
        confidence_threshold=0.5
    )

    # Create batch evaluator
    batch_evaluator = BatchModelEvaluator(base_config)

    # Evaluate all models
    console.print("[bold cyan]Evaluating all models with robust betting logic...[/bold cyan]")
    comparison_df = batch_evaluator.evaluate_directory(
        output_filename='model_comparison_robust.csv'
    )

    return comparison_df


if __name__ == "__main__":
    # Example: Run batch evaluation
    batch_evaluate()

    # Example: Run single model evaluation
    # main()

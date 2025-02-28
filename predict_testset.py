"""
UFC Fight Prediction and Betting Analysis System

This module analyzes UFC fight predictions and evaluates different betting strategies.
"""

import pandas as pd
import xgboost as xgb
import numpy as np
import os
import datetime
import sys
from io import StringIO
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from rich.table import Table
from rich.columns import Columns
from rich.text import Text
from rich.console import Group, Console
from rich.panel import Panel


class DataProcessor:
    """Handles data loading and preprocessing operations."""

    @staticmethod
    def load_data(val_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        """
        Load and prepare validation and test datasets.

        Args:
            val_path: Path to validation data CSV
            test_path: Path to test data CSV

        Returns:
            Preprocessed data for model evaluation
        """
        # Load datasets
        val_data = pd.read_csv(val_path)
        test_data = pd.read_csv(test_path)

        # Define columns to preserve for display
        display_columns = ['current_fight_date', 'fighter_a', 'fighter_b']

        # Extract target variables and features
        y_val, y_test = val_data['winner'], test_data['winner']
        X_val = DataProcessor.preprocess_features(val_data.drop(['winner'] + display_columns, axis=1))
        X_test = DataProcessor.preprocess_features(test_data.drop(['winner'] + display_columns, axis=1))

        # Combine features and display columns for test data
        test_data_with_display = pd.concat([X_test, test_data[display_columns]], axis=1)

        return X_val, X_test, y_val, y_test, test_data_with_display

    @staticmethod
    def preprocess_features(data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess feature data for model input.

        Args:
            data: Raw feature DataFrame

        Returns:
            Preprocessed feature DataFrame
        """
        # Convert categorical columns to category type
        category_columns = [col for col in data.columns if col.endswith(('fight_1', 'fight_2', 'fight_3'))]
        data = data.copy()
        data[category_columns] = data[category_columns].astype("category")
        return data


class ModelManager:
    """Handles model loading and prediction operations."""

    def __init__(self, model_dir: str, use_ensemble: bool = True):
        """
        Initialize with model directory and ensemble setting.

        Args:
            model_dir: Directory containing model files
            use_ensemble: Whether to use ensemble of models
        """
        self.model_dir = model_dir
        self.use_ensemble = use_ensemble
        self.models = []

    def load_models(self, model_files: List[str]) -> None:
        """
        Load models from files.

        Args:
            model_files: List of model filenames
        """
        if self.use_ensemble:
            for model_file in model_files:
                model_path = os.path.abspath(f'{self.model_dir}/{model_file}')
                self.models.append(self._load_model(model_path))
        else:
            # Just load the last model if not using ensemble
            model_path = os.path.abspath(f'{self.model_dir}/{model_files[-1]}')
            self.models.append(self._load_model(model_path))

    def _load_model(self, model_path: str, model_type: str = 'xgboost') -> Any:
        """
        Load a single model from file.

        Args:
            model_path: Path to model file
            model_type: Type of model

        Returns:
            Loaded model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            if model_type == 'xgboost':
                model = xgb.XGBClassifier(enable_categorical=True)
                model.load_model(model_path)
                return model
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def get_feature_names(self) -> List[str]:
        """Get feature names from the first model."""
        if not self.models:
            raise ValueError("No models loaded")
        return self.models[0].get_booster().feature_names

    def predict(self, X_val: pd.DataFrame, y_val: pd.Series, X_test: pd.DataFrame,
                use_calibration: bool = True) -> List[np.ndarray]:
        """
        Generate predictions using loaded models.

        Args:
            X_val: Validation features
            y_val: Validation targets
            X_test: Test features
            use_calibration: Whether to calibrate model probabilities

        Returns:
            List of probability predictions
        """
        # Ensure feature order consistency
        expected_features = self.get_feature_names()
        X_val = X_val.reindex(columns=expected_features)
        X_test = X_test.reindex(columns=expected_features)

        # Generate predictions
        y_pred_proba_list = []
        for model in self.models:
            if use_calibration:
                calibrated_model = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')
                calibrated_model.fit(X_val, y_val)
                y_pred_proba = calibrated_model.predict_proba(X_test)
            else:
                y_pred_proba = model.predict_proba(X_test)
            y_pred_proba_list.append(y_pred_proba)

        return y_pred_proba_list


class OddsCalculator:
    """Handles betting odds calculations and conversions."""

    @staticmethod
    def american_to_decimal(odds: float) -> float:
        """Convert American odds to decimal odds."""
        return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1

    @staticmethod
    def calculate_profit(odds: float, stake: float) -> float:
        """
        Calculate potential profit from a bet.

        Args:
            odds: American odds
            stake: Bet amount

        Returns:
            Potential profit
        """
        if odds < 0:
            return (100 / abs(odds)) * stake
        else:
            return (odds / 100) * stake

    @staticmethod
    def calculate_kelly_fraction(p: float, b: float) -> float:
        """
        Calculate Kelly criterion fraction.

        Args:
            p: Probability of winning
            b: Odds ratio (payout per unit staked)

        Returns:
            Optimal fraction to bet
        """
        q = 1 - p
        return max(0, (p - (q / b)))

    @staticmethod
    def calculate_average_odds(open_odds: float, close_odds: float) -> float:
        """
        Calculate average of open and closing odds.

        Args:
            open_odds: Opening odds
            close_odds: Closing odds

        Returns:
            Average odds in American format
        """
        # Convert to decimal for averaging
        decimal_open = OddsCalculator.american_to_decimal(open_odds)
        decimal_close = OddsCalculator.american_to_decimal(close_odds)
        avg_decimal = (decimal_open + decimal_close) / 2

        # Convert back to American odds
        if avg_decimal > 2:
            return round((avg_decimal - 1) * 100)
        else:
            return round(-100 / (avg_decimal - 1))


@dataclass
class BettingConfig:
    """Configuration parameters for betting strategies."""
    confidence_threshold: float = 0.5
    initial_bankroll: float = 10000
    kelly_fraction: float = 0.5  # Fraction of Kelly bet to use
    fixed_bet_fraction: float = 0.1  # Fraction of bankroll for fixed betting
    default_bet: float = 0.0  # Minimum bet size
    max_bet_percentage: float = 0.1  # Maximum bet as percentage of bankroll
    min_odds: int = -300  # Minimum odds to bet on
    max_underdog_odds: int = 200  # Maximum underdog odds to bet on
    use_ensemble: bool = True  # Whether to use model ensemble
    use_calibration: bool = True  # Whether to calibrate model probabilities
    odds_type: str = 'average'  # Which odds to use: 'open', 'close', or 'average'


@dataclass
class BetResult:
    """Stores the results of a single bet."""
    fight_index: int
    fighter_a: str
    fighter_b: str
    fight_date: str
    true_winner: str
    predicted_winner: str
    confidence: float
    odds: float
    models_agreeing: int

    # Fixed fraction betting results
    fixed_starting_bankroll: Optional[float] = None
    fixed_available_bankroll: Optional[float] = None
    fixed_stake: Optional[float] = None
    fixed_potential_profit: Optional[float] = None
    fixed_profit: Optional[float] = None
    fixed_bankroll_after: Optional[float] = None
    fixed_roi: Optional[float] = None

    # Kelly criterion betting results
    kelly_starting_bankroll: Optional[float] = None
    kelly_available_bankroll: Optional[float] = None
    kelly_stake: Optional[float] = None
    kelly_potential_profit: Optional[float] = None
    kelly_profit: Optional[float] = None
    kelly_bankroll_after: Optional[float] = None
    kelly_roi: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        result = {
            'Fight': self.fight_index + 1,
            'Fighter A': self.fighter_a,
            'Fighter B': self.fighter_b,
            'Date': self.fight_date,
            'True Winner': self.true_winner,
            'Predicted Winner': self.predicted_winner,
            'Confidence': f"{self.confidence:.2%}",
            'Odds': self.odds,
            'Models Agreeing': self.models_agreeing
        }

        if self.fixed_stake is not None:
            result.update({
                'Fixed Fraction Starting Bankroll': f"${self.fixed_starting_bankroll:.2f}",
                'Fixed Fraction Available Bankroll': f"${self.fixed_available_bankroll:.2f}",
                'Fixed Fraction Stake': f"${self.fixed_stake:.2f}",
                'Fixed Fraction Potential Profit': f"${self.fixed_potential_profit:.2f}",
                'Fixed Fraction Profit': self.fixed_profit,
                'Fixed Fraction Bankroll After': f"${self.fixed_bankroll_after:.2f}",
                'Fixed Fraction ROI': self.fixed_roi
            })

        if self.kelly_stake is not None:
            result.update({
                'Kelly Starting Bankroll': f"${self.kelly_starting_bankroll:.2f}",
                'Kelly Available Bankroll': f"${self.kelly_available_bankroll:.2f}",
                'Kelly Stake': f"${self.kelly_stake:.2f}",
                'Kelly Potential Profit': f"${self.kelly_potential_profit:.2f}",
                'Kelly Profit': self.kelly_profit,
                'Kelly Bankroll After': f"${self.kelly_bankroll_after:.2f}",
                'Kelly ROI': self.kelly_roi
            })

        return result


@dataclass
class BettingStats:
    """Summarizes betting performance statistics."""
    initial_bankroll: float

    # Fixed fraction stats
    fixed_final_bankroll: float
    fixed_total_volume: float
    fixed_correct_bets: int
    fixed_total_bets: int

    # Kelly criterion stats
    kelly_final_bankroll: float
    kelly_total_volume: float
    kelly_correct_bets: int
    kelly_total_bets: int

    # General prediction stats
    confident_predictions: int
    correct_confident_predictions: int

    # Tracking data
    daily_fixed_bankrolls: Dict[str, float]
    daily_kelly_bankrolls: Dict[str, float]

    # Derived stats
    @property
    def fixed_accuracy(self) -> float:
        """Return accuracy of fixed bets."""
        return self.fixed_correct_bets / self.fixed_total_bets if self.fixed_total_bets > 0 else 0

    @property
    def kelly_accuracy(self) -> float:
        """Return accuracy of Kelly bets."""
        return self.kelly_correct_bets / self.kelly_total_bets if self.kelly_total_bets > 0 else 0

    @property
    def confident_accuracy(self) -> float:
        """Return accuracy of confident predictions."""
        return self.correct_confident_predictions / self.confident_predictions if self.confident_predictions > 0 else 0

    @property
    def avg_fixed_bet_size(self) -> float:
        """Return average fixed bet size."""
        return self.fixed_total_volume / self.fixed_total_bets if self.fixed_total_bets > 0 else 0

    @property
    def avg_kelly_bet_size(self) -> float:
        """Return average Kelly bet size."""
        return self.kelly_total_volume / self.kelly_total_bets if self.kelly_total_bets > 0 else 0

    @property
    def fixed_net_profit(self) -> float:
        """Return net profit from fixed betting."""
        return self.fixed_final_bankroll - self.initial_bankroll

    @property
    def kelly_net_profit(self) -> float:
        """Return net profit from Kelly betting."""
        return self.kelly_final_bankroll - self.initial_bankroll

    @property
    def fixed_roi(self) -> float:
        """Return ROI from fixed betting."""
        return (self.fixed_net_profit / self.initial_bankroll) * 100

    @property
    def kelly_roi(self) -> float:
        """Return ROI from Kelly betting."""
        return (self.kelly_net_profit / self.initial_bankroll) * 100


class BettingEvaluator:
    """Evaluates betting strategies using model predictions."""

    def __init__(self, config: BettingConfig):
        """
        Initialize with betting configuration.

        Args:
            config: Betting configuration parameters
        """
        self.config = config
        self.odds_calculator = OddsCalculator()

    def evaluate_bets(self, y_test: pd.Series, y_pred_proba_list: List[np.ndarray],
                      test_data: pd.DataFrame) -> Tuple[BettingStats, List[BetResult]]:
        """
        Evaluate betting performance.

        Args:
            y_test: True fight outcomes
            y_pred_proba_list: Model probability predictions
            test_data: Test data with features

        Returns:
            Betting statistics and list of bet results
        """
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

        bet_results = []
        processed_fights = set()

        # Sort test data by date
        test_data = test_data.sort_values(
            by=['current_fight_date', 'fighter_a', 'fighter_b'],
            ascending=[True, True, True]
        ).reset_index(drop=True)

        # Initialize daily tracking
        daily_fixed_bankrolls = {}
        daily_kelly_bankrolls = {}
        daily_fixed_profits = {}
        daily_kelly_profits = {}

        current_date = None
        available_fixed_bankroll = fixed_bankroll
        available_kelly_bankroll = kelly_bankroll

        # Evaluate each fight
        for i in range(len(test_data)):
            row = test_data.iloc[i]
            fight_id = frozenset([row['fighter_a'], row['fighter_b']])
            fight_date = row['current_fight_date']

            # Skip duplicate fights
            if fight_id in processed_fights:
                continue
            processed_fights.add(fight_id)

            # Handle new date
            if fight_date != current_date:
                if current_date is not None:
                    # Update bankrolls at the end of each day
                    fixed_bankroll += daily_fixed_profits.get(current_date, 0)
                    kelly_bankroll += daily_kelly_profits.get(current_date, 0)
                    daily_fixed_bankrolls[current_date] = fixed_bankroll
                    daily_kelly_bankrolls[current_date] = kelly_bankroll

                current_date = fight_date
                available_fixed_bankroll = fixed_bankroll
                available_kelly_bankroll = kelly_bankroll
                daily_fixed_profits[current_date] = 0
                daily_kelly_profits[current_date] = 0

            # Determine true winner
            true_winner = row['fighter_a'] if y_test.iloc[i] == 1 else row['fighter_b']

            # Calculate prediction confidence
            if self.config.use_ensemble:
                y_pred_proba_avg = np.mean([y_pred_proba[i] for y_pred_proba in y_pred_proba_list], axis=0)
            else:
                y_pred_proba_avg = y_pred_proba_list[0][i]

            winning_probability = max(y_pred_proba_avg)
            predicted_winner = row['fighter_a'] if y_pred_proba_avg[1] > y_pred_proba_avg[0] else row['fighter_b']

            # Count models agreeing with prediction
            if self.config.use_ensemble:
                models_agreeing = sum([
                    1 for y_pred_proba in y_pred_proba_list
                    if (y_pred_proba[i][1] > y_pred_proba[i][0]) == (y_pred_proba_avg[1] > y_pred_proba_avg[0])
                ])
            else:
                models_agreeing = 1

            # Track prediction performance
            confident_predictions += 1
            if predicted_winner == true_winner:
                correct_confident_predictions += 1

            # Only consider bets that meet confidence threshold
            if winning_probability >= self.config.confidence_threshold and models_agreeing >= (5 if self.config.use_ensemble else 1):
                # Get appropriate odds based on configuration
                odds = self._get_odds(row, predicted_winner)

                # Skip if odds don't meet criteria
                if odds < self.config.min_odds or odds > self.config.max_underdog_odds:
                    continue

                # Create bet result object
                bet_result = BetResult(
                    fight_index=i,
                    fighter_a=row['fighter_a'],
                    fighter_b=row['fighter_b'],
                    fight_date=fight_date,
                    true_winner=true_winner,
                    predicted_winner=predicted_winner,
                    confidence=winning_probability,
                    odds=odds,
                    models_agreeing=models_agreeing
                )

                # Evaluate fixed fraction betting
                fixed_result = self._evaluate_fixed_bet(
                    available_fixed_bankroll, odds, winning_probability,
                    predicted_winner, true_winner, daily_fixed_profits, current_date
                )

                if fixed_result:
                    fixed_total_bets += 1
                    available_fixed_bankroll -= fixed_result['stake']
                    fixed_total_volume += fixed_result['stake']
                    bet_result.fixed_starting_bankroll = fixed_bankroll
                    bet_result.fixed_available_bankroll = fixed_result['available_bankroll']
                    bet_result.fixed_stake = fixed_result['stake']
                    bet_result.fixed_potential_profit = fixed_result['potential_profit']
                    bet_result.fixed_profit = fixed_result['profit']
                    bet_result.fixed_bankroll_after = fixed_bankroll + daily_fixed_profits[current_date]
                    bet_result.fixed_roi = (fixed_result['profit'] / fixed_result['available_bankroll']) * 100

                    if predicted_winner == true_winner:
                        fixed_correct_bets += 1

                # Evaluate Kelly criterion betting
                kelly_result = self._evaluate_kelly_bet(
                    available_kelly_bankroll, odds, winning_probability,
                    predicted_winner, true_winner, daily_kelly_profits, current_date
                )

                if kelly_result:
                    kelly_total_bets += 1
                    available_kelly_bankroll -= kelly_result['stake']
                    kelly_total_volume += kelly_result['stake']
                    bet_result.kelly_starting_bankroll = kelly_bankroll
                    bet_result.kelly_available_bankroll = kelly_result['available_bankroll']
                    bet_result.kelly_stake = kelly_result['stake']
                    bet_result.kelly_potential_profit = kelly_result['potential_profit']
                    bet_result.kelly_profit = kelly_result['profit']
                    bet_result.kelly_bankroll_after = kelly_bankroll + daily_kelly_profits[current_date]
                    bet_result.kelly_roi = (kelly_result['profit'] / kelly_result['available_bankroll']) * 100

                    if predicted_winner == true_winner:
                        kelly_correct_bets += 1

                bet_results.append(bet_result)

        # Handle the last date
        if current_date is not None:
            fixed_bankroll += daily_fixed_profits.get(current_date, 0)
            kelly_bankroll += daily_kelly_profits.get(current_date, 0)
            daily_fixed_bankrolls[current_date] = fixed_bankroll
            daily_kelly_bankrolls[current_date] = kelly_bankroll

        # Create summary statistics
        stats = BettingStats(
            initial_bankroll=self.config.initial_bankroll,
            fixed_final_bankroll=fixed_bankroll,
            fixed_total_volume=fixed_total_volume,
            fixed_correct_bets=fixed_correct_bets,
            fixed_total_bets=fixed_total_bets,
            kelly_final_bankroll=kelly_bankroll,
            kelly_total_volume=kelly_total_volume,
            kelly_correct_bets=kelly_correct_bets,
            kelly_total_bets=kelly_total_bets,
            confident_predictions=confident_predictions,
            correct_confident_predictions=correct_confident_predictions,
            daily_fixed_bankrolls=daily_fixed_bankrolls,
            daily_kelly_bankrolls=daily_kelly_bankrolls
        )

        return stats, bet_results

    def _get_odds(self, row: pd.Series, predicted_winner: str) -> float:
        """Get appropriate odds based on configuration."""
        if predicted_winner == row['fighter_a']:
            open_odds = row['current_fight_open_odds']
            close_odds = row['current_fight_closing_odds']
        else:
            open_odds = row['current_fight_open_odds_b']
            close_odds = row['current_fight_closing_odds_b']

        if self.config.odds_type == 'open':
            return open_odds
        elif self.config.odds_type == 'close':
            return close_odds
        else:  # 'average'
            return self.odds_calculator.calculate_average_odds(open_odds, close_odds)

    def _evaluate_fixed_bet(self, available_bankroll: float, odds: float,
                           winning_probability: float, predicted_winner: str,
                           true_winner: str, daily_profits: Dict[str, float],
                           current_date: str) -> Optional[Dict[str, float]]:
        """Evaluate fixed fraction betting strategy."""
        max_bet = available_bankroll * self.config.max_bet_percentage
        stake = min(
            available_bankroll * self.config.fixed_bet_fraction,
            available_bankroll,
            max_bet
        )

        if stake <= 0:
            return None

        potential_profit = self.odds_calculator.calculate_profit(odds, stake)

        if predicted_winner == true_winner:
            profit = potential_profit
            daily_profits[current_date] += profit
        else:
            profit = -stake
            daily_profits[current_date] += profit

        return {
            'available_bankroll': available_bankroll,
            'stake': stake,
            'potential_profit': potential_profit,
            'profit': profit
        }

    def _evaluate_kelly_bet(self, available_bankroll: float, odds: float,
                           winning_probability: float, predicted_winner: str,
                           true_winner: str, daily_profits: Dict[str, float],
                           current_date: str) -> Optional[Dict[str, float]]:
        """Evaluate Kelly criterion betting strategy."""
        max_bet = available_bankroll * self.config.max_bet_percentage

        # Calculate Kelly stake
        b = odds / 100 if odds > 0 else 100 / abs(odds)
        full_kelly_fraction = self.odds_calculator.calculate_kelly_fraction(winning_probability, b)
        adjusted_kelly_fraction = full_kelly_fraction * self.config.kelly_fraction

        # Calculate stake with constraints
        stake = available_bankroll * adjusted_kelly_fraction
        stake = min(stake, available_bankroll, max_bet)

        # Use default bet if Kelly bet is too small
        if stake <= available_bankroll * self.config.default_bet:
            stake = min(available_bankroll * self.config.default_bet, available_bankroll, max_bet)

        if stake <= 0:
            return None

        potential_profit = self.odds_calculator.calculate_profit(odds, stake)

        if predicted_winner == true_winner:
            profit = potential_profit
            daily_profits[current_date] += profit
        else:
            profit = -stake
            daily_profits[current_date] += profit

        return {
            'available_bankroll': available_bankroll,
            'stake': stake,
            'potential_profit': potential_profit,
            'profit': profit
        }


class ResultsAnalyzer:
    """Analyzes betting results and calculates performance metrics."""

    @staticmethod
    def calculate_daily_roi(daily_bankrolls: Dict[str, float],
                            initial_bankroll: float) -> Dict[str, float]:
        """
        Calculate daily ROI from bankroll history.

        Args:
            daily_bankrolls: Mapping from dates to bankroll values
            initial_bankroll: Starting bankroll

        Returns:
            Dictionary mapping dates to ROI percentages
        """
        daily_roi = {}
        previous_bankroll = initial_bankroll

        for date, bankroll in sorted(daily_bankrolls.items()):
            daily_profit = bankroll - previous_bankroll
            daily_roi[date] = (daily_profit / previous_bankroll) * 100
            previous_bankroll = bankroll

        return daily_roi

    @staticmethod
    def calculate_monthly_roi(daily_bankrolls: Dict[str, float],
                             initial_bankroll: float) -> Tuple[Dict[str, float], Dict[str, float], float]:
        """
        Calculate monthly ROI and profits from daily bankroll history.

        Args:
            daily_bankrolls: Mapping from dates to bankroll values
            initial_bankroll: Starting bankroll

        Returns:
            Tuple of (monthly ROI, monthly profit, total ROI)
        """
        monthly_roi = {}
        monthly_profit = {}
        current_month = None
        current_bankroll = initial_bankroll
        month_start_bankroll = initial_bankroll
        total_profit = 0

        sorted_dates = sorted(daily_bankrolls.keys())
        for date in sorted_dates:
            bankroll = daily_bankrolls[date]
            month = date[:7]  # Extract YYYY-MM

            if month != current_month:
                if current_month is not None:
                    profit = current_bankroll - month_start_bankroll
                    monthly_profit[current_month] = profit
                    total_profit += profit
                    roi = (profit / month_start_bankroll) * 100
                    monthly_roi[current_month] = roi

                current_month = month
                month_start_bankroll = current_bankroll

            current_bankroll = bankroll

        # Handle the last month
        if current_month is not None:
            profit = current_bankroll - month_start_bankroll
            monthly_profit[current_month] = profit
            total_profit += profit
            roi = (profit / month_start_bankroll) * 100
            monthly_roi[current_month] = roi

        total_roi = (total_profit / initial_bankroll) * 100

        return monthly_roi, monthly_profit, total_roi

    @staticmethod
    def get_model_metrics(y_test: pd.Series, y_pred_proba_list: List[np.ndarray]) -> Dict[str, float]:
        """
        Calculate model performance metrics.

        Args:
            y_test: True outcomes
            y_pred_proba_list: Model probability predictions

        Returns:
            Dictionary of performance metrics
        """
        # Average predictions if using ensemble
        y_pred_avg = np.mean([y_pred_proba[:, 1] for y_pred_proba in y_pred_proba_list], axis=0)
        y_pred = (y_pred_avg > 0.5).astype(int)

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, np.column_stack((1 - y_pred_avg, y_pred_avg))[:, 1])
        }


class ResultsVisualizer:
    """Visualizes betting results using rich library."""

    def __init__(self):
        """Initialize the visualizer with a rich console."""
        self.console = Console(width=160)

    def print_fight_results(self, bet_results: List[BetResult]) -> None:
        """
        Print detailed results for each fight.

        Args:
            bet_results: List of bet results
        """
        for bet in bet_results:
            bet_dict = bet.to_dict()
            fighter_a = bet_dict['Fighter A'].title()
            fighter_b = bet_dict['Fighter B'].title()
            date_obj = datetime.datetime.strptime(bet_dict['Date'], '%Y-%m-%d')
            formatted_date = date_obj.strftime('%B %d, %Y')

            # Calculate stake percentages
            fixed_avail = float(bet_dict.get('Fixed Fraction Available Bankroll', '0').replace('$', ''))
            fixed_stake = float(bet_dict.get('Fixed Fraction Stake', '0').replace('$', ''))
            fixed_stake_pct = (fixed_stake / fixed_avail) * 100 if fixed_avail > 0 else 0

            kelly_avail = float(bet_dict.get('Kelly Available Bankroll', '0').replace('$', ''))
            kelly_stake = float(bet_dict.get('Kelly Stake', '0').replace('$', ''))
            kelly_stake_pct = (kelly_stake / kelly_avail) * 100 if kelly_avail > 0 else 0

            # Create panels for each strategy
            fixed_panel = Panel(
                f"Starting Bankroll: {bet_dict.get('Fixed Fraction Starting Bankroll', 'N/A')}\n"
                f"Available Bankroll: {bet_dict.get('Fixed Fraction Available Bankroll', 'N/A')}\n"
                f"Stake: {bet_dict.get('Fixed Fraction Stake', 'N/A')}\n"
                f"Stake Percentage: {fixed_stake_pct:.2f}%\n"
                f"Potential Profit: {bet_dict.get('Fixed Fraction Potential Profit', 'N/A')}\n"
                f"Bankroll After Bet: {bet_dict.get('Fixed Fraction Bankroll After', 'N/A')}\n"
                f"Profit: ${bet_dict.get('Fixed Fraction Profit', 0):.2f}\n"
                f"ROI (of available bankroll): {bet_dict.get('Fixed Fraction ROI', 0):.2f}%",
                title="Fixed Fraction",
                expand=True,
                width=42
            )

            kelly_panel = Panel(
                f"Starting Bankroll: {bet_dict.get('Kelly Starting Bankroll', 'N/A')}\n"
                f"Available Bankroll: {bet_dict.get('Kelly Available Bankroll', 'N/A')}\n"
                f"Stake: {bet_dict.get('Kelly Stake', 'N/A')}\n"
                f"Stake Percentage: {kelly_stake_pct:.2f}%\n"
                f"Potential Profit: {bet_dict.get('Kelly Potential Profit', 'N/A')}\n"
                f"Bankroll After Bet: {bet_dict.get('Kelly Bankroll After', 'N/A')}\n"
                f"Profit: ${bet_dict.get('Kelly Profit', 0):.2f}\n"
                f"ROI (of available bankroll): {bet_dict.get('Kelly ROI', 0):.2f}%",
                title="Kelly",
                expand=True,
                width=42
            )

            # Fight information panel
            fight_info = Group(
                Text(f"True Winner: {bet_dict['True Winner'].title()}", style="green"),
                Text(f"Predicted Winner: {bet_dict['Predicted Winner'].title()}", style="blue"),
                Text(f"Confidence: {bet_dict['Confidence']}", style="yellow"),
                Text(f"Models Agreeing: {bet_dict['Models Agreeing']}/5", style="cyan")
            )

            # Main panel combining all info
            main_panel = Panel(
                Group(
                    Panel(fight_info, title="Fight Information"),
                    Columns([fixed_panel, kelly_panel], equal=False, expand=False, align="left")
                ),
                title=f"Fight {bet_dict['Fight']}: {fighter_a} vs {fighter_b} on {formatted_date}",
                subtitle=f"Odds: {bet_dict['Odds']}",
                width=89
            )

            self.console.print(main_panel, style="magenta")
            self.console.print()

    def print_daily_roi(self, daily_fixed_roi: Dict[str, float],
                        daily_kelly_roi: Dict[str, float]) -> None:
        """
        Print daily ROI table.

        Args:
            daily_fixed_roi: Fixed strategy daily ROI
            daily_kelly_roi: Kelly strategy daily ROI
        """
        self.console.print("\nDaily ROI:")
        table = Table(title="Daily Return on Investment")
        table.add_column("Date", style="cyan")
        table.add_column("Fixed Fraction ROI", justify="right", style="magenta")
        table.add_column("Kelly ROI", justify="right", style="green")

        for date in sorted(daily_fixed_roi.keys()):
            fixed_roi = f"{daily_fixed_roi[date]:.2f}%"
            kelly_roi = f"{daily_kelly_roi[date]:.2f}%"
            table.add_row(date, fixed_roi, kelly_roi)

        self.console.print(table)

    def print_monthly_roi(self, fixed_monthly_roi: Dict[str, float],
                          kelly_monthly_roi: Dict[str, float],
                          fixed_total_roi: float,
                          kelly_total_roi: float) -> None:
        """
        Print monthly ROI table.

        Args:
            fixed_monthly_roi: Fixed strategy monthly ROI
            kelly_monthly_roi: Kelly strategy monthly ROI
            fixed_total_roi: Fixed strategy total ROI
            kelly_total_roi: Kelly strategy total ROI
        """
        table = Table(title="Monthly Return on Investment")
        table.add_column("Month", style="cyan")
        table.add_column("Fixed Fraction ROI", justify="right", style="magenta")
        table.add_column("Kelly ROI", justify="right", style="green")

        for month in sorted(fixed_monthly_roi.keys()):
            fixed_monthly = f"{fixed_monthly_roi[month]:.2f}%"
            kelly_monthly = f"{kelly_monthly_roi[month]:.2f}%"
            table.add_row(month, fixed_monthly, kelly_monthly)

        table.add_row("Total", f"{fixed_total_roi:.2f}%", f"{kelly_total_roi:.2f}%")
        self.console.print(table)

    def print_overall_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Print overall model metrics.

        Args:
            metrics: Dictionary of model performance metrics
        """
        table = Table(title="Overall Model Metrics (all predictions)")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="magenta")

        table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
        table.add_row("Precision", f"{metrics['precision']:.4f}")
        table.add_row("Recall", f"{metrics['recall']:.4f}")
        table.add_row("F1 Score", f"{metrics['f1']:.4f}")
        table.add_row("AUC", f"{metrics['auc']:.4f}")

        self.console.print(table)

    def print_betting_results(self, stats: BettingStats, config: BettingConfig,
                             earliest_fight_date: str,
                             fixed_monthly_profit: Dict[str, float],
                             kelly_monthly_profit: Dict[str, float],
                             total_fights: int) -> None:
        """
        Print summary of betting results.

        Args:
            stats: Betting performance statistics
            config: Betting configuration
            earliest_fight_date: Earliest fight date in dataset
            fixed_monthly_profit: Monthly profits for fixed strategy
            kelly_monthly_profit: Monthly profits for Kelly strategy
            total_fights: Total number of fights evaluated
        """
        # Calculate derived metrics
        fixed_net_profit = stats.fixed_net_profit
        kelly_net_profit = stats.kelly_net_profit

        fixed_scale = (stats.avg_fixed_bet_size / fixed_net_profit) * 100 if fixed_net_profit != 0 else 0
        kelly_scale = (stats.avg_kelly_bet_size / kelly_net_profit) * 100 if kelly_net_profit != 0 else 0

        # Results table
        table = Table(title=f"Betting Results ({config.confidence_threshold:.0%} confidence threshold)")
        table.add_column("Metric", style="cyan")
        table.add_column("Fixed Fraction", justify="right", style="magenta")
        table.add_column("Kelly", justify="right", style="green")

        table.add_row("Total fights", str(total_fights), str(total_fights))
        table.add_row("Confident predictions", str(stats.confident_predictions), str(stats.confident_predictions))
        table.add_row("Correct predictions", str(stats.correct_confident_predictions), str(stats.correct_confident_predictions))
        table.add_row("Total bets", str(stats.fixed_total_bets), str(stats.kelly_total_bets))
        table.add_row("Correct bets", str(stats.fixed_correct_bets), str(stats.kelly_correct_bets))
        table.add_row("Betting Accuracy", f"{stats.fixed_accuracy:.2%}", f"{stats.kelly_accuracy:.2%}")
        table.add_row("Confident Prediction Accuracy", f"{stats.confident_accuracy:.2%}", f"{stats.confident_accuracy:.2%}")

        self.console.print(table)

        # Strategy panels
        fixed_panel = Panel(
            f"Initial bankroll: ${config.initial_bankroll:.2f}\n"
            f"Final bankroll: ${stats.fixed_final_bankroll:.2f}\n"
            f"Total volume: ${stats.fixed_total_volume:.2f}\n"
            f"Net profit: ${fixed_net_profit:.2f}\n"
            f"ROI: {stats.fixed_roi:.2f}%\n"
            f"Fixed bet fraction: {config.fixed_bet_fraction:.3f}\n"
            f"Average bet size: ${stats.avg_fixed_bet_size:.2f}\n"
            f"Risk: {fixed_scale:.2f}%",
            title="Fixed Fraction Betting Results"
        )

        kelly_panel = Panel(
            f"Initial bankroll: ${config.initial_bankroll:.2f}\n"
            f"Final bankroll: ${stats.kelly_final_bankroll:.2f}\n"
            f"Total volume: ${stats.kelly_total_volume:.2f}\n"
            f"Net profit: ${kelly_net_profit:.2f}\n"
            f"ROI: {stats.kelly_roi:.2f}%\n"
            f"Kelly fraction: {config.kelly_fraction:.3f}\n"
            f"Average bet size: ${stats.avg_kelly_bet_size:.2f}\n"
            f"Risk: {kelly_scale:.2f}%",
            title="Kelly Criterion Betting Results"
        )

        self.console.print(Columns([fixed_panel, kelly_panel]))


def run_betting_analysis(config: BettingConfig) -> None:
    """
    Run the complete betting analysis pipeline.

    Args:
        config: Betting configuration parameters
    """
    # Redirect stdout to capture output for final display
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    # Initialize components
    data_processor = DataProcessor()
    visualizer = ResultsVisualizer()
    analyzer = ResultsAnalyzer()

    # Load data
    X_val, X_test, y_val, y_test, test_data = data_processor.load_data(
        'data/train test data/val_data.csv',
        'data/train test data/test_data 6d.csv'
    )

    # Initialize and load models
    model_dir = 'models/xgboost/jan2024-dec2024/split 125'
    model_files = [
        'model_0.6709_auc_diff_0.0242.json',
        'model_0.6709_auc_diff_0.0239.json',
        'model_0.6709_auc_diff_0.0273.json',
        'model_0.6734_auc_diff_0.0252.json',
        'model_0.6734_auc_diff_0.0251.json'
    ]

    model_manager = ModelManager(model_dir, config.use_ensemble)
    model_manager.load_models(model_files)

    # Generate predictions
    y_pred_proba_list = model_manager.predict(X_val, y_val, X_test, use_calibration=config.use_calibration)

    # Evaluate betting strategies
    evaluator = BettingEvaluator(config)
    stats, bet_results = evaluator.evaluate_bets(y_test, y_pred_proba_list, test_data)

    # Visualize fight results
    visualizer.print_fight_results(bet_results)

    # Calculate and display ROI
    earliest_fight_date = test_data['current_fight_date'].min()
    daily_fixed_roi = analyzer.calculate_daily_roi(stats.daily_fixed_bankrolls, config.initial_bankroll)
    daily_kelly_roi = analyzer.calculate_daily_roi(stats.daily_kelly_bankrolls, config.initial_bankroll)
    visualizer.print_daily_roi(daily_fixed_roi, daily_kelly_roi)

    # Calculate monthly performance
    fixed_monthly_roi, fixed_monthly_profit, fixed_total_roi = analyzer.calculate_monthly_roi(
        stats.daily_fixed_bankrolls, config.initial_bankroll
    )
    kelly_monthly_roi, kelly_monthly_profit, kelly_total_roi = analyzer.calculate_monthly_roi(
        stats.daily_kelly_bankrolls, config.initial_bankroll
    )

    # Print monthly ROI
    visualizer.print_monthly_roi(
        fixed_monthly_roi, kelly_monthly_roi, fixed_total_roi, kelly_total_roi
    )

    # Print betting results summary
    visualizer.print_betting_results(
        stats, config, earliest_fight_date,
        fixed_monthly_profit, kelly_monthly_profit, len(test_data)
    )

    # Calculate and print model metrics
    metrics = analyzer.get_model_metrics(y_test, y_pred_proba_list)
    visualizer.print_overall_metrics(metrics)

    # Restore stdout and print final output
    sys.stdout = old_stdout
    output = mystdout.getvalue()

    # Display results in a panel
    console = Console(width=93)
    main_panel = Panel(
        output,
        title=f"Past Fight Testing (Odds Type: {config.odds_type.capitalize()})",
        border_style="bold magenta",
        expand=True,
    )
    console.print(main_panel)


if __name__ == "__main__":
    # Create configuration
    config = BettingConfig(
        confidence_threshold=0.5,
        use_calibration=True,
        initial_bankroll=10000,
        kelly_fraction=0.5,
        fixed_bet_fraction=0.1,
        max_bet_percentage=0.1,
        min_odds=-300,
        use_ensemble=True,
        odds_type='close'  # Options: 'open', 'close', 'average'
    )

    # Run analysis
    run_betting_analysis(config)
"""
UFC Fight Prediction Module

This module provides functionality to predict fight outcomes between UFC fighters
based on historical data, fighter statistics, and betting odds.
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from sklearn.calibration import CalibratedClassifierCV


class OddsCalculator:
    """Handles betting odds calculations and conversions."""

    @staticmethod
    def round_to_nearest(value: float) -> int:
        """Round to nearest integer."""
        return round(value)

    @staticmethod
    def calculate_complementary_odd(odd: float) -> float:
        """
        Calculate the complementary betting odd based on original odd.

        Args:
            odd: Original betting odd in American format

        Returns:
            Complementary betting odd
        """
        if odd > 0:
            prob = 100 / (odd + 100)
        else:
            prob = abs(odd) / (abs(odd) + 100)

        # Make probabilities sum to 104.5% (standard vig)
        complementary_prob = 1.045 - prob

        if complementary_prob >= 0.5:
            complementary_odd = -100 * complementary_prob / (1 - complementary_prob)
        else:
            complementary_odd = 100 * (1 - complementary_prob) / complementary_prob

        return OddsCalculator.round_to_nearest(complementary_odd)

    @staticmethod
    def american_to_decimal(odds: float) -> float:
        """
        Convert American odds to decimal odds.

        Args:
            odds: American format odds

        Returns:
            Decimal format odds
        """
        if odds > 0:
            return 1 + (odds / 100.0)
        else:
            return 1 + (100.0 / abs(odds))

    @staticmethod
    def calculate_kelly_fraction(probability: float, american_odd: float, fractional_kelly: float = 1.0) -> float:
        """
        Calculate the Kelly Criterion bet size.

        Args:
            probability: Estimated probability of winning
            american_odd: Odds in American format
            fractional_kelly: Fraction of full Kelly to use (0.5 = half Kelly)

        Returns:
            Optimal fraction of bankroll to bet
        """
        decimal_odds = OddsCalculator.american_to_decimal(american_odd)
        b = decimal_odds - 1  # Decimal odds to payout ratio
        p = probability
        f = (p * (b + 1) - 1) / b
        return f * fractional_kelly

    @staticmethod
    def process_odds(fight_data: Dict[str, Any]) -> Tuple[List[float], float, float, List[float], float, float]:
        """
        Process opening and closing odds for a fight.

        Args:
            fight_data: Dictionary containing fight odds data

        Returns:
            Tuple containing processed odds and derived metrics
        """
        # Process opening odds
        odds_a = fight_data.get('open_odds', np.nan)
        odds_b = fight_data.get('open_odds_b', np.nan)

        # Handle opening odds
        if pd.notna(odds_a) and pd.notna(odds_b):
            odds_a = OddsCalculator.round_to_nearest(odds_a)
            odds_b = OddsCalculator.round_to_nearest(odds_b)
        elif pd.notna(odds_a):
            odds_a = OddsCalculator.round_to_nearest(odds_a)
            odds_b = OddsCalculator.calculate_complementary_odd(odds_a)
        elif pd.notna(odds_b):
            odds_b = OddsCalculator.round_to_nearest(odds_b)
            odds_a = OddsCalculator.calculate_complementary_odd(odds_b)
        else:
            # Default to -110/-110 if no odds provided
            odds_a, odds_b = -110, -110

        opening_odds = [odds_a, odds_b]
        opening_odds_diff = odds_a - odds_b
        opening_odds_ratio = odds_a / odds_b if odds_b != 0 else 0

        # Process closing odds
        close_a = fight_data.get('closing_range_end', np.nan)
        close_b = fight_data.get('closing_range_end_b', np.nan)

        # Handle closing odds
        if pd.notna(close_a) and pd.notna(close_b):
            close_a = OddsCalculator.round_to_nearest(close_a)
            close_b = OddsCalculator.round_to_nearest(close_b)
        elif pd.notna(close_a):
            close_a = OddsCalculator.round_to_nearest(close_a)
            close_b = OddsCalculator.calculate_complementary_odd(close_a)
        elif pd.notna(close_b):
            close_b = OddsCalculator.round_to_nearest(close_b)
            close_a = OddsCalculator.calculate_complementary_odd(close_b)
        else:
            close_a, close_b = -110, -110

        closing_odds = [close_a, close_b]
        closing_odds_diff = close_a - close_b
        closing_odds_ratio = close_a / close_b if close_b != 0 else 0

        return (
            opening_odds, opening_odds_diff, opening_odds_ratio,
            closing_odds, closing_odds_diff, closing_odds_ratio
        )


class FighterStats:
    """Calculates and processes fighter statistics."""

    @staticmethod
    def calculate_stats(
            fighter_name: str,
            df: pd.DataFrame,
            current_fight_date: pd.Timestamp,
            n_past_fights: int
    ) -> Tuple[float, float, float, float, float]:
        """
        Calculate various statistics for a fighter based on their past fights.

        Args:
            fighter_name: Fighter's name
            df: DataFrame with historical fight data
            current_fight_date: Date of the upcoming fight
            n_past_fights: Number of past fights to consider

        Returns:
            Tuple of (age, experience, days_since_last_fight, win_streak, loss_streak)
        """
        # Get all fighter's past fights and most recent ones
        fighter_all_fights = df[(df['fighter'] == fighter_name) & (df['fight_date'] < current_fight_date)]
        fighter_all_fights = fighter_all_fights.sort_values(by='fight_date', ascending=False)

        fighter_recent_fights = fighter_all_fights.head(n_past_fights)

        if fighter_recent_fights.empty:
            return np.nan, np.nan, np.nan, np.nan, np.nan

        # Extract key dates and stats
        last_fight_date = fighter_recent_fights['fight_date'].iloc[0]
        last_known_age = fighter_recent_fights['age'].iloc[0]
        first_fight_date = fighter_all_fights['fight_date'].iloc[-1]

        # Calculate time-based stats
        days_since_last_fight = (current_fight_date - last_fight_date).days
        current_age = last_known_age + days_since_last_fight / 365.25
        years_of_experience = (current_fight_date - first_fight_date).days / 365.25

        # Get current streaks
        win_streak = fighter_recent_fights['win_streak'].iloc[0]
        loss_streak = fighter_recent_fights['loss_streak'].iloc[0]

        # Adjust streaks based on most recent result
        most_recent_result = fighter_recent_fights['winner'].iloc[0]
        if most_recent_result == 1:  # Won last fight
            win_streak += 1
            loss_streak = 0
        elif most_recent_result == 0:  # Lost last fight
            loss_streak += 1
            win_streak = 0

        return current_age, years_of_experience, days_since_last_fight, win_streak, loss_streak


class DataProcessor:
    """Handles data loading and preprocessing."""

    @staticmethod
    def load_and_preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess fight data for model input.

        Args:
            data: Raw data DataFrame

        Returns:
            Preprocessed DataFrame
        """
        # Convert fight history columns to categorical
        category_columns = [col for col in data.columns if col.endswith(('fight_1', 'fight_2', 'fight_3'))]
        data = data.copy()
        data[category_columns] = data[category_columns].astype("category")
        return data

    @staticmethod
    def rename_fighter_columns(col: str) -> str:
        """
        Standardize column names related to fighters.

        Args:
            col: Original column name

        Returns:
            Standardized column name
        """
        if col == 'fighter':
            return 'fighter_a'

        if 'fighter' in col and not col.startswith('fighter'):
            if 'b_fighter_b' in col:
                return col.replace('b_fighter_b', 'fighter_b_opponent')
            elif 'b_fighter' in col:
                return col.replace('b_fighter', 'fighter_a_opponent')
            elif 'fighter' in col and 'fighter_b' not in col:
                return col.replace('fighter', 'fighter_a')

        return col

    @staticmethod
    def safe_divide(a: float, b: float, default: float = 0) -> float:
        """Safely divide two numbers, handling division by zero."""
        return a / b if b != 0 else default


class MatchupCreator:
    """Creates matchup data for fight prediction."""

    def __init__(self, historical_data_path: str, test_data_path: str = 'data/train test data/test_data.csv'):
        """
        Initialize with paths to required data files.

        Args:
            historical_data_path: Path to historical fight data
            test_data_path: Path to test data for column alignment
        """
        self.historical_data_path = historical_data_path
        self.test_data_path = test_data_path
        self.odds_calculator = OddsCalculator()

    def create_matchup(
            self,
            fighter_a: str,
            fighter_b: str,
            current_fight_data: Dict[str, Any],
            n_past_fights: int = 3,
            output_dir: str = ''
    ) -> Optional[pd.DataFrame]:
        """
        Create a matchup DataFrame for prediction.

        Args:
            fighter_a: Name of first fighter
            fighter_b: Name of second fighter
            current_fight_data: Dict with fight details including odds and date
            n_past_fights: Number of past fights to consider
            output_dir: Directory to save output CSV

        Returns:
            DataFrame with matchup features or None if insufficient data
        """
        # Load historical fight data
        df = pd.read_csv(self.historical_data_path)
        df['fight_date'] = pd.to_datetime(df['fight_date'])

        # Define columns to exclude from features
        columns_to_exclude = [
            'fighter', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
            'result', 'winner', 'weight_class', 'scheduled_rounds',
            'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b'
        ]

        # Define features to include
        features_to_include = [
            col for col in df.columns
            if col not in columns_to_exclude and col != 'age' and not col.endswith('_age')
        ]

        current_fight_date = pd.to_datetime(current_fight_data['current_fight_date'])

        # Get fighters' recent fights
        fighter_df = self._get_fighter_recent_fights(df, fighter_a, current_fight_date, n_past_fights)
        opponent_df = self._get_fighter_recent_fights(df, fighter_b, current_fight_date, n_past_fights)

        # Check if we have enough data
        if len(fighter_df) < n_past_fights or len(opponent_df) < n_past_fights:
            print(f"Not enough past fight data for {fighter_a} or {fighter_b}")
            return None

        # Extract feature values
        matchup_df = self._build_matchup_dataframe(
            df, fighter_a, fighter_b, fighter_df, opponent_df,
            features_to_include, current_fight_date, current_fight_data, n_past_fights
        )

        # Ensure correct column order by matching test data
        matchup_df = self._align_with_test_data(matchup_df)

        # Save if output directory is provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, 'specific_matchup_data.csv')
            matchup_df.to_csv(output_file, index=False)
            print(f"Matchup data saved to {output_file}")

        return matchup_df

    def _get_fighter_recent_fights(
            self,
            df: pd.DataFrame,
            fighter_name: str,
            current_date: pd.Timestamp,
            n_fights: int
    ) -> pd.DataFrame:
        """Get a fighter's most recent fights before the specified date."""
        return df[(df['fighter'].str.lower() == fighter_name.lower()) &
                  (df['fight_date'] < current_date)] \
            .sort_values(by='fight_date', ascending=False) \
            .head(n_fights)

    def _build_matchup_dataframe(
            self,
            df: pd.DataFrame,
            fighter_a: str,
            fighter_b: str,
            fighter_df: pd.DataFrame,
            opponent_df: pd.DataFrame,
            features_to_include: List[str],
            current_fight_date: pd.Timestamp,
            current_fight_data: Dict[str, Any],
            n_past_fights: int
    ) -> pd.DataFrame:
        """Build a DataFrame with all matchup features."""
        # Calculate mean feature values from past fights
        fighter_features = fighter_df[features_to_include].mean().values
        opponent_features = opponent_df[features_to_include].mean().values

        # Get past fight results
        results_fighter = fighter_df[['result', 'winner', 'weight_class', 'scheduled_rounds']] \
            .head(n_past_fights).values.flatten()
        results_opponent = opponent_df[['result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']] \
            .head(n_past_fights).values.flatten()

        # Pad results arrays to consistent length
        results_fighter = np.pad(
            results_fighter,
            (0, n_past_fights * 4 - len(results_fighter)),
            'constant',
            constant_values=np.nan
        )
        results_opponent = np.pad(
            results_opponent,
            (0, n_past_fights * 4 - len(results_opponent)),
            'constant',
            constant_values=np.nan
        )

        # Process odds data
        odds_data = self.odds_calculator.process_odds(current_fight_data)
        (current_fight_odds, current_fight_odds_diff, current_fight_odds_ratio,
         current_fight_closing_odds, current_fight_closing_odds_diff,
         current_fight_closing_odds_ratio) = odds_data

        # Calculate fighter stats
        age_a, exp_a, days_a, win_streak_a, loss_streak_a = FighterStats.calculate_stats(
            fighter_a, df, current_fight_date, n_past_fights
        )
        age_b, exp_b, days_b, win_streak_b, loss_streak_b = FighterStats.calculate_stats(
            fighter_b, df, current_fight_date, n_past_fights
        )

        # Calculate derived stats with safe division
        safe_div = DataProcessor.safe_divide
        age_diff = age_a - age_b
        age_ratio = safe_div(age_a, age_b)
        exp_diff = exp_a - exp_b
        exp_ratio = safe_div(exp_a, exp_b, 1)
        days_diff = days_a - days_b
        days_ratio = safe_div(days_a, days_b, 1)
        win_streak_diff = win_streak_a - win_streak_b
        win_streak_ratio = safe_div(win_streak_a, win_streak_b, 1)
        loss_streak_diff = loss_streak_a - loss_streak_b
        loss_streak_ratio = safe_div(loss_streak_a, loss_streak_b, 1)

        # Calculate ELO stats
        elo_a = fighter_df['fight_outcome_elo'].iloc[0]
        elo_b = opponent_df['fight_outcome_elo'].iloc[0]
        elo_diff = elo_a - elo_b
        elo_a_win_chance = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        elo_b_win_chance = 1 - elo_a_win_chance
        elo_ratio = safe_div(elo_a, elo_b)
        elo_stats = [elo_a, elo_b, elo_diff, elo_a_win_chance, elo_b_win_chance, elo_ratio]

        # Build column names for different feature groups
        feature_columns = self._build_feature_column_names(features_to_include, n_past_fights)
        results_columns = self._build_results_column_names(n_past_fights)
        odds_age_columns = self._build_odds_age_column_names()
        new_columns = self._build_new_column_names()

        # Combine all column names
        column_names = ['fighter', 'fighter_b'] + feature_columns + results_columns + \
                       odds_age_columns + new_columns + ['current_fight_date']

        # Combine all feature values
        combined_features = np.concatenate([
            fighter_features, opponent_features, results_fighter, results_opponent,
            current_fight_odds, [current_fight_odds_diff, current_fight_odds_ratio],
            current_fight_closing_odds, [current_fight_closing_odds_diff, current_fight_closing_odds_ratio],
            [age_a, age_b, age_diff, age_ratio],
            elo_stats,
            [win_streak_a, win_streak_b, win_streak_diff, win_streak_ratio],
            [loss_streak_a, loss_streak_b, loss_streak_diff, loss_streak_ratio],
            [exp_a, exp_b, exp_diff, exp_ratio],
            [days_a, days_b, days_diff, days_ratio]
        ])

        # Create DataFrame and add additional columns
        matchup_df = pd.DataFrame([combined_features], columns=column_names[2:-1])
        matchup_df.insert(0, 'fighter', fighter_a)
        matchup_df.insert(1, 'fighter_b', fighter_b)
        matchup_df['current_fight_date'] = current_fight_data['current_fight_date']

        # Standardize column names
        matchup_df.columns = [DataProcessor.rename_fighter_columns(col) for col in matchup_df.columns]

        # Add differential and ratio columns for features
        matchup_df = self._add_diff_ratio_columns(matchup_df, features_to_include, n_past_fights)

        return matchup_df

    def _build_feature_column_names(self, features: List[str], n_past_fights: int) -> List[str]:
        """Build column names for feature columns."""
        return [f"{feature}_fighter_avg_last_{n_past_fights}" for feature in features] + \
            [f"{feature}_fighter_b_avg_last_{n_past_fights}" for feature in features]

    def _build_results_column_names(self, n_past_fights: int) -> List[str]:
        """Build column names for fight results."""
        results_columns = []
        for i in range(1, n_past_fights + 1):
            results_columns += [
                f"result_fight_{i}", f"winner_fight_{i}",
                f"weight_class_fight_{i}", f"scheduled_rounds_fight_{i}",
                f"result_b_fight_{i}", f"winner_b_fight_{i}",
                f"weight_class_b_fight_{i}", f"scheduled_rounds_b_fight_{i}"
            ]
        return results_columns

    def _build_odds_age_column_names(self) -> List[str]:
        """Build column names for odds and age features."""
        return [
            'current_fight_open_odds', 'current_fight_open_odds_b',
            'current_fight_open_odds_diff', 'current_fight_open_odds_ratio',
            'current_fight_closing_odds', 'current_fight_closing_odds_b',
            'current_fight_closing_odds_diff', 'current_fight_closing_odds_ratio',
            'current_fight_age', 'current_fight_age_b',
            'current_fight_age_diff', 'current_fight_age_ratio'
        ]

    def _build_new_column_names(self) -> List[str]:
        """Build column names for derived statistics."""
        return [
            'current_fight_pre_fight_elo_a', 'current_fight_pre_fight_elo_b',
            'current_fight_pre_fight_elo_diff',
            'current_fight_pre_fight_elo_a_win_chance', 'current_fight_pre_fight_elo_b_win_chance',
            'current_fight_pre_fight_elo_ratio',
            'current_fight_win_streak_a', 'current_fight_win_streak_b',
            'current_fight_win_streak_diff', 'current_fight_win_streak_ratio',
            'current_fight_loss_streak_a', 'current_fight_loss_streak_b',
            'current_fight_loss_streak_diff', 'current_fight_loss_streak_ratio',
            'current_fight_years_experience_a', 'current_fight_years_experience_b',
            'current_fight_years_experience_diff', 'current_fight_years_experience_ratio',
            'current_fight_days_since_last_a', 'current_fight_days_since_last_b',
            'current_fight_days_since_last_diff', 'current_fight_days_since_last_ratio'
        ]

    def _add_diff_ratio_columns(
            self,
            df: pd.DataFrame,
            features: List[str],
            n_past_fights: int
    ) -> pd.DataFrame:
        """Add difference and ratio columns for features."""
        diff_columns = {}
        ratio_columns = {}

        for feature in features:
            col_a = f"{feature}_fighter_a_avg_last_{n_past_fights}"
            col_b = f"{feature}_fighter_b_avg_last_{n_past_fights}"

            if col_a in df.columns and col_b in df.columns:
                diff_columns[f"matchup_{feature}_diff_avg_last_{n_past_fights}"] = df[col_a] - df[col_b]

                # Safely handle division by zero
                denominator = df[col_b].replace(0, 1)
                ratio_columns[f"matchup_{feature}_ratio_avg_last_{n_past_fights}"] = df[col_a] / denominator

        return pd.concat([df, pd.DataFrame(diff_columns), pd.DataFrame(ratio_columns)], axis=1)

    def _align_with_test_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align DataFrame columns with test data schema."""
        # Read test data schema (just the headers)
        test_df = pd.read_csv(self.test_data_path, nrows=0)
        correct_columns = test_df.columns.tolist()

        # Add missing columns as NaN
        for col in correct_columns:
            if col not in df.columns:
                df[col] = np.nan

        # Reorder columns to match test data
        return df[correct_columns]


class ModelManager:
    """Handles loading and prediction with ML models."""

    @staticmethod
    def load_model(model_path: str, model_type: str = 'xgboost') -> Any:
        """
        Load a trained model from file.

        Args:
            model_path: Path to the model file
            model_type: Type of model ('xgboost', etc.)

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

    @staticmethod
    def ensemble_prediction(
            matchup_df: pd.DataFrame,
            model_dir: str,
            val_data_path: str,
            use_calibration: bool = True
    ) -> Tuple[str, float]:
        """
        Generate predictions using an ensemble of models.

        Args:
            matchup_df: DataFrame with matchup features
            model_dir: Directory containing model files
            val_data_path: Path to validation data for calibration
            use_calibration: Whether to calibrate probabilities

        Returns:
            Tuple of (predicted_winner, winning_probability)
        """
        # List of model files to use
        model_files = [
            'model_0.6709_auc_diff_0.0242.json',
            'model_0.6709_auc_diff_0.0239.json',
            'model_0.6709_auc_diff_0.0273.json',
            'model_0.6734_auc_diff_0.0252.json',
            'model_0.6734_auc_diff_0.0251.json'
        ]

        # Load all models
        models = []
        for model_file in model_files:
            model_path = os.path.join(model_dir, model_file)
            model = ModelManager.load_model(model_path, 'xgboost')
            models.append(model)

        # Ensure features match model expectations
        expected_features = models[0].get_booster().feature_names
        X = matchup_df.reindex(columns=expected_features)
        X = DataProcessor.load_and_preprocess_data(X)

        # Load validation data for calibration if needed
        if use_calibration:
            val_data = pd.read_csv(val_data_path)
            X_val = val_data.drop(['winner'], axis=1).reindex(columns=expected_features)
            X_val = DataProcessor.load_and_preprocess_data(X_val)
            y_val = val_data['winner']

        # Get predictions from all models
        y_pred_proba_list = []
        for model in models:
            if use_calibration:
                calibrated_model = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')
                calibrated_model.fit(X_val, y_val)
                y_pred_proba = calibrated_model.predict_proba(X)
            else:
                y_pred_proba = model.predict_proba(X)
            y_pred_proba_list.append(y_pred_proba)

        # Average predictions across models
        y_pred_proba_avg = np.mean(y_pred_proba_list, axis=0)
        fighter_a_probability = y_pred_proba_avg[0][1]
        fighter_b_probability = 1 - fighter_a_probability

        # Determine predicted winner
        if fighter_a_probability > fighter_b_probability:
            predicted_winner = matchup_df['fighter_a'].iloc[0]
            winning_probability = fighter_a_probability
        else:
            predicted_winner = matchup_df['fighter_b'].iloc[0]
            winning_probability = fighter_b_probability

        return predicted_winner, winning_probability


@dataclass
class FightPrediction:
    """Stores fight prediction results."""
    fighter_a: str
    fighter_b: str
    predicted_winner: str
    winning_probability: float
    kelly_fraction: float
    fight_date: str
    odds_used: int


class UFCPredictor:
    """Main class for UFC fight prediction."""

    def __init__(
            self,
            historical_data_path: str = "data/combined_sorted_fighter_stats.csv",
            model_dir: str = "models/xgboost/jan2024-july2024/split 125/",
            val_data_path: str = "data/train test data/val_data.csv",
            output_dir: str = "data/matchup data"
    ):
        """
        Initialize UFC predictor with paths to required files.

        Args:
            historical_data_path: Path to historical UFC data
            model_dir: Directory containing model files
            val_data_path: Path to validation data
            output_dir: Directory to save output files
        """
        self.historical_data_path = historical_data_path
        self.model_dir = model_dir
        self.val_data_path = val_data_path
        self.output_dir = output_dir
        self.matchup_creator = MatchupCreator(historical_data_path)
        self.odds_calculator = OddsCalculator()

    def predict_fight(
            self,
            fighter_a: str,
            fighter_b: str,
            fight_date: str,
            fighter_a_odds: Dict[str, float],
            fighter_b_odds: Dict[str, float],
            fractional_kelly: float = 0.5,
            use_calibration: bool = True,
            n_past_fights: int = 3
    ) -> Optional[FightPrediction]:
        """
        Predict the outcome of a UFC fight.

        Args:
            fighter_a: Name of first fighter
            fighter_b: Name of second fighter
            fight_date: Date of the fight (YYYY-MM-DD)
            fighter_a_odds: Dict with open and closing odds for fighter_a
            fighter_b_odds: Dict with open and closing odds for fighter_b
            fractional_kelly: Fraction of Kelly criterion to use
            use_calibration: Whether to calibrate model probabilities
            n_past_fights: Number of past fights to consider

        Returns:
            FightPrediction object with prediction results or None if prediction fails
        """
        # Prepare fight data
        current_fight_data = {
            'open_odds': fighter_a_odds.get('open'),
            'open_odds_b': fighter_b_odds.get('open'),
            'closing_range_end': fighter_a_odds.get('closing'),
            'closing_range_end_b': fighter_b_odds.get('closing'),
            'current_fight_date': fight_date
        }

        # Create matchup data
        matchup_df = self.matchup_creator.create_matchup(
            fighter_a, fighter_b, current_fight_data,
            n_past_fights=n_past_fights, output_dir=self.output_dir
        )

        if matchup_df is None:
            return None

        # Get prediction
        predicted_winner, winning_probability = ModelManager.ensemble_prediction(
            matchup_df, self.model_dir, self.val_data_path, use_calibration
        )

        # Calculate Kelly criterion bet size
        if predicted_winner.lower() == fighter_a.lower():
            odds_used = current_fight_data['closing_range_end']
        else:
            odds_used = current_fight_data['closing_range_end_b']

        kelly_fraction = self.odds_calculator.calculate_kelly_fraction(
            winning_probability, odds_used, fractional_kelly
        )

        # Return prediction results
        return FightPrediction(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            predicted_winner=predicted_winner,
            winning_probability=winning_probability,
            kelly_fraction=kelly_fraction,
            fight_date=fight_date,
            odds_used=odds_used
        )


def main():
    """Run example prediction."""
    # Fight details
    fighter_a = "william gomis"
    fighter_b = "Hyder Amil"

    # Odds data
    fighter_a_odds = {'open': -186, 'closing': -220}
    fighter_b_odds = {'open': 150, 'closing': 185}

    # Setup predictor
    predictor = UFCPredictor(
        historical_data_path="data/combined_sorted_fighter_stats.csv",
        model_dir="models/xgboost/jan2024-dec2024/split 125/",
        val_data_path="data/train test data/val_data.csv",
        output_dir="data/matchup data"
    )

    # Get prediction
    prediction = predictor.predict_fight(
        fighter_a=fighter_a,
        fighter_b=fighter_b,
        fight_date="2025-03-01",
        fighter_a_odds=fighter_a_odds,
        fighter_b_odds=fighter_b_odds,
        fractional_kelly=0.5,
        use_calibration=True
    )

    # Print results
    if prediction:
        print(f"Predicted winner: {prediction.predicted_winner}")
        print(f"Winning probability: {prediction.winning_probability:.2%}")

        if prediction.kelly_fraction > 0:
            print(f"Fractional Kelly bet fraction: {prediction.kelly_fraction:.2%} of bankroll")
        else:
            print("Kelly fraction is negative or zero, suggesting not to bet.")
    else:
        print("Could not generate matchup data.")


if __name__ == "__main__":
    main()

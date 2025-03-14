"""
UFC Fight Predictor

This script combines FighterMatchupPredictor and XGBoost model ensemble to predict fight outcomes.
"""

import pandas as pd
import xgboost as xgb
import numpy as np
import os
from datetime import datetime
from typing import List, Dict, Optional, Union

from sklearn.calibration import CalibratedClassifierCV

# Import from the original module
from data_cleaner import FightDataProcessor, DataUtils, OddsUtils


class FighterMatchupPredictor:
    """Creates matchup data for a specific pair of fighters."""

    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the matchup predictor with data directory.

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        self.fight_processor = FightDataProcessor(data_dir)
        self.utils = DataUtils()
        self.odds_utils = OddsUtils()

    def create_fighter_matchup(
        self,
        fighter_a: str,
        fighter_b: str,
        closing_odds_a: float,
        closing_odds_b: float,
        fight_date: Optional[Union[str, datetime]] = None,
        n_past_fights: int = 3
    ) -> pd.DataFrame:
        """
        Create a specific matchup between two fighters.

        Args:
            fighter_a: Name of first fighter
            fighter_b: Name of second fighter
            closing_odds_a: Closing odds for fighter A
            closing_odds_b: Closing odds for fighter B
            fight_date: Date of the fight in YYYY-MM-DD format or datetime object
            n_past_fights: Number of past fights to consider for statistics (default: 3)

        Returns:
            DataFrame with the matchup data formatted for prediction
        """
        print(f"Creating matchup: {fighter_a} vs {fighter_b}")
        print(f"Odds: {fighter_a} ({closing_odds_a}) vs {fighter_b} ({closing_odds_b})")

        # Process the fight date
        if fight_date is None:
            current_date = datetime.now()
            print(f"Using current date: {current_date.strftime('%Y-%m-%d')}")
        elif isinstance(fight_date, str):
            try:
                current_date = datetime.strptime(fight_date, '%Y-%m-%d')
                print(f"Using specified fight date: {fight_date}")
            except ValueError:
                print(f"Invalid date format. Using current date instead.")
                current_date = datetime.now()
        else:
            current_date = fight_date
            print(f"Using specified fight date: {current_date.strftime('%Y-%m-%d')}")

        # Load combined fighter stats data
        fighter_stats_file = os.path.join(self.data_dir, "combined_sorted_fighter_stats.csv")
        df = self.fight_processor._load_csv(fighter_stats_file)

        # Convert fighter names to lowercase for case-insensitive comparison
        df['fighter_lower'] = df['fighter'].str.lower()
        fighter_a_lower = fighter_a.lower()
        fighter_b_lower = fighter_b.lower()

        # Check if fighters exist in the dataset (case-insensitive)
        if fighter_a_lower not in df['fighter_lower'].values:
            raise ValueError(f"Fighter '{fighter_a}' not found in the dataset")
        if fighter_b_lower not in df['fighter_lower'].values:
            raise ValueError(f"Fighter '{fighter_b}' not found in the dataset")

        # Get the most recent fight data for each fighter (case-insensitive)
        df_fighter_a = df[(df['fighter_lower'] == fighter_a_lower)].sort_values(by='fight_date', ascending=False)
        df_fighter_b = df[(df['fighter_lower'] == fighter_b_lower)].sort_values(by='fight_date', ascending=False)

        if len(df_fighter_a) == 0 or len(df_fighter_b) == 0:
            raise ValueError("Not enough fight data available for one or both fighters")

        # Define columns to exclude from features
        columns_to_exclude = [
            'fighter', 'fighter_lower', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
            'result', 'winner', 'weight_class', 'scheduled_rounds',
            'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b'
        ]

        # Define features to include
        features_to_include = [
            col for col in df.columns if col not in columns_to_exclude and
                                         col != 'age' and not col.endswith('_age')
        ]

        # Create the matchup data
        matchup_data = self._process_matchup(
            df, df_fighter_a, df_fighter_b,
            fighter_a, fighter_b,
            closing_odds_a, closing_odds_b,
            n_past_fights, features_to_include,
            current_date
        )

        # Generate column names
        column_names = self._generate_column_names(
            features_to_include, ['winner'], n_past_fights, True
        )

        # Create the matchup DataFrame
        matchup_df = pd.DataFrame([matchup_data], columns=column_names)

        # Standardize column names
        matchup_df.columns = [self.utils.rename_columns_general(col) for col in matchup_df.columns]

        # Calculate additional differential and ratio columns
        matchup_df = self._calculate_matchup_features(matchup_df, features_to_include, n_past_fights)

        # Read the list of columns to remove from removed_features.txt
        removed_features_file = os.path.join(self.data_dir, "train test data", "removed_features.txt")
        removed_features = []
        if os.path.exists(removed_features_file):
            with open(removed_features_file, 'r') as f:
                removed_features = [line.strip() for line in f if line.strip()]

        # Drop fight_date column
        if 'fight_date' in matchup_df.columns:
            matchup_df = matchup_df.drop(columns=['fight_date'])

        # Drop columns listed in removed_features.txt
        columns_to_drop = [col for col in removed_features if col in matchup_df.columns]
        if columns_to_drop:
            matchup_df = matchup_df.drop(columns=columns_to_drop)

        # Load test dataset to get its columns
        test_data_file = os.path.join(self.data_dir, "train test data", "test_data.csv")
        if os.path.exists(test_data_file):
            test_df = self.fight_processor._load_csv(test_data_file)
            test_columns = test_df.columns.tolist()

            # Keep only columns that are in the test dataset
            columns_to_keep = [col for col in matchup_df.columns if col in test_columns]

            # Always keep fighter names if they exist
            if 'fighter_a' in matchup_df.columns and 'fighter_a' not in columns_to_keep:
                columns_to_keep.insert(0, 'fighter_a')
            if 'fighter_b' in matchup_df.columns and 'fighter_b' not in columns_to_keep:
                columns_to_keep.insert(1, 'fighter_b')

            matchup_df = matchup_df[columns_to_keep]
            print(f"Keeping only the {len(columns_to_keep)} features present in test_data.csv")
        else:
            print(f"Warning: test_data.csv not found at {test_data_file}")

        # Save output
        output_filename = os.path.join(self.data_dir, f'matchup data/specific_matchup_data.csv')
        self.fight_processor._save_csv(matchup_df, output_filename)

        print(f"Created matchup prediction data for {fighter_a} vs {fighter_b}")
        print(f"Output saved to: {output_filename}")

        return matchup_df

    def _process_matchup(
        self,
        df: pd.DataFrame,
        df_fighter_a: pd.DataFrame,
        df_fighter_b: pd.DataFrame,
        fighter_a: str,
        fighter_b: str,
        closing_odds_a: float,
        closing_odds_b: float,
        n_past_fights: int,
        features_to_include: List[str],
        current_date: Optional[datetime] = None
    ) -> List:
        """Process specific matchup to create feature vector."""
        # Extract features from available past fights
        fighter_a_features = df_fighter_a[features_to_include].head(n_past_fights).mean().values
        fighter_b_features = df_fighter_b[features_to_include].head(n_past_fights).mean().values

        # Extract recent fight results
        # Only extract the available fight results, up to tester number
        tester = 6 - n_past_fights
        num_a_results = min(len(df_fighter_a), tester)
        num_b_results = min(len(df_fighter_b), tester)

        results_fighter_a = df_fighter_a[['result', 'winner', 'weight_class', 'scheduled_rounds']].head(
            num_a_results).values.flatten() if num_a_results > 0 else np.array([])

        results_fighter_b = df_fighter_b[['result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']].head(
            num_b_results).values.flatten() if num_b_results > 0 else np.array([])

        # Pad results with None values to ensure consistent length
        results_fighter_a = np.pad(
            results_fighter_a,
            (0, tester * 4 - len(results_fighter_a)),
            'constant',
            constant_values=np.nan
        )
        results_fighter_b = np.pad(
            results_fighter_b,
            (0, tester * 4 - len(results_fighter_b)),
            'constant',
            constant_values=np.nan
        )

        # Process odds data
        current_fight_odds, current_fight_odds_diff, current_fight_odds_ratio = self.odds_utils.process_odds_pair(
            closing_odds_a, closing_odds_b
        )

        current_fight_closing_odds, current_fight_closing_odds_diff, current_fight_closing_odds_ratio = self.odds_utils.process_odds_pair(
            closing_odds_a, closing_odds_b
        )

        # Get ages from most recent fights
        current_fight_ages = [
            df_fighter_a['age'].iloc[0] if 'age' in df_fighter_a.columns else 0,
            df_fighter_b['age_b'].iloc[0] if 'age_b' in df_fighter_b.columns else 0
        ]
        current_fight_age_diff = current_fight_ages[0] - current_fight_ages[1]
        current_fight_age_ratio = self.utils.safe_divide(current_fight_ages[0], current_fight_ages[1])

        # Process Elo stats
        elo_a = df_fighter_a['pre_fight_elo'].iloc[0] if 'pre_fight_elo' in df_fighter_a.columns else 1500
        elo_b = df_fighter_b['pre_fight_elo_b'].iloc[0] if 'pre_fight_elo_b' in df_fighter_b.columns else 1500
        elo_diff = elo_a - elo_b

        # Calculate win probabilities based on Elo
        a_win_prob = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        b_win_prob = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))

        elo_stats = [elo_a, elo_b, elo_diff, a_win_prob, b_win_prob]
        elo_ratio = self.utils.safe_divide(elo_a, elo_b)

        # Process other fighter statistics
        other_stats = self._process_other_stats(df_fighter_a, df_fighter_b)

        # Dummy label (this will be predicted)
        labels = [0]  # Placeholder for winner

        # Use provided current_date or default to now
        if current_date is None:
            current_date = datetime.now()

        # Get most recent fight dates
        most_recent_date_a = df_fighter_a['fight_date'].max() if 'fight_date' in df_fighter_a.columns else None
        most_recent_date_b = df_fighter_b['fight_date'].max() if 'fight_date' in df_fighter_b.columns else None
        most_recent_date = max(most_recent_date_a,
                              most_recent_date_b) if most_recent_date_a and most_recent_date_b else most_recent_date_a or most_recent_date_b

        # Combine all features
        combined_features = np.concatenate([
            fighter_a_features, fighter_b_features, results_fighter_a, results_fighter_b,
            current_fight_odds, [current_fight_odds_diff, current_fight_odds_ratio],
            current_fight_closing_odds, [current_fight_closing_odds_diff, current_fight_closing_odds_ratio],
            current_fight_ages, [current_fight_age_diff, current_fight_age_ratio],
            elo_stats, [elo_ratio], other_stats
        ])

        # Final matchup data
        return [fighter_a, fighter_b, most_recent_date] + combined_features.tolist() + labels + [current_date]

    def _process_other_stats(self, df_fighter_a: pd.DataFrame, df_fighter_b: pd.DataFrame) -> List[float]:
        """Process other fighter statistics."""
        # Win/loss streak stats
        win_streak_a = df_fighter_a['win_streak'].iloc[0] if 'win_streak' in df_fighter_a.columns else 0
        win_streak_b = df_fighter_b['win_streak_b'].iloc[0] if 'win_streak_b' in df_fighter_b.columns else 0
        win_streak_diff = win_streak_a - win_streak_b
        win_streak_ratio = self.utils.safe_divide(win_streak_a, win_streak_b)

        loss_streak_a = df_fighter_a['loss_streak'].iloc[0] if 'loss_streak' in df_fighter_a.columns else 0
        loss_streak_b = df_fighter_b['loss_streak_b'].iloc[0] if 'loss_streak_b' in df_fighter_b.columns else 0
        loss_streak_diff = loss_streak_a - loss_streak_b
        loss_streak_ratio = self.utils.safe_divide(loss_streak_a, loss_streak_b)

        # Experience stats
        exp_a = df_fighter_a['years_of_experience'].iloc[0] if 'years_of_experience' in df_fighter_a.columns else 0
        exp_b = df_fighter_b['years_of_experience_b'].iloc[0] if 'years_of_experience_b' in df_fighter_b.columns else 0
        exp_diff = exp_a - exp_b
        exp_ratio = self.utils.safe_divide(exp_a, exp_b)

        # Last fight stats
        days_since_a = df_fighter_a['days_since_last_fight'].iloc[0] if 'days_since_last_fight' in df_fighter_a.columns else 0
        days_since_b = df_fighter_b['days_since_last_fight_b'].iloc[0] if 'days_since_last_fight_b' in df_fighter_b.columns else 0
        days_since_diff = days_since_a - days_since_b
        days_since_ratio = self.utils.safe_divide(days_since_a, days_since_b)

        return [
            win_streak_a, win_streak_b, win_streak_diff, win_streak_ratio,
            loss_streak_a, loss_streak_b, loss_streak_diff, loss_streak_ratio,
            exp_a, exp_b, exp_diff, exp_ratio,
            days_since_a, days_since_b, days_since_diff, days_since_ratio
        ]

    def _generate_column_names(
        self,
        features_to_include: List[str],
        method_columns: List[str],
        n_past_fights: int,
        include_names: bool
    ) -> List[str]:
        """Generate column names for the matchup DataFrame."""
        # Results columns
        tester = 6 - n_past_fights
        results_columns = []
        for i in range(1, tester + 1):
            results_columns += [
                f"result_fight_{i}", f"winner_fight_{i}", f"weight_class_fight_{i}", f"scheduled_rounds_fight_{i}",
                f"result_b_fight_{i}", f"winner_b_fight_{i}", f"weight_class_b_fight_{i}",
                f"scheduled_rounds_b_fight_{i}"
            ]

        # New feature columns
        new_columns = [
            'current_fight_pre_fight_elo_a', 'current_fight_pre_fight_elo_b', 'current_fight_pre_fight_elo_diff',
            'current_fight_pre_fight_elo_a_win_chance', 'current_fight_pre_fight_elo_b_win_chance',
            'current_fight_pre_fight_elo_ratio', 'current_fight_win_streak_a', 'current_fight_win_streak_b',
            'current_fight_win_streak_diff', 'current_fight_win_streak_ratio', 'current_fight_loss_streak_a',
            'current_fight_loss_streak_b', 'current_fight_loss_streak_diff', 'current_fight_loss_streak_ratio',
            'current_fight_years_experience_a', 'current_fight_years_experience_b',
            'current_fight_years_experience_diff',
            'current_fight_years_experience_ratio', 'current_fight_days_since_last_a',
            'current_fight_days_since_last_b', 'current_fight_days_since_last_diff',
            'current_fight_days_since_last_ratio'
        ]

        # Base columns
        base_columns = ['fight_date'] if not include_names else ['fighter_a', 'fighter_b', 'fight_date']

        # Feature columns
        feature_columns = (
            [f"{feature}_fighter_avg_last_{n_past_fights}" for feature in features_to_include] +
            [f"{feature}_fighter_b_avg_last_{n_past_fights}" for feature in features_to_include]
        )

        # Odds and age columns
        odds_age_columns = [
            'current_fight_open_odds', 'current_fight_open_odds_b', 'current_fight_open_odds_diff',
            'current_fight_open_odds_ratio',
            'current_fight_closing_odds', 'current_fight_closing_odds_b', 'current_fight_closing_odds_diff',
            'current_fight_closing_odds_ratio',
            'current_fight_age', 'current_fight_age_b', 'current_fight_age_diff', 'current_fight_age_ratio'
        ]

        # Combine all column names
        return (
            base_columns + feature_columns + results_columns + odds_age_columns + new_columns +
            [f"{method}" for method in method_columns] + ['current_fight_date']
        )

    def _calculate_matchup_features(
        self,
        df: pd.DataFrame,
        features_to_include: List[str],
        n_past_fights: int
    ) -> pd.DataFrame:
        """Calculate additional differential and ratio features."""
        diff_columns = {}
        ratio_columns = {}

        for feature in features_to_include:
            col_a = f"{feature}_fighter_a_avg_last_{n_past_fights}"
            col_b = f"{feature}_fighter_b_avg_last_{n_past_fights}"

            if col_a in df.columns and col_b in df.columns:
                diff_columns[f"matchup_{feature}_diff_avg_last_{n_past_fights}"] = df[col_a] - df[col_b]
                ratio_columns[f"matchup_{feature}_ratio_avg_last_{n_past_fights}"] = self.utils.safe_divide(
                    df[col_a], df[col_b]
                )

        return pd.concat([df, pd.DataFrame(diff_columns), pd.DataFrame(ratio_columns)], axis=1)


class UFCPredictor:
    """Makes fight predictions using the XGBoost model ensemble."""

    def __init__(self, data_dir: str = "../data",
                 model_dir: str = "../models/xgboost/jan2024-dec2025/dynamicmatchup 200"):
        """Initialize the predictor with data and model directories."""
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.matchup_predictor = FighterMatchupPredictor(data_dir)
        self.models = []
        self.use_ensemble = True
        self.use_calibration = True

    def load_models(self, model_files: List[str], use_ensemble: bool = True, use_calibration: bool = True):
        """Load the XGBoost models."""
        self.use_ensemble = use_ensemble
        self.use_calibration = use_calibration

        if use_ensemble:
            for model_file in model_files:
                model_path = os.path.abspath(f'{self.model_dir}/{model_file}')
                self.models.append(self._load_model(model_path))
        else:
            # Just load the last model if not using ensemble
            model_path = os.path.abspath(f'{self.model_dir}/{model_files[-1]}')
            self.models.append(self._load_model(model_path))

        print(f"Loaded {len(self.models)} model(s)")

    def _load_model(self, model_path: str) -> xgb.XGBClassifier:
        """Load a single XGBoost model from file."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            # Enable categorical feature support
            model = xgb.XGBClassifier(enable_categorical=True)
            model.load_model(model_path)
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def preprocess_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to category type."""
        category_columns = [col for col in data.columns if col.endswith(('fight_1', 'fight_2', 'fight_3'))]
        data = data.copy()
        data[category_columns] = data[category_columns].astype("category")
        return data

    def _make_single_prediction(self, fighter_a: str, fighter_b: str, closing_odds_a: float,
                                closing_odds_b: float, fight_date: str = None, is_swapped: bool = False) -> Dict:
        """
        Make a single prediction with the given fighter order.

        Args:
            fighter_a: Name of first fighter
            fighter_b: Name of second fighter
            closing_odds_a: Closing odds for fighter A
            closing_odds_b: Closing odds for fighter B
            fight_date: Fight date in YYYY-MM-DD format
            is_swapped: Whether this is a swapped prediction

        Returns:
            Dictionary with prediction results
        """
        # Step 1: Load validation data for calibration
        val_data_path = os.path.join(self.data_dir, "train test data", "val_data.csv")
        if not os.path.exists(val_data_path):
            raise FileNotFoundError(f"Validation data not found at {val_data_path}")

        val_data = pd.read_csv(val_data_path)
        display_columns = ['current_fight_date', 'fighter_a', 'fighter_b']
        y_val = val_data['winner']
        X_val = self.preprocess_features(val_data.drop(['winner'] + display_columns, axis=1))

        # Step 2: Create the matchup
        matchup_df = self.matchup_predictor.create_fighter_matchup(
            fighter_a, fighter_b, closing_odds_a, closing_odds_b, fight_date
        )

        # Save fighter names for output
        fighter_a_orig = matchup_df['fighter_a'].iloc[0] if 'fighter_a' in matchup_df.columns else fighter_a
        fighter_b_orig = matchup_df['fighter_b'].iloc[0] if 'fighter_b' in matchup_df.columns else fighter_b

        # Step 3: Prepare matchup data for prediction
        prediction_df = matchup_df.copy()

        # Step 4: Drop display columns if they exist
        for col in display_columns + ['fighter_a', 'fighter_b', 'winner', 'current_fight_date']:
            if col in prediction_df.columns:
                prediction_df = prediction_df.drop(columns=[col])

        # Step 5: Process categorical features
        prediction_df = self.preprocess_features(prediction_df)

        # Step 6: Ensure feature consistency with the model
        model_features = self.models[0].get_booster().feature_names

        # Ensure X_val has all the required features for calibration
        X_val = X_val.reindex(columns=model_features)

        # Ensure prediction_df has all required features
        prediction_df = prediction_df.reindex(columns=model_features)

        # Step 7: Make predictions
        all_predictions = []
        for model in self.models:
            if self.use_calibration:
                # Use a calibrated model
                calibrated_model = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
                calibrated_model.fit(X_val, y_val)
                y_pred_proba = calibrated_model.predict_proba(prediction_df)
            else:
                # Use the model directly
                y_pred_proba = model.predict_proba(prediction_df)

            all_predictions.append(y_pred_proba)

        # Step 8: Process prediction results
        if self.use_ensemble:
            # Average the predictions from all models
            y_pred_proba_avg = np.mean(all_predictions, axis=0)[0]

            # Count models agreeing with the ensemble prediction
            if y_pred_proba_avg[1] > y_pred_proba_avg[0]:
                # Fighter A predicted to win
                predicted_winner = fighter_a_orig
                predicted_winner_probability = y_pred_proba_avg[1]
                models_agreeing = sum([1 for pred in all_predictions if pred[0][1] > pred[0][0]])
            else:
                # Fighter B predicted to win
                predicted_winner = fighter_b_orig
                predicted_winner_probability = y_pred_proba_avg[0]
                models_agreeing = sum([1 for pred in all_predictions if pred[0][0] > pred[0][1]])

            fighter_a_win_prob = y_pred_proba_avg[1]
            fighter_b_win_prob = y_pred_proba_avg[0]

        else:
            # Single model prediction
            y_pred_proba = all_predictions[0][0]
            if y_pred_proba[1] > y_pred_proba[0]:
                # Fighter A predicted to win
                predicted_winner = fighter_a_orig
                predicted_winner_probability = y_pred_proba[1]
            else:
                # Fighter B predicted to win
                predicted_winner = fighter_b_orig
                predicted_winner_probability = y_pred_proba[0]

            fighter_a_win_prob = y_pred_proba[1]
            fighter_b_win_prob = y_pred_proba[0]
            models_agreeing = 1

        # Print prediction results
        position_text = "SWAPPED POSITION" if is_swapped else "ORIGINAL POSITION"
        print(f"\n{position_text} Prediction Results:")
        print(f"{fighter_a_orig} win probability: {fighter_a_win_prob:.2%}")
        print(f"{fighter_b_orig} win probability: {fighter_b_win_prob:.2%}")
        print(f"Predicted winner: {predicted_winner} with {predicted_winner_probability:.2%} probability")
        print(f"Models agreeing: {models_agreeing}/{len(self.models)}")

        return {
            'fighter_a': fighter_a_orig,
            'fighter_b': fighter_b_orig,
            'fighter_a_win_prob': fighter_a_win_prob,
            'fighter_b_win_prob': fighter_b_win_prob,
            'predicted_winner': predicted_winner,
            'predicted_win_prob': predicted_winner_probability,
            'models_agreeing': models_agreeing,
            'total_models': len(self.models)
        }

    def predict_fight(self, fighter_a: str, fighter_b: str, closing_odds_a: float,
                      closing_odds_b: float, fight_date: str = None) -> Dict:
        """
        Predict the outcome of a fight between two fighters using both fighter positions.
        This averages predictions from both original and swapped fighter positions to
        reduce any potential position bias in the model.

        Args:
            fighter_a: Name of first fighter
            fighter_b: Name of second fighter
            closing_odds_a: Closing odds for fighter A
            closing_odds_b: Closing odds for fighter B
            fight_date: Fight date in YYYY-MM-DD format

        Returns:
            Dictionary with prediction results
        """
        if not self.models:
            raise ValueError("No models loaded. Call load_models() first.")

        # Step 1: Make prediction with original fighter order
        print("\n--- Making prediction with original fighter order ---")
        original_prediction = self._make_single_prediction(
            fighter_a, fighter_b, closing_odds_a, closing_odds_b, fight_date, is_swapped=False
        )

        # Step 2: Make prediction with swapped fighter order
        print("\n--- Making prediction with swapped fighter order ---")
        swapped_prediction = self._make_single_prediction(
            fighter_b, fighter_a, closing_odds_b, closing_odds_a, fight_date, is_swapped=True
        )

        # Step 3: Average the win probabilities
        # In swapped prediction, fighter_a refers to fighter_b from original, and vice versa
        fighter_a_win_prob = (original_prediction['fighter_a_win_prob'] +
                              swapped_prediction['fighter_b_win_prob']) / 2
        fighter_b_win_prob = (original_prediction['fighter_b_win_prob'] +
                              swapped_prediction['fighter_a_win_prob']) / 2

        # Normalize to ensure probabilities sum to 1
        total_prob = fighter_a_win_prob + fighter_b_win_prob
        fighter_a_win_prob /= total_prob
        fighter_b_win_prob /= total_prob

        # Step 4: Determine the final predicted winner
        if fighter_a_win_prob > fighter_b_win_prob:
            predicted_winner = original_prediction['fighter_a']
            predicted_win_prob = fighter_a_win_prob
        else:
            predicted_winner = original_prediction['fighter_b']
            predicted_win_prob = fighter_b_win_prob

        # Create the final prediction result
        result = {
            'fighter_a': original_prediction['fighter_a'],
            'fighter_b': original_prediction['fighter_b'],
            'fighter_a_win_prob': fighter_a_win_prob,
            'fighter_b_win_prob': fighter_b_win_prob,
            'predicted_winner': predicted_winner,
            'predicted_win_prob': predicted_win_prob,
            'models_agreeing_original': original_prediction['models_agreeing'],
            'models_agreeing_swapped': swapped_prediction['models_agreeing'],
            'total_models': original_prediction['total_models']
        }

        # Print the final averaged results
        print(f"\n=== FINAL AVERAGED PREDICTION RESULTS ===")
        print(f"{result['fighter_a']} win probability: {result['fighter_a_win_prob']:.2%}")
        print(f"{result['fighter_b']} win probability: {result['fighter_b_win_prob']:.2%}")
        print(f"Predicted winner: {result['predicted_winner']} with {result['predicted_win_prob']:.2%} probability")
        print(f"Models agreeing: Original: {result['models_agreeing_original']}/{result['total_models']}, "
              f"Swapped: {result['models_agreeing_swapped']}/{result['total_models']}")

        return result


def main():
    """Run example prediction with simple output."""
    # Model files
    model_files = [
        'model_0.7071_auc_diff_0.0325.json',
        'model_0.7071_auc_diff_0.0280.json',
        'model_0.7071_auc_diff_0.0246.json',
        'model_0.7071_auc_diff_0.0236.json',
        'model_0.7071_auc_diff_0.0234.json'
    ]

    # Fight details
    fighter_a = "michael johnson"
    fighter_b = "ottman azaitar"

    # Closing odds
    closing_odds_a = -225
    closing_odds_b = 220

    # Specify fight date in YYYY-MM-DD format
    fight_date = "2024-12-14"

    # Create predictor
    predictor = UFCPredictor()

    # Load models
    predictor.load_models(model_files, use_ensemble=True)

    # Make prediction with position swap averaging
    predictor.predict_fight(
        fighter_a,
        fighter_b,
        closing_odds_a,
        closing_odds_b,
        fight_date
    )


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time}")

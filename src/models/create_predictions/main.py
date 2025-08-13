"""
UFC Fighter Matchup Prediction System
Complete system maintaining original functionality and import structure.
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_PAST_FIGHTS = 3
DEFAULT_DATA_DIR = "data"
DEBUG = True

PATHS = {
    'processed_fighter_stats': "processed/combined_sorted_fighter_stats.csv",
    'test_data': "train_test/test_data.csv",
    'removed_features': "train_test/removed_features.txt",
    'live_data': "live_data",
    'matchup_data': "matchup data"
}

IMPORTANT_COLUMNS = [
    'fighter_a', 'fighter_b',
    'current_fight_age', 'current_fight_age_b', 'current_fight_age_diff', 'current_fight_age_ratio',
    'current_fight_win_streak_a', 'current_fight_win_streak_b', 'current_fight_win_streak_diff',
    'current_fight_win_streak_ratio',
    'current_fight_loss_streak_a', 'current_fight_loss_streak_b', 'current_fight_loss_streak_diff',
    'current_fight_loss_streak_ratio',
    'current_fight_years_experience_a', 'current_fight_years_experience_b',
    'current_fight_years_experience_diff', 'current_fight_years_experience_ratio',
    'current_fight_days_since_last_a', 'current_fight_days_since_last_b',
    'current_fight_days_since_last_diff', 'current_fight_days_since_last_ratio',
    'current_fight_pre_fight_elo_a', 'current_fight_pre_fight_elo_b',
    'current_fight_pre_fight_elo_diff', 'current_fight_pre_fight_elo_a_win_chance',
    'current_fight_pre_fight_elo_b_win_chance', 'current_fight_pre_fight_elo_ratio',
    'current_fight_closing_open_diff_a', 'current_fight_closing_open_diff_b'
]

EXCLUDED_COLUMNS = [
    'fighter', 'fighter_lower', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
    'result', 'winner', 'weight_class', 'scheduled_rounds',
    'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b'
]


def get_absolute_path(base_dir, rel_path):
    """Get absolute path from relative path and base directory."""
    return os.path.join(os.path.abspath(base_dir), rel_path)


def resolve_data_dir():
    """Find the data directory from various possible locations."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_dirs = [
        os.path.join(current_dir, "data"),
        os.path.join(os.path.dirname(current_dir), "data"),
        "data",
    ]

    if 'HOME' in os.environ:
        possible_dirs.append(os.path.join(os.environ['HOME'], "UFC/data"))
    if 'USERPROFILE' in os.environ:
        possible_dirs.append(os.path.join(os.environ['USERPROFILE'], "PycharmProjects/UFC/data"))

    for path in possible_dirs:
        if os.path.exists(path):
            return os.path.abspath(path)

    return os.path.abspath(DEFAULT_DATA_DIR)


# ============================================================================
# UTILITY FUNCTIONS WITH FALLBACKS
# ============================================================================

# Try to import original utilities
try:
    from src.data_processing.cleaning.data_cleaner import DataUtils, OddsUtils

    data_utils = DataUtils()
    odds_utils = OddsUtils()
    ORIGINAL_UTILS_AVAILABLE = True
except ImportError:
    ORIGINAL_UTILS_AVAILABLE = False
    print("WARNING: Original utility modules not found. Using fallback implementations.")


class FallbackDataUtils:
    """Fallback implementation of DataUtils if original is not available"""

    def safe_divide(self, a, b):
        if pd.isna(a) or pd.isna(b) or b == 0:
            return 1.0
        return a / b

    def rename_columns_general(self, col):
        new_col = str(col).lower().replace(' ', '_')
        return new_col


class FallbackOddsUtils:
    """Fallback implementation of OddsUtils if original is not available"""

    def process_odds_pair(self, odds_a: float, odds_b: float) -> Tuple[List[float], float, float]:
        decimal_a = (odds_a / 100) + 1 if odds_a > 0 else (100 / abs(odds_a)) + 1
        decimal_b = (odds_b / 100) + 1 if odds_b > 0 else (100 / abs(odds_b)) + 1
        diff = decimal_a - decimal_b
        ratio = decimal_a / decimal_b if decimal_b != 0 else 1.0
        return [decimal_a, decimal_b], diff, ratio


# Create utility objects
if not ORIGINAL_UTILS_AVAILABLE:
    data_utils = FallbackDataUtils()
    odds_utils = FallbackOddsUtils()


def safe_divide(a, b):
    """Safely divide a by b, handling edge cases."""
    return data_utils.safe_divide(a, b)


def process_odds_pair(odds_a: float, odds_b: float) -> Tuple[List[float], float, float]:
    """Process a pair of American odds."""
    return odds_utils.process_odds_pair(odds_a, odds_b)


def rename_column(col):
    """Standardize column name."""
    return data_utils.rename_columns_general(col)


def resolve_fight_date(fight_date):
    """Process the fight date to ensure a valid datetime object."""
    if fight_date is None:
        return datetime.now()
    elif isinstance(fight_date, str):
        try:
            return datetime.strptime(fight_date, '%Y-%m-%d')
        except ValueError:
            print("Invalid date format. Using current date instead.")
            return datetime.now()
    else:
        return fight_date


def generate_matchup_filename(fighter_a, fighter_b):
    """Generate a filename for a matchup between two fighters."""
    return f"{fighter_a.replace(' ', '_')}_vs_{fighter_b.replace(' ', '_')}_matchup.csv"


def ensure_directory_exists(directory):
    """Ensure a directory exists, creating it if necessary."""
    abs_path = os.path.abspath(directory)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def create_result_columns(n_past_fights, tester):
    """Generate column names for fight result columns."""
    results_columns = []
    for i in range(1, tester + 1):
        results_columns += [
            f"result_fight_{i}", f"winner_fight_{i}", f"weight_class_fight_{i}", f"scheduled_rounds_fight_{i}",
            f"result_b_fight_{i}", f"winner_b_fight_{i}", f"weight_class_b_fight_{i}",
            f"scheduled_rounds_b_fight_{i}"
        ]
    return results_columns


# ============================================================================
# FIGHT DATA PROCESSOR WITH FALLBACK
# ============================================================================

# Try to import original FightDataProcessor
try:
    from src.data_processing.cleaning.data_cleaner import FightDataProcessor

    ORIGINAL_PROCESSOR_AVAILABLE = True
except ImportError:
    ORIGINAL_PROCESSOR_AVAILABLE = False
    print("WARNING: Original FightDataProcessor not found. Some functionality may be limited.")


class FightDataProcessorFallback:
    """Fallback implementation if the original FightDataProcessor is not available"""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def _load_csv(self, filepath):
        """Load a CSV file into a pandas DataFrame"""
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            print(f"Error loading CSV file {filepath}: {e}")
            raise

    def _save_csv(self, df, filepath):
        """Save a pandas DataFrame to a CSV file"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)
            print(f"Saved to {filepath}")
        except Exception as e:
            print(f"Error saving CSV file {filepath}: {e}")
            raise


# ============================================================================
# FIGHTER MATCHUP PREDICTOR
# ============================================================================

class FighterMatchupPredictor:
    """Creates matchup data for a specific pair of fighters."""

    def __init__(self, data_dir: str = "data"):
        """Initialize the matchup predictor with a data directory."""
        self.data_dir = os.path.abspath(data_dir)

        # Initialize the fight processor
        if ORIGINAL_PROCESSOR_AVAILABLE:
            self.fight_processor = FightDataProcessor(self.data_dir)
        else:
            self.fight_processor = FightDataProcessorFallback(self.data_dir)

    def create_fighter_matchup(
            self,
            fighter_a: str,
            fighter_b: str,
            open_odds_a: float,
            open_odds_b: float,
            closing_odds_a: float,
            closing_odds_b: float,
            fight_date: Optional[Union[str, datetime]] = None,
            n_past_fights: int = 3,
            save_individual_file: bool = True
    ) -> pd.DataFrame:
        """Create a specific matchup between two fighters."""
        if DEBUG:
            print(f"Creating matchup: {fighter_a} vs {fighter_b}")
            print(f"Open Odds: {fighter_a} ({open_odds_a}) vs {fighter_b} ({open_odds_b})")
            print(f"Closing Odds: {fighter_a} ({closing_odds_a}) vs {fighter_b} ({closing_odds_b})")
            print(f"Using data directory: {self.data_dir}")

        # Process the fight date
        current_date = resolve_fight_date(fight_date)
        if DEBUG:
            print(f"Using fight date: {current_date.strftime('%Y-%m-%d')}")

        # Load combined fighter stats data
        fighter_stats_file = os.path.join(self.data_dir, PATHS['processed_fighter_stats'])
        if DEBUG:
            print(f"Attempting to load fighter stats from: {fighter_stats_file}")

        if not os.path.exists(fighter_stats_file):
            raise FileNotFoundError(f"Fighter stats file not found at: {fighter_stats_file}")

        df = self.fight_processor._load_csv(fighter_stats_file)
        df['fight_date'] = pd.to_datetime(df['fight_date'])
        df['fighter_lower'] = df['fighter'].str.lower()
        fighter_a_lower = fighter_a.lower()
        fighter_b_lower = fighter_b.lower()

        # Check if fighters exist
        if fighter_a_lower not in df['fighter_lower'].values:
            raise ValueError(f"Fighter '{fighter_a}' not found in the dataset")
        if fighter_b_lower not in df['fighter_lower'].values:
            raise ValueError(f"Fighter '{fighter_b}' not found in the dataset")

        # Define columns to exclude from features
        features_to_include = [
            col for col in df.columns if col not in EXCLUDED_COLUMNS
                                         and col != 'age' and not col.endswith('_age')
        ]

        # Calculate tester value
        tester = 6 - n_past_fights

        # Get recent fights for each fighter
        fighter_a_df = df[(df['fighter_lower'] == fighter_a_lower) & (df['fight_date'] < current_date)] \
            .sort_values(by='fight_date', ascending=False).head(n_past_fights)
        fighter_b_df = df[(df['fighter_lower'] == fighter_b_lower) & (df['fight_date'] < current_date)] \
            .sort_values(by='fight_date', ascending=False).head(n_past_fights)

        if len(fighter_a_df) == 0 or len(fighter_b_df) == 0:
            raise ValueError("Not enough fight data available for one or both fighters")

        # Extract features from past fights
        fighter_a_features = fighter_a_df[features_to_include].mean().values
        fighter_b_features = fighter_b_df[features_to_include].mean().values

        # Extract and pad recent fight results
        num_a_results = min(len(fighter_a_df), tester)
        num_b_results = min(len(fighter_b_df), tester)
        results_fighter_a = fighter_a_df[['result', 'winner', 'weight_class', 'scheduled_rounds']] \
            .head(num_a_results).values.flatten() if num_a_results > 0 else np.array([])
        results_fighter_b = fighter_b_df[['result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']] \
            .head(num_b_results).values.flatten() if num_b_results > 0 else np.array([])
        results_fighter_a = np.pad(results_fighter_a, (0, tester * 4 - len(results_fighter_a)),
                                   'constant', constant_values=np.nan)
        results_fighter_b = np.pad(results_fighter_b, (0, tester * 4 - len(results_fighter_b)),
                                   'constant', constant_values=np.nan)

        # Calculate current stats for both fighters
        current_stats_a = self._calculate_fighter_stats(fighter_a_lower, df, current_date, n_past_fights)
        current_stats_b = self._calculate_fighter_stats(fighter_b_lower, df, current_date, n_past_fights)

        age_a, exp_a, days_since_a, win_streak_a, loss_streak_a = current_stats_a
        age_b, exp_b, days_since_b, win_streak_b, loss_streak_b = current_stats_b

        current_fight_age_diff = age_a - age_b
        current_fight_age_ratio = safe_divide(age_a, age_b)

        # Get ELO ratings
        if 'fight_outcome_elo' in fighter_a_df.columns and 'fight_outcome_elo' in fighter_b_df.columns:
            elo_a = fighter_a_df['fight_outcome_elo'].iloc[0]
            elo_b = fighter_b_df['fight_outcome_elo'].iloc[0]
        elif 'pre_fight_elo' in fighter_a_df.columns and 'pre_fight_elo_b' in fighter_b_df.columns:
            elo_a = fighter_a_df['pre_fight_elo'].iloc[0]
            elo_b = fighter_b_df['pre_fight_elo_b'].iloc[0]
        else:
            elo_a, elo_b = 1500, 1500
        elo_diff = elo_a - elo_b

        # Create current fight stats
        current_fight = pd.Series({
            'fighter': fighter_a,
            'fighter_b': fighter_b,
            'open_odds': open_odds_a,
            'open_odds_b': open_odds_b,
            'closing_range_end': closing_odds_a,
            'closing_range_end_b': closing_odds_b,
            'age': age_a,
            'age_b': age_b,
            'pre_fight_elo': elo_a,
            'pre_fight_elo_b': elo_b,
            'pre_fight_elo_diff': elo_diff,
            'win_streak': win_streak_a,
            'win_streak_b': win_streak_b,
            'loss_streak': loss_streak_a,
            'loss_streak_b': loss_streak_b,
            'years_of_experience': exp_a,
            'years_of_experience_b': exp_b,
            'days_since_last_fight': days_since_a,
            'days_since_last_fight_b': days_since_b,
            'fight_date': current_date
        })

        # Process odds
        current_fight_odds, current_fight_odds_diff, current_fight_odds_ratio = \
            process_odds_pair(current_fight['open_odds'], current_fight['open_odds_b'])

        # Process closing odds
        current_fight_closing_odds, current_fight_closing_odds_diff, current_fight_closing_odds_ratio = \
            process_odds_pair(current_fight['closing_range_end'], current_fight['closing_range_end_b'])

        # Calculate the difference between closing and opening odds
        current_fight_closing_open_diff_a = current_fight['closing_range_end'] - current_fight['open_odds']
        current_fight_closing_open_diff_b = current_fight['closing_range_end_b'] - current_fight['open_odds_b']

        # Process ages
        current_fight_ages = [current_fight['age'], current_fight['age_b']]

        # Process ELO stats
        elo_a_win_prob = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        elo_b_win_prob = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))
        elo_ratio = safe_divide(elo_a, elo_b)
        elo_stats = [elo_a, elo_b, elo_diff, elo_a_win_prob, elo_b_win_prob]

        # Create explicit other stats
        win_streak_diff = win_streak_a - win_streak_b
        win_streak_ratio = safe_divide(win_streak_a, win_streak_b)
        loss_streak_diff = loss_streak_a - loss_streak_b
        loss_streak_ratio = safe_divide(loss_streak_a, loss_streak_b)
        exp_diff = exp_a - exp_b
        exp_ratio = safe_divide(exp_a, exp_b)
        days_since_diff = days_since_a - days_since_b
        days_since_ratio = safe_divide(days_since_a, days_since_b)

        # Create comprehensive other_stats array
        other_stats = [
            win_streak_a, win_streak_b, win_streak_diff, win_streak_ratio,
            loss_streak_a, loss_streak_b, loss_streak_diff, loss_streak_ratio,
            exp_a, exp_b, exp_diff, exp_ratio,
            days_since_a, days_since_b, days_since_diff, days_since_ratio
        ]

        # Dummy label for prediction
        labels = [0]

        # Determine the most recent fight date
        most_recent_date_a = fighter_a_df['fight_date'].max() if len(fighter_a_df) > 0 else None
        most_recent_date_b = fighter_b_df['fight_date'].max() if len(fighter_b_df) > 0 else None
        most_recent_date = (max(most_recent_date_a, most_recent_date_b)
                            if most_recent_date_a and most_recent_date_b
                            else most_recent_date_a or most_recent_date_b)

        # Create a dictionary to map column names to values
        matchup_values = {}

        # Map all the calculated statistics to their expected column names
        matchup_values['current_fight_age'] = age_a
        matchup_values['current_fight_age_b'] = age_b
        matchup_values['current_fight_age_diff'] = current_fight_age_diff
        matchup_values['current_fight_age_ratio'] = current_fight_age_ratio

        matchup_values['current_fight_win_streak_a'] = win_streak_a
        matchup_values['current_fight_win_streak_b'] = win_streak_b
        matchup_values['current_fight_win_streak_diff'] = win_streak_diff
        matchup_values['current_fight_win_streak_ratio'] = win_streak_ratio

        matchup_values['current_fight_loss_streak_a'] = loss_streak_a
        matchup_values['current_fight_loss_streak_b'] = loss_streak_b
        matchup_values['current_fight_loss_streak_diff'] = loss_streak_diff
        matchup_values['current_fight_loss_streak_ratio'] = loss_streak_ratio

        matchup_values['current_fight_years_experience_a'] = exp_a
        matchup_values['current_fight_years_experience_b'] = exp_b
        matchup_values['current_fight_years_experience_diff'] = exp_diff
        matchup_values['current_fight_years_experience_ratio'] = exp_ratio

        matchup_values['current_fight_days_since_last_a'] = days_since_a
        matchup_values['current_fight_days_since_last_b'] = days_since_b
        matchup_values['current_fight_days_since_last_diff'] = days_since_diff
        matchup_values['current_fight_days_since_last_ratio'] = days_since_ratio

        matchup_values['current_fight_pre_fight_elo_a'] = elo_a
        matchup_values['current_fight_pre_fight_elo_b'] = elo_b
        matchup_values['current_fight_pre_fight_elo_diff'] = elo_diff
        matchup_values['current_fight_pre_fight_elo_a_win_chance'] = elo_a_win_prob
        matchup_values['current_fight_pre_fight_elo_b_win_chance'] = elo_b_win_prob
        matchup_values['current_fight_pre_fight_elo_ratio'] = elo_ratio

        matchup_values['current_fight_closing_open_diff_a'] = current_fight_closing_open_diff_a
        matchup_values['current_fight_closing_open_diff_b'] = current_fight_closing_open_diff_b

        # Combine all features
        combined_features = np.concatenate([
            fighter_a_features, fighter_b_features,
            results_fighter_a, results_fighter_b,
            current_fight_odds, [current_fight_odds_diff, current_fight_odds_ratio],
            current_fight_closing_odds, [current_fight_closing_odds_diff, current_fight_closing_odds_ratio,
                                         current_fight_closing_open_diff_a, current_fight_closing_open_diff_b],
            current_fight_ages, [current_fight_age_diff, current_fight_age_ratio],
            elo_stats, [elo_ratio],
            other_stats
        ])

        # Generate column names
        column_names = self._generate_column_names(features_to_include, ['winner'], n_past_fights, tester, True)
        matchup_data = [fighter_a, fighter_b, most_recent_date] + combined_features.tolist() + labels + [current_date]
        matchup_df = pd.DataFrame([matchup_data], columns=column_names)
        matchup_df.columns = [rename_column(col) for col in matchup_df.columns]

        # Add/update columns with explicitly calculated statistics
        for col_name, col_value in matchup_values.items():
            matchup_df[col_name] = col_value

        matchup_df = self._calculate_matchup_features(matchup_df, features_to_include, n_past_fights)

        # Remove unnecessary columns
        removed_features_file = os.path.join(self.data_dir, PATHS['removed_features'])
        if os.path.exists(removed_features_file):
            with open(removed_features_file, 'r') as f:
                removed_features = [line.strip() for line in f if line.strip()]
            columns_to_drop = [col for col in removed_features if col in matchup_df.columns]
            if columns_to_drop:
                matchup_df = matchup_df.drop(columns=columns_to_drop)
        if 'fight_date' in matchup_df.columns:
            matchup_df = matchup_df.drop(columns=['fight_date'])

        # Create a copy of the DataFrame to ensure we preserve calculated values
        pre_aligned_df = matchup_df.copy()

        # Align with test dataset features if available
        test_data_file = os.path.join(self.data_dir, PATHS['test_data'])
        if os.path.exists(test_data_file):
            test_df = self.fight_processor._load_csv(test_data_file)
            test_columns = test_df.columns.tolist()

            # First merge all columns from test data
            all_columns = test_columns.copy()

            # Add our important columns if they're not already in the test data
            for col in IMPORTANT_COLUMNS:
                if col not in all_columns and col in matchup_df.columns:
                    all_columns.append(col)

            # Create new DataFrame with the desired columns
            final_df = pd.DataFrame(columns=all_columns)

            # Copy data from original DataFrame
            for col in all_columns:
                if col in matchup_df.columns:
                    final_df[col] = matchup_df[col]
                else:
                    final_df[col] = np.nan

            matchup_df = final_df

            if DEBUG:
                print(f"Columns from test data: {len(test_columns)}")
                print(f"Important columns kept: {sum(1 for col in IMPORTANT_COLUMNS if col in matchup_df.columns)}")
                print(f"Total columns kept: {len(matchup_df.columns)}")
        else:
            print(f"Warning: test_data.csv not found at {test_data_file}")

        # Verify the values in important columns were maintained
        for col in IMPORTANT_COLUMNS:
            if col in matchup_df.columns and col in pre_aligned_df.columns:
                if pd.isna(matchup_df[col].iloc[0]) and not pd.isna(pre_aligned_df[col].iloc[0]):
                    print(f"WARNING: Value lost during alignment for {col}")
                    matchup_df[col] = pre_aligned_df[col]

        # Save the matchup data if requested
        if save_individual_file:
            matchup_dir = os.path.join(self.data_dir, PATHS['live_data'])
            ensure_directory_exists(matchup_dir)

            output_filename = os.path.join(
                matchup_dir,
                generate_matchup_filename(fighter_a, fighter_b)
            )
            self.fight_processor._save_csv(matchup_df, output_filename)
            if DEBUG:
                print(f"Created matchup prediction data for {fighter_a} vs {fighter_b}")
                print(f"Output saved to: {output_filename}")

        # Debug info
        if DEBUG:
            print(f"Final DataFrame has {len(matchup_df.columns)} columns including:")
            for col in IMPORTANT_COLUMNS:
                if col in matchup_df.columns:
                    print(f"- {col}: {matchup_df[col].values[0]}")
                else:
                    print(f"- {col}: NOT PRESENT")

        return matchup_df

    def _calculate_fighter_stats(
            self,
            fighter_name_lower: str,
            df: pd.DataFrame,
            current_date: datetime,
            n_past_fights: int
    ) -> tuple:
        """Calculate fighter statistics."""
        fighter_all_fights = df[(df['fighter_lower'] == fighter_name_lower) & (df['fight_date'] < current_date)]
        fighter_all_fights = fighter_all_fights.sort_values(by='fight_date', ascending=False)
        fighter_recent_fights = fighter_all_fights.head(n_past_fights)

        if fighter_recent_fights.empty:
            print(f"No fight data found for fighter with lowercase name: {fighter_name_lower}")
            return np.nan, np.nan, np.nan, np.nan, np.nan

        # Get necessary dates
        last_fight_date = fighter_recent_fights['fight_date'].iloc[0]
        first_fight_date = fighter_all_fights['fight_date'].iloc[-1] if len(fighter_all_fights) > 0 else last_fight_date

        # Calculate days since last fight
        days_since_last_fight = (current_date - last_fight_date).days

        # Extract age and calculate current age
        if 'age' in fighter_recent_fights.columns:
            last_known_age = fighter_recent_fights['age'].iloc[0]
            if pd.isna(last_known_age):
                last_known_age = 30
                print(f"Using default age 30 for {fighter_name_lower} (age was NaN)")
        else:
            last_known_age = 30
            print(f"Age column not found for {fighter_name_lower}, using default age 30")

        current_age = np.ceil(last_known_age + days_since_last_fight / 365.25)
        years_of_experience = (current_date - first_fight_date).days / 365.25

        # Get winning and losing streaks
        if 'win_streak' in fighter_recent_fights.columns:
            win_streak = fighter_recent_fights['win_streak'].iloc[0]
            if pd.isna(win_streak):
                win_streak = 0
                print(f"Win streak was NaN for {fighter_name_lower}, using 0")
        else:
            win_streak = 0
            print(f"Win streak column not found for {fighter_name_lower}")

        if 'loss_streak' in fighter_recent_fights.columns:
            loss_streak = fighter_recent_fights['loss_streak'].iloc[0]
            if pd.isna(loss_streak):
                loss_streak = 0
                print(f"Loss streak was NaN for {fighter_name_lower}, using 0")
        else:
            loss_streak = 0
            print(f"Loss streak column not found for {fighter_name_lower}")

        # Adjust streaks based on most recent result
        if 'winner' in fighter_recent_fights.columns:
            most_recent_result = fighter_recent_fights['winner'].iloc[0]
            if not pd.isna(most_recent_result):
                if most_recent_result == 1:
                    win_streak += 1
                    loss_streak = 0
                elif most_recent_result == 0:
                    loss_streak += 1
                    win_streak = 0
            else:
                print(f"Most recent result was NaN for {fighter_name_lower}")
        else:
            print(f"Winner column not found for {fighter_name_lower}")

        if DEBUG:
            print(f"Calculated stats for {fighter_name_lower}: age={current_age}, exp={years_of_experience}, "
                  f"days={days_since_last_fight}, win_streak={win_streak}, loss_streak={loss_streak}")

        return current_age, years_of_experience, days_since_last_fight, win_streak, loss_streak

    def _generate_column_names(
            self,
            features_to_include: List[str],
            method_columns: List[str],
            n_past_fights: int,
            tester: int,
            include_names: bool
    ) -> List[str]:
        """Generate column names for the matchup DataFrame."""
        results_columns = create_result_columns(n_past_fights, tester)

        new_columns = [
            'current_fight_pre_fight_elo_a', 'current_fight_pre_fight_elo_b', 'current_fight_pre_fight_elo_diff',
            'current_fight_pre_fight_elo_a_win_chance', 'current_fight_pre_fight_elo_b_win_chance',
            'current_fight_pre_fight_elo_ratio', 'current_fight_win_streak_a', 'current_fight_win_streak_b',
            'current_fight_win_streak_diff', 'current_fight_win_streak_ratio',
            'current_fight_loss_streak_a', 'current_fight_loss_streak_b', 'current_fight_loss_streak_diff',
            'current_fight_loss_streak_ratio', 'current_fight_years_experience_a',
            'current_fight_years_experience_b', 'current_fight_years_experience_diff',
            'current_fight_years_experience_ratio', 'current_fight_days_since_last_a',
            'current_fight_days_since_last_b', 'current_fight_days_since_last_diff',
            'current_fight_days_since_last_ratio'
        ]
        base_columns = ['fight_date'] if not include_names else ['fighter_a', 'fighter_b', 'fight_date']
        feature_columns = (
                [f"{feature}_fighter_avg_last_{n_past_fights}" for feature in features_to_include] +
                [f"{feature}_fighter_b_avg_last_{n_past_fights}" for feature in features_to_include]
        )
        odds_age_columns = [
            'current_fight_open_odds', 'current_fight_open_odds_b', 'current_fight_open_odds_diff',
            'current_fight_open_odds_ratio', 'current_fight_closing_odds', 'current_fight_closing_odds_b',
            'current_fight_closing_odds_diff', 'current_fight_closing_odds_ratio',
            'current_fight_closing_open_diff_a', 'current_fight_closing_open_diff_b',
            'current_fight_age', 'current_fight_age_b', 'current_fight_age_diff', 'current_fight_age_ratio'
        ]
        return base_columns + feature_columns + results_columns + odds_age_columns + new_columns + \
            [f"{method}" for method in method_columns] + ['current_fight_date']

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
                ratio_columns[f"matchup_{feature}_ratio_avg_last_{n_past_fights}"] = \
                    safe_divide(df[col_a], df[col_b])
        return pd.concat([df, pd.DataFrame(diff_columns), pd.DataFrame(ratio_columns)], axis=1)


# ============================================================================
# UFC MATCHUP CREATOR
# ============================================================================

class UFCMatchupCreator:
    """Creates matchup data for UFC fights."""

    def __init__(self, data_dir: str = "data"):
        """Initialize the matchup creator with a data directory."""
        self.data_dir = os.path.abspath(data_dir)
        self.matchup_predictor = FighterMatchupPredictor(self.data_dir)

    def create_matchup(
            self,
            fighter_a: str,
            fighter_b: str,
            open_odds_a: float,
            open_odds_b: float,
            closing_odds_a: float,
            closing_odds_b: float,
            fight_date: str = None
    ) -> pd.DataFrame:
        """Create a matchup file for a UFC fight."""
        return self.matchup_predictor.create_fighter_matchup(
            fighter_a, fighter_b, open_odds_a, open_odds_b, closing_odds_a, closing_odds_b, fight_date
        )

    def create_multiple_matchups(
            self,
            matchups: List[Dict],
            output_filename: str = "all_matchups.csv"
    ) -> pd.DataFrame:
        """Create multiple matchups for UFC fights and combine them into a single DataFrame."""
        all_dfs = []
        for i, matchup in enumerate(matchups):
            if DEBUG:
                print(f"\nProcessing matchup {i + 1} of {len(matchups)}")

            fighter_a = matchup['fighter_a']
            fighter_b = matchup['fighter_b']
            open_odds_a = matchup['open_odds_a']
            open_odds_b = matchup['open_odds_b']
            closing_odds_a = matchup['closing_odds_a']
            closing_odds_b = matchup['closing_odds_b']
            fight_date = matchup.get('fight_date', None)

            try:
                df = self.matchup_predictor.create_fighter_matchup(
                    fighter_a, fighter_b, open_odds_a, open_odds_b,
                    closing_odds_a, closing_odds_b, fight_date,
                    save_individual_file=False
                )
                all_dfs.append(df)
            except Exception as e:
                print(f"Error creating matchup {fighter_a} vs {fighter_b}: {str(e)}")

        if not all_dfs:
            print("No matchups were successfully created")
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)

        if 'fighter_a' in combined_df.columns:
            combined_df = combined_df.sort_values(by='fighter_a')

        matchup_dir = os.path.join(self.data_dir, PATHS['matchup_data'])
        ensure_directory_exists(matchup_dir)

        output_path = os.path.join(matchup_dir, output_filename)
        self.matchup_predictor.fight_processor._save_csv(combined_df, output_path)
        if DEBUG:
            print(f"\nCreated {len(all_dfs)} matchups successfully out of {len(matchups)} requested")
            print(f"Combined output saved to: {output_path}")

        return combined_df


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

EXAMPLE_MATCHUPS = [
    {
        'fighter_a': "Andre Fili",
        'fighter_b': "Christian Rodriguez",
        'open_odds_a': 100,
        'open_odds_b': 100,
        'closing_odds_a': 207,
        'closing_odds_b': -242,
        'fight_date': "2025-08-09"
    },
    {
        'fighter_a': "Cody Brundage",
        'fighter_b': "Eric McConico",
        'open_odds_a': 100,
        'open_odds_b': 100,
        'closing_odds_a': -162,
        'closing_odds_b': 143,
        'fight_date': "2025-08-09"
    },
    {
        'fighter_a': "Jean Matsumoto",
        'fighter_b': "Miles Johns",
        'open_odds_a': 100,
        'open_odds_b': 100,
        'closing_odds_a': -244,
        'closing_odds_b': 208,
        'fight_date': "2025-08-09"
    }
]


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create UFC fighter matchups for prediction')

    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directory containing the data files')
    parser.add_argument('--single', action='store_true',
                        help='Create only a single matchup (first one from the list)')
    parser.add_argument('--output', type=str, default='all_matchups.csv',
                        help='Output filename for combined matchups')
    parser.add_argument('--fighters', nargs=2, metavar=('FIGHTER_A', 'FIGHTER_B'),
                        help='Fighter names for manual matchup creation')
    parser.add_argument('--odds', nargs=4, type=float, metavar=('OPEN_A', 'OPEN_B', 'CLOSE_A', 'CLOSE_B'),
                        help='Odds for manual matchup (open A, open B, close A, close B)')
    parser.add_argument('--date', type=str, default=None,
                        help='Fight date in YYYY-MM-DD format')

    return parser.parse_args()


def main():
    """Create fighter matchup files."""
    args = parse_args()

    # Determine data directory
    data_dir = args.data_dir if args.data_dir else resolve_data_dir()
    print(f"Using data directory: {data_dir}")

    # Create matchup creator
    matchup_creator = UFCMatchupCreator(data_dir)

    try:
        # Check if we have a manual matchup definition
        if args.fighters and args.odds:
            fighter_a, fighter_b = args.fighters
            open_odds_a, open_odds_b, closing_odds_a, closing_odds_b = args.odds

            print(f"\nCreating matchup for {fighter_a} vs {fighter_b}")
            matchup_creator.create_matchup(
                fighter_a, fighter_b,
                open_odds_a, open_odds_b,
                closing_odds_a, closing_odds_b,
                args.date
            )
        # Use the example matchups
        elif args.single:
            print("\nCreating a single matchup:")
            matchup = EXAMPLE_MATCHUPS[0]
            matchup_creator.create_matchup(
                matchup['fighter_a'],
                matchup['fighter_b'],
                matchup['open_odds_a'],
                matchup['open_odds_b'],
                matchup['closing_odds_a'],
                matchup['closing_odds_b'],
                matchup['fight_date']
            )
        else:
            print("\nCreating multiple matchups:")
            matchup_creator.create_multiple_matchups(EXAMPLE_MATCHUPS, args.output)

    except Exception as e:
        print(f"Error creating matchup(s): {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time}")

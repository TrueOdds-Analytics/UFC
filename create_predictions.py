import os
from datetime import datetime
from typing import List, Dict, Optional, Union

import numpy as np
import pandas as pd

# Import from the original module
from data_manipulation.data_cleaner import FightDataProcessor, DataUtils, OddsUtils


class FighterMatchupPredictor:
    """Creates matchup data for a specific pair of fighters."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the matchup predictor with a data directory.

        Args:
            data_dir: Directory containing data files.
        """
        # Use absolute path resolution to avoid path concatenation issues
        self.data_dir = os.path.abspath(data_dir)
        self.fight_processor = FightDataProcessor(self.data_dir)
        self.utils = DataUtils()
        self.odds_utils = OddsUtils()

    def create_fighter_matchup(
            self,
            fighter_a: str,
            fighter_b: str,
            open_odds_a: float,
            open_odds_b: float,
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
            open_odds_a: Opening odds for fighter A
            open_odds_b: Opening odds for fighter B
            closing_odds_a: Closing odds for fighter A
            closing_odds_b: Closing odds for fighter B
            fight_date: Date of the fight in YYYY-MM-DD format or datetime object
            n_past_fights: Number of past fights to consider for statistics (default: 3)

        Returns:
            DataFrame with the matchup data formatted for prediction
        """
        print(f"Creating matchup: {fighter_a} vs {fighter_b}")
        print(f"Open Odds: {fighter_a} ({open_odds_a}) vs {fighter_b} ({open_odds_b})")
        print(f"Closing Odds: {fighter_a} ({closing_odds_a}) vs {fighter_b} ({closing_odds_b})")
        print(f"Using data directory: {self.data_dir}")

        # Process the fight date
        if fight_date is None:
            current_date = datetime.now()
            print(f"Using current date: {current_date.strftime('%Y-%m-%d')}")
        elif isinstance(fight_date, str):
            try:
                current_date = datetime.strptime(fight_date, '%Y-%m-%d')
                print(f"Using specified fight date: {fight_date}")
            except ValueError:
                print("Invalid date format. Using current date instead.")
                current_date = datetime.now()
        else:
            current_date = fight_date
            print(f"Using specified fight date: {current_date.strftime('%Y-%m-%d')}")

        # Load combined fighter stats data
        fighter_stats_file = os.path.join(self.data_dir, "combined_sorted_fighter_stats.csv")
        print(f"Attempting to load fighter stats from: {fighter_stats_file}")

        # Check if file exists before loading
        if not os.path.exists(fighter_stats_file):
            raise FileNotFoundError(f"Fighter stats file not found at: {fighter_stats_file}")

        df = self.fight_processor._load_csv(fighter_stats_file)

        # Convert dates
        df['fight_date'] = pd.to_datetime(df['fight_date'])

        # Convert fighter names to lowercase for case-insensitive comparison
        df['fighter_lower'] = df['fighter'].str.lower()
        fighter_a_lower = fighter_a.lower()
        fighter_b_lower = fighter_b.lower()

        # Check if fighters exist in the dataset (case-insensitive)
        if fighter_a_lower not in df['fighter_lower'].values:
            raise ValueError(f"Fighter '{fighter_a}' not found in the dataset")
        if fighter_b_lower not in df['fighter_lower'].values:
            raise ValueError(f"Fighter '{fighter_b}' not found in the dataset")

        # Define columns to exclude from features
        columns_to_exclude = [
            'fighter', 'fighter_lower', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
            'result', 'winner', 'weight_class', 'scheduled_rounds',
            'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b'
        ]
        features_to_include = [
            col for col in df.columns if col not in columns_to_exclude
                                         and col != 'age' and not col.endswith('_age')
        ]

        # Calculate tester value (consistent with MatchupProcessor)
        tester = 6 - n_past_fights

        # Get recent fights for each fighter (before the current fight date)
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

        # Calculate current stats for both fighters using the working implementation approach
        current_stats_a = self._calculate_fighter_stats_fixed(fighter_a_lower, df, current_date, n_past_fights)
        current_stats_b = self._calculate_fighter_stats_fixed(fighter_b_lower, df, current_date, n_past_fights)

        # Debug output to verify stats were calculated correctly
        print(f"Stats for {fighter_a}: {current_stats_a}")
        print(f"Stats for {fighter_b}: {current_stats_b}")

        age_a, exp_a, days_since_a, win_streak_a, loss_streak_a = current_stats_a
        age_b, exp_b, days_since_b, win_streak_b, loss_streak_b = current_stats_b

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
        # Include all required fields explicitly
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
            self._process_fight_odds(current_fight['open_odds'], current_fight['open_odds_b'])

        # Process closing odds
        current_fight_closing_odds, current_fight_closing_odds_diff, current_fight_closing_odds_ratio = \
            self._process_fight_odds(current_fight['closing_range_end'], current_fight['closing_range_end_b'])

        # Calculate the difference between closing and opening odds for each fighter
        current_fight_closing_open_diff_a = current_fight['closing_range_end'] - current_fight['open_odds']
        current_fight_closing_open_diff_b = current_fight['closing_range_end_b'] - current_fight['open_odds_b']

        # Process ages
        current_fight_ages = [current_fight['age'], current_fight['age_b']]
        current_fight_age_diff = current_fight['age'] - current_fight['age_b']
        current_fight_age_ratio = self.utils.safe_divide(current_fight['age'], current_fight['age_b'])

        # Process ELO stats
        elo_a_win_prob = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        elo_b_win_prob = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))
        elo_ratio = self.utils.safe_divide(elo_a, elo_b)
        elo_stats = [elo_a, elo_b, elo_diff, elo_a_win_prob, elo_b_win_prob]

        # Create explicit other stats
        win_streak_diff = win_streak_a - win_streak_b
        win_streak_ratio = self.utils.safe_divide(win_streak_a, win_streak_b)
        loss_streak_diff = loss_streak_a - loss_streak_b
        loss_streak_ratio = self.utils.safe_divide(loss_streak_a, loss_streak_b)
        exp_diff = exp_a - exp_b
        exp_ratio = self.utils.safe_divide(exp_a, exp_b)
        days_since_diff = days_since_a - days_since_b
        days_since_ratio = self.utils.safe_divide(days_since_a, days_since_b)

        # Create comprehensive other_stats array with all required stats
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

        # Add the new closing-opening odds difference columns
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
        matchup_df.columns = [self.utils.rename_columns_general(col) for col in matchup_df.columns]

        # Add/update columns with explicitly calculated statistics
        for col_name, col_value in matchup_values.items():
            matchup_df[col_name] = col_value

        matchup_df = self._calculate_matchup_features(matchup_df, features_to_include, n_past_fights)

        # Remove unnecessary columns
        removed_features_file = os.path.join(self.data_dir, "train test data", "removed_features.txt")
        if os.path.exists(removed_features_file):
            with open(removed_features_file, 'r') as f:
                removed_features = [line.strip() for line in f if line.strip()]
            columns_to_drop = [col for col in removed_features if col in matchup_df.columns]
            if columns_to_drop:
                matchup_df = matchup_df.drop(columns=columns_to_drop)
        if 'fight_date' in matchup_df.columns:
            matchup_df = matchup_df.drop(columns=['fight_date'])

        # Define important columns that should always be kept
        important_columns = [
            'fighter_a', 'fighter_b',
            'current_fight_age', 'current_fight_age_b', 'current_fight_age_diff', 'current_fight_age_ratio',
            'current_fight_win_streak_a', 'current_fight_win_streak_b', 'current_fight_win_streak_diff',
            'current_fight_win_streak_ratio',
            'current_fight_loss_streak_a', 'current_fight_loss_streak_b', 'current_fight_loss_streak_diff',
            'current_fight_loss_streak_ratio',
            'current_fight_years_experience_a', 'current_fight_years_experience_b',
            'current_fight_years_experience_diff', 'current_fight_years_experience_ratio',
            'current_fight_days_since_last_a', 'current_fight_days_since_last_b', 'current_fight_days_since_last_diff',
            'current_fight_days_since_last_ratio',
            'current_fight_pre_fight_elo_a', 'current_fight_pre_fight_elo_b', 'current_fight_pre_fight_elo_diff',
            'current_fight_pre_fight_elo_a_win_chance', 'current_fight_pre_fight_elo_b_win_chance',
            'current_fight_pre_fight_elo_ratio',
            'current_fight_closing_open_diff_a', 'current_fight_closing_open_diff_b'  # Added new columns
        ]

        # Create a copy of the DataFrame to ensure we preserve calculated values
        pre_aligned_df = matchup_df.copy()

        # Align with test dataset features if available
        test_data_file = os.path.join(self.data_dir, "train test data", "test_data.csv")
        if os.path.exists(test_data_file):
            test_df = self.fight_processor._load_csv(test_data_file)
            test_columns = test_df.columns.tolist()

            # First merge all columns from test data
            all_columns = test_columns.copy()

            # Add our important columns if they're not already in the test data
            for col in important_columns:
                if col not in all_columns and col in matchup_df.columns:
                    all_columns.append(col)

            # Create new DataFrame with the desired columns, filling missing ones with NaN
            final_df = pd.DataFrame(columns=all_columns)

            # Copy data from original DataFrame, preserving our calculated values
            for col in all_columns:
                if col in matchup_df.columns:
                    final_df[col] = matchup_df[col]
                else:
                    final_df[col] = np.nan

            # Update matchup_df with the aligned DataFrame
            matchup_df = final_df

            print(f"Columns from test data: {len(test_columns)}")
            print(f"Important columns kept: {sum(1 for col in important_columns if col in matchup_df.columns)}")
            print(f"Total columns kept: {len(matchup_df.columns)}")
        else:
            print(f"Warning: test_data.csv not found at {test_data_file}")

        # Verify the values in important columns were maintained
        for col in important_columns:
            if col in matchup_df.columns and col in pre_aligned_df.columns:
                if pd.isna(matchup_df[col].iloc[0]) and not pd.isna(pre_aligned_df[col].iloc[0]):
                    print(f"WARNING: Value lost during alignment for {col}")
                    # Restore the value from the pre-aligned DataFrame
                    matchup_df[col] = pre_aligned_df[col]

        # Create matchup data directory if it doesn't exist
        matchup_dir = os.path.join(self.data_dir, 'matchup data')
        os.makedirs(matchup_dir, exist_ok=True)

        # Save the matchup data
        output_filename = os.path.join(matchup_dir, 'specific_matchup_data.csv')
        self.fight_processor._save_csv(matchup_df, output_filename)
        print(f"Created matchup prediction data for {fighter_a} vs {fighter_b}")
        print(f"Output saved to: {output_filename}")

        # Debug info - print all columns
        print(f"Final DataFrame has {len(matchup_df.columns)} columns including:")
        for col in important_columns:
            if col in matchup_df.columns:
                print(f"- {col}: {matchup_df[col].values[0]}")
            else:
                print(f"- {col}: NOT PRESENT")

        return matchup_df

    def _calculate_fighter_stats_fixed(
            self,
            fighter_name_lower: str,
            df: pd.DataFrame,
            current_date: datetime,
            n_past_fights: int
    ) -> tuple:
        """
        Calculate fighter statistics using the same approach as the working implementation.
        Uses lowercase fighter name for consistent matching.

        Args:
            fighter_name_lower: Lowercase fighter name to search by
            df: DataFrame with historical fight data
            current_date: Date of the upcoming fight
            n_past_fights: Number of past fights to consider

        Returns:
            Tuple of (age, experience, days_since_last_fight, win_streak, loss_streak)
        """
        # Use lowercase comparison for consistent results
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
                # Use a default age if it's NaN
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
                if most_recent_result == 1:  # Won last fight
                    win_streak += 1
                    loss_streak = 0
                elif most_recent_result == 0:  # Lost last fight
                    loss_streak += 1
                    win_streak = 0
            else:
                print(f"Most recent result was NaN for {fighter_name_lower}")
        else:
            print(f"Winner column not found for {fighter_name_lower}")

        print(f"Calculated stats for {fighter_name_lower}: age={current_age}, exp={years_of_experience}, "
              f"days={days_since_last_fight}, win_streak={win_streak}, loss_streak={loss_streak}")

        return current_age, years_of_experience, days_since_last_fight, win_streak, loss_streak

    def _calculate_fighter_stats(
            self,
            fighter_name: str,
            df: pd.DataFrame,
            current_date: datetime,
            n_past_fights: int
    ) -> tuple:
        """
        Calculate fighter statistics (age, experience, days since last fight, win and loss streaks).
        This is the original implementation which has issues with case sensitivity.
        """
        fighter_all_fights = df[(df['fighter'] == fighter_name) & (df['fight_date'] < current_date)] \
            .sort_values(by='fight_date', ascending=False)
        fighter_recent_fights = fighter_all_fights.head(n_past_fights)
        if fighter_recent_fights.empty:
            return np.nan, np.nan, np.nan, np.nan, np.nan

        last_fight_date = fighter_recent_fights['fight_date'].iloc[0]
        first_fight_date = fighter_all_fights['fight_date'].iloc[-1] if len(fighter_all_fights) > 0 else last_fight_date
        days_since_last_fight = (current_date - last_fight_date).days

        last_known_age = fighter_recent_fights['age'].iloc[0] if 'age' in fighter_recent_fights.columns else 30
        current_age = np.ceil(last_known_age + days_since_last_fight / 365.25)
        years_of_experience = (current_date - first_fight_date).days / 365.25

        win_streak = fighter_recent_fights['win_streak'].iloc[0] if 'win_streak' in fighter_recent_fights.columns else 0
        loss_streak = fighter_recent_fights['loss_streak'].iloc[
            0] if 'loss_streak' in fighter_recent_fights.columns else 0

        if 'winner' in fighter_recent_fights.columns:
            most_recent_result = fighter_recent_fights['winner'].iloc[0]
            if most_recent_result == 1:
                win_streak += 1
                loss_streak = 0
            elif most_recent_result == 0:
                loss_streak += 1
                win_streak = 0
        return current_age, years_of_experience, days_since_last_fight, win_streak, loss_streak

    def _process_fight_odds(self, odds_a: float, odds_b: float) -> tuple:
        """Process betting odds for a fight (returns [decimal_odds_a, decimal_odds_b], difference, ratio)."""
        return self.odds_utils.process_odds_pair(odds_a, odds_b)

    def _generate_column_names(
            self,
            features_to_include: List[str],
            method_columns: List[str],
            n_past_fights: int,
            tester: int,
            include_names: bool
    ) -> List[str]:
        """Generate column names for the matchup DataFrame."""
        results_columns = []
        for i in range(1, tester + 1):
            results_columns += [
                f"result_fight_{i}", f"winner_fight_{i}", f"weight_class_fight_{i}", f"scheduled_rounds_fight_{i}",
                f"result_b_fight_{i}", f"winner_b_fight_{i}", f"weight_class_b_fight_{i}",
                f"scheduled_rounds_b_fight_{i}"
            ]
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
            'current_fight_closing_open_diff_a', 'current_fight_closing_open_diff_b',  # Added new columns
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
                    self.utils.safe_divide(df[col_a], df[col_b])
        return pd.concat([df, pd.DataFrame(diff_columns), pd.DataFrame(ratio_columns)], axis=1)


class UFCMatchupCreator:
    """Creates matchup data for UFC fights."""

    def __init__(self, data_dir: str = "data"):
        """Initialize the matchup creator with a data directory."""
        # Use absolute path resolution
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
        """
        Create a matchup file for a UFC fight.

        Args:
            fighter_a: Name of first fighter
            fighter_b: Name of second fighter
            open_odds_a: Opening odds for fighter A
            open_odds_b: Opening odds for fighter B
            closing_odds_a: Closing odds for fighter A
            closing_odds_b: Closing odds for fighter B
            fight_date: Fight date in YYYY-MM-DD format

        Returns:
            DataFrame with the matchup data
        """
        return self.matchup_predictor.create_fighter_matchup(
            fighter_a, fighter_b, open_odds_a, open_odds_b, closing_odds_a, closing_odds_b, fight_date
        )


def main():
    """Create a fighter matchup file."""
    # Fighters
    fighter_a = "Alex Pereira"
    fighter_b = "magomed ankalaev"

    # Closing Odds
    closing_odds_a = -107
    closing_odds_b = -107

    # Open Odds
    open_odds_a = -200
    open_odds_b = 150

    # Specify fight date
    fight_date = "2024-12-14"

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Data directory should be in the same directory level as the script
    data_dir = os.path.join(current_dir, "data")

    print(f"Current directory: {current_dir}")
    print(f"Using data directory: {data_dir}")

    # Verify data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        # Try to find data directory in the project
        possible_dirs = [
            os.path.join(current_dir, "data"),
            os.path.join(os.path.dirname(current_dir), "data"),
            "data",  # Try relative to working directory
            "C:/Users/William/PycharmProjects/UFC/data"  # Hardcoded path based on your error message
        ]

        for path in possible_dirs:
            if os.path.exists(path):
                data_dir = path
                print(f"Found data directory at: {data_dir}")
                break
        else:
            print("Could not find data directory. Please check the path.")
            return

    # Create matchup data
    matchup_creator = UFCMatchupCreator(data_dir)
    try:
        matchup_creator.create_matchup(
            fighter_a,
            fighter_b,
            open_odds_a,
            open_odds_b,
            closing_odds_a,
            closing_odds_b,
            fight_date
        )
    except Exception as e:
        print(f"Error creating matchup: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time}")
"""
UFC Fight Analysis Module

This module contains classes and functions for processing and analyzing UFC fight data.
It handles data loading, preprocessing, feature engineering, and dataset preparation
for machine learning.
"""

import warnings
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import os
from datetime import datetime
# Calculate Elo ratings (imported from Elo module)
from data_manipulation.Elo import calculate_elo_ratings
from helper import DataUtils, OddsUtils, FighterUtils, DateUtils

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


class FightDataProcessor:
    """Process and transform UFC fight data for analysis."""

    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the processor with data directory.

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        self.utils = DataUtils()
        self.odds_utils = OddsUtils()
        self.fighter_utils = FighterUtils()

    def _load_csv(self, filepath: str) -> pd.DataFrame:
        """Load a CSV file into a DataFrame."""
        full_path = os.path.join(self.data_dir, filepath) if not os.path.isabs(filepath) else filepath
        return pd.read_csv(full_path)

    def _save_csv(self, df: pd.DataFrame, filepath: str) -> None:
        """Save DataFrame to CSV file."""
        full_path = os.path.join(self.data_dir, filepath) if not os.path.isabs(filepath) else filepath
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        df.to_csv(full_path, index=False)
        print(f"Saved to {full_path}")

    def combine_rounds_stats(self, file_path: str) -> pd.DataFrame:
        """
        Process round-level fight data into fighter career statistics.

        Args:
            file_path: Path to the UFC stats CSV file

        Returns:
            DataFrame with processed fighter statistics
        """
        print("Loading and preprocessing data...")
        ufc_stats = self._load_csv(file_path)
        fighter_stats = self._load_csv('../data_manipulation/Fight Scraping/rough_data/ufc_fighter_tott.csv')
        ufc_stats = self.utils.preprocess_data(ufc_stats, fighter_stats)

        # Get numeric columns for aggregation
        numeric_columns = self._get_numeric_columns(ufc_stats)

        print("Aggregating stats...")
        # Get maximum round information
        max_round_data = ufc_stats.groupby('id').agg({
            'last_round': 'max',
            'time': 'max'
        }).reset_index()

        # Aggregate numeric stats by fighter and fight
        aggregated_stats = ufc_stats.groupby(['id', 'fighter'])[numeric_columns].sum().reset_index()

        # Calculate basic rates
        aggregated_stats = self._calculate_basic_rates(aggregated_stats)

        # Get non-numeric data
        non_numeric_data = self._extract_non_numeric_data(ufc_stats)

        print("Merging aggregated stats with non-numeric data...")
        # Merge all components
        merged_stats = pd.merge(aggregated_stats, non_numeric_data, on=['id', 'fighter'], how='left')
        merged_stats = pd.merge(merged_stats, max_round_data, on='id', how='left')

        print("Calculating career stats...")
        # Calculate career-level statistics
        final_stats = merged_stats.groupby('fighter', group_keys=False).apply(
            lambda x: self.fighter_utils.aggregate_fighter_stats(x, numeric_columns)
        )

        # Calculate per-minute stats
        final_stats = self._calculate_per_minute_stats(final_stats)

        # Calculate additional rates
        final_stats = self._calculate_additional_rates(final_stats)

        # Filter and process data
        final_stats = self._filter_unwanted_results(final_stats)
        final_stats = self._factorize_categorical_columns(final_stats)

        # Process odds data
        final_stats = self.odds_utils.process_odds_data(final_stats)

        # Clean up columns
        columns_to_drop = ['new_Open', 'new_Closing Range Start', 'new_Closing Range End', 'new_Movement', 'dob']
        final_stats = final_stats.drop(columns=columns_to_drop, errors='ignore')

        # Remove duplicate columns
        duplicate_columns = final_stats.columns[final_stats.columns.duplicated()]
        final_stats = final_stats.loc[:, ~final_stats.columns.duplicated()]
        if len(duplicate_columns) > 0:
            print(f"Dropped duplicate columns: {list(duplicate_columns)}")

        print("Calculating additional stats...")
        # Sort by fighter and date
        final_stats = final_stats.sort_values(['fighter', 'fight_date'])

        # Calculate experience, streaks, and time-based stats
        final_stats = final_stats.groupby('fighter', group_keys=False).apply(
            self.fighter_utils.calculate_experience_and_days
        )
        final_stats = final_stats.groupby('fighter', group_keys=False).apply(
            self.fighter_utils.update_streaks
        )
        final_stats['days_since_last_fight'] = final_stats['days_since_last_fight'].fillna(0)

        print("Calculating takedowns and knockdowns per 15 minutes...")
        final_stats = self.fighter_utils.calculate_time_based_stats(final_stats)

        print("Calculating total fights, wins, and losses...")
        final_stats = final_stats.groupby('fighter', group_keys=False).apply(
            self.fighter_utils.calculate_total_fight_stats
        )

        print("Saving processed data...")
        self._save_csv(final_stats, 'combined_rounds.csv')

        return final_stats

    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """Extract relevant numeric columns for aggregation."""
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col not in ['id', 'last_round', 'age']]
        if 'time' not in numeric_columns:
            numeric_columns.append('time')
        return numeric_columns

    def _calculate_basic_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic strike and takedown rates."""
        df['significant_strikes_rate'] = self.utils.safe_divide(
            df['significant_strikes_landed'],
            df['significant_strikes_attempted']
        )
        df['takedown_rate'] = self.utils.safe_divide(
            df['takedown_successful'],
            df['takedown_attempted']
        )
        return df

    def _extract_non_numeric_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract non-numeric columns from the DataFrame."""
        non_numeric_columns = df.select_dtypes(exclude=['int64', 'float64']).columns.difference(
            ['id', 'fighter']
        )
        return df.drop_duplicates(subset=['id', 'fighter'])[
            ['id', 'fighter', 'age'] + list(non_numeric_columns)
            ]

    def _calculate_per_minute_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate per-minute statistics."""
        df['fight_duration_minutes'] = df['time'] / 60
        for col in ['significant_strikes_landed', 'significant_strikes_attempted',
                    'total_strikes_landed', 'total_strikes_attempted']:
            df[f'{col}_per_min'] = self.utils.safe_divide(df[col], df['fight_duration_minutes'])
        return df

    def _calculate_additional_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional rate statistics."""
        df["total_strikes_rate"] = self.utils.safe_divide(
            df["total_strikes_landed"],
            df["total_strikes_attempted"]
        )
        df["combined_success_rate"] = (df["takedown_rate"] + df["total_strikes_rate"]) / 2
        return df

    def _filter_unwanted_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out unwanted fight results."""
        df = df[~df['winner'].isin(['NC/NC', 'D/D'])]
        df = df[~df['result'].isin(['DQ', 'DQ ', 'Could Not Continue ', 'Overturned ', 'Other '])]
        return df

    def _factorize_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to numeric codes and print the mapping."""
        for column in ['result', 'winner', 'scheduled_rounds']:
            df[column], unique = pd.factorize(df[column])
            mapping = {index: label for index, label in enumerate(unique)}
            print(f"Mapping for {column}: {mapping}")
        return df

    def combine_fighters_stats(self, file_path: str) -> pd.DataFrame:
        """
        Create pairwise fighter statistics for all fights.

        Args:
            file_path: Path to the combined rounds CSV file

        Returns:
            DataFrame with paired fighter statistics
        """
        df = self._load_csv(file_path)

        # Remove event columns and sort
        df = df.drop(columns=[col for col in df.columns if 'event' in col.lower()])
        df = df.sort_values(by=['id', 'fighter'])

        # Create mirrored fight pairs
        fights_dict = {}
        for _, row in df.iterrows():
            fight_id = row['id']
            fights_dict.setdefault(fight_id, []).append(row)

        # Combine original and mirrored rows
        combined_fights = []
        skipped_fights = 0

        for fight_id, fighters in fights_dict.items():
            if len(fighters) == 2:
                fighter_1, fighter_2 = fighters
                # Original pairing (fighter 1 vs fighter 2)
                original = pd.concat([pd.Series(fighter_1), pd.Series(fighter_2).add_suffix('_b')])
                # Mirrored pairing (fighter 2 vs fighter 1)
                mirrored = pd.concat([pd.Series(fighter_2), pd.Series(fighter_1).add_suffix('_b')])
                combined_fights.extend([original, mirrored])
            else:
                skipped_fights += 1

        if skipped_fights > 0:
            print(f"Skipped {skipped_fights} fights with missing fighter data")

        # Create and process combined DataFrame
        final_combined_df = pd.DataFrame(combined_fights).reset_index(drop=True)

        # Define columns for processing
        final_combined_df = self._calculate_differential_and_ratio_features(final_combined_df)

        # Filter and sort
        final_combined_df = final_combined_df[~final_combined_df['winner'].isin(['NC', 'D'])]
        final_combined_df['fight_date'] = pd.to_datetime(final_combined_df['fight_date'])
        final_combined_df = final_combined_df.sort_values(
            by=['fighter', 'fight_date'],
            ascending=[True, True]
        )

        # Save the result
        self._save_csv(final_combined_df, 'combined_sorted_fighter_stats.csv')

        return final_combined_df

    def _calculate_differential_and_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate differential and ratio features between fighter pairs."""
        # Define columns to process
        base_columns = [
            'knockdowns', 'significant_strikes_landed', 'significant_strikes_attempted',
            'significant_strikes_rate', 'total_strikes_landed', 'total_strikes_attempted',
            'takedown_successful', 'takedown_attempted', 'takedown_rate', 'submission_attempt',
            'reversals', 'head_landed', 'head_attempted', 'body_landed', 'body_attempted',
            'leg_landed', 'leg_attempted', 'distance_landed', 'distance_attempted',
            'clinch_landed', 'clinch_attempted', 'ground_landed', 'ground_attempted'
        ]
        other_columns = [
            'open_odds', 'closing_range_start', 'closing_range_end', 'pre_fight_elo',
            'years_of_experience', 'win_streak', 'loss_streak', 'days_since_last_fight',
            'significant_strikes_landed_per_min', 'significant_strikes_attempted_per_min',
            'total_strikes_landed_per_min', 'total_strikes_attempted_per_min', 'takedowns_per_15min',
            'knockdowns_per_15min', 'total_fights', 'total_wins', 'total_losses',
            'wins_by_ko', 'losses_by_ko', 'wins_by_submission', 'losses_by_submission', 'wins_by_decision',
            'losses_by_decision', 'win_rate_by_ko', 'loss_rate_by_ko', 'win_rate_by_submission',
            'loss_rate_by_submission', 'win_rate_by_decision', 'loss_rate_by_decision'
        ]
        columns_to_process = (
                base_columns +
                [f"{col}_career" for col in base_columns] +
                [f"{col}_career_avg" for col in base_columns] +
                other_columns
        )

        # Calculate differential features
        diff_features = {}
        for col in columns_to_process:
            if col in df.columns and f"{col}_b" in df.columns:
                diff_features[f"{col}_diff"] = df[col] - df[f"{col}_b"]

        # Calculate ratio features
        ratio_features = {}
        for col in columns_to_process:
            if col in df.columns and f"{col}_b" in df.columns:
                ratio_features[f"{col}_ratio"] = self.utils.safe_divide(df[col], df[f"{col}_b"])

        # Combine all features
        return pd.concat([df, pd.DataFrame(diff_features), pd.DataFrame(ratio_features)], axis=1)


class MatchupProcessor:
    """Process and prepare matchup data for predictive modeling."""

    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the processor with data directory.

        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = data_dir
        self.fight_processor = FightDataProcessor(data_dir)
        self.utils = DataUtils()
        self.odds_utils = OddsUtils()

    def create_matchup_data(self, file_path: str, tester: int, include_names: bool = False) -> pd.DataFrame:
        """
        Create matchup data for predictive modeling.

        Args:
            file_path: Path to the fighter stats CSV
            tester: Determines the number of most recent fights to use
            include_names: Whether to include fighter names in output

        Returns:
            DataFrame with matchup features
        """
        print(f"Creating matchup data with {tester} recent fights...")
        df = self.fight_processor._load_csv(file_path)
        n_past_fights = 6 - tester

        # Define columns to exclude from features
        columns_to_exclude = [
            'fighter', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
            'result', 'winner', 'weight_class', 'scheduled_rounds',
            'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b'
        ]

        # Define features to include
        features_to_include = [
            col for col in df.columns if col not in columns_to_exclude and
                                         col != 'age' and not col.endswith('_age')
        ]

        # Method columns (target variables)
        method_columns = ['winner']

        # Process matchups
        matchup_data = self._process_matchups(
            df, features_to_include, method_columns, n_past_fights, tester, include_names
        )

        # Create DataFrame
        column_names = self._generate_column_names(
            features_to_include, method_columns, n_past_fights, tester, include_names
        )
        matchup_df = pd.DataFrame(matchup_data, columns=column_names)

        # Drop fight_date column if present
        matchup_df = matchup_df.drop(columns=['fight_date'], errors='ignore')

        # Standardize column names
        matchup_df.columns = [self.utils.rename_columns_general(col) for col in matchup_df.columns]

        # Calculate additional differential and ratio columns
        matchup_df = self._calculate_matchup_features(matchup_df, features_to_include, n_past_fights)

        # Save output
        output_filename = f'matchup data/matchup_data_{n_past_fights}_avg{"_name" if include_names else ""}.csv'
        self.fight_processor._save_csv(matchup_df, output_filename)

        return matchup_df

    def _process_matchups(
            self,
            df: pd.DataFrame,
            features_to_include: List[str],
            method_columns: List[str],
            n_past_fights: int,
            tester: int,
            include_names: bool
    ) -> List[List]:
        """Process each fight to create matchup feature vectors with support for fighters with fewer fights."""
        matchup_data = []
        skipped_count = 0
        processed_count = 0
        partial_data_count = 0

        # Process each current fight
        for _, current_fight in df.iterrows():
            fighter_a_name = current_fight['fighter']
            fighter_b_name = current_fight['fighter_b']

            # Get past fights for each fighter
            fighter_a_df = df[
                (df['fighter'] == fighter_a_name) &
                (df['fight_date'] < current_fight['fight_date'])
                ].sort_values(by='fight_date', ascending=False).head(n_past_fights)

            fighter_b_df = df[
                (df['fighter'] == fighter_b_name) &
                (df['fight_date'] < current_fight['fight_date'])
                ].sort_values(by='fight_date', ascending=False).head(n_past_fights)

            # Skip if either fighter has no past fights
            if len(fighter_a_df) == 0 or len(fighter_b_df) == 0:
                skipped_count += 1
                continue

            # Flag if we have partial data (at least one fighter with fewer than n_past_fights)
            has_partial_data = len(fighter_a_df) < n_past_fights or len(fighter_b_df) < n_past_fights
            if has_partial_data:
                partial_data_count += 1

            # Extract features from available past fights
            fighter_a_features = fighter_a_df[features_to_include].mean().values
            fighter_b_features = fighter_b_df[features_to_include].mean().values

            # Extract recent fight results
            # Only extract the available fight results, up to tester number
            num_a_results = min(len(fighter_a_df), tester)
            num_b_results = min(len(fighter_b_df), tester)

            results_fighter_a = fighter_a_df[['result', 'winner', 'weight_class', 'scheduled_rounds']].head(
                num_a_results).values.flatten() if num_a_results > 0 else np.array([])

            results_fighter_b = fighter_b_df[['result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']].head(
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

            # Get target labels
            labels = current_fight[method_columns].values

            # Process odds and age data
            current_fight_odds, current_fight_odds_diff, current_fight_odds_ratio = self._process_fight_odds(
                current_fight['open_odds'], current_fight['open_odds_b']
            )

            current_fight_closing_odds, current_fight_closing_odds_diff, current_fight_closing_odds_ratio = self._process_fight_odds(
                current_fight['closing_range_end'], current_fight['closing_range_end_b']
            )

            current_fight_ages = [current_fight['age'], current_fight['age_b']]
            current_fight_age_diff = current_fight['age'] - current_fight['age_b']
            current_fight_age_ratio = self.utils.safe_divide(current_fight['age'], current_fight['age_b'])

            # Process Elo and other stats
            elo_stats, elo_ratio = self._process_elo_stats(current_fight)
            other_stats = self._process_other_stats(current_fight)

            # Combine all features
            combined_features = np.concatenate([
                fighter_a_features, fighter_b_features, results_fighter_a, results_fighter_b,
                current_fight_odds, [current_fight_odds_diff, current_fight_odds_ratio],
                current_fight_closing_odds, [current_fight_closing_odds_diff, current_fight_closing_odds_ratio],
                current_fight_ages, [current_fight_age_diff, current_fight_age_ratio],
                elo_stats, [elo_ratio], other_stats
            ])
            combined_row = np.concatenate([combined_features, labels])

            # Get most recent date and current fight date
            most_recent_date_a = fighter_a_df['fight_date'].max() if len(fighter_a_df) > 0 else None
            most_recent_date_b = fighter_b_df['fight_date'].max() if len(fighter_b_df) > 0 else None
            most_recent_date = max(most_recent_date_a,
                                   most_recent_date_b) if most_recent_date_a and most_recent_date_b else most_recent_date_a or most_recent_date_b
            current_fight_date = current_fight['fight_date']

            # Add to matchup data
            if not include_names:
                matchup_data.append([most_recent_date] + combined_row.tolist() + [current_fight_date])
            else:
                matchup_data.append(
                    [fighter_a_name, fighter_b_name, most_recent_date] + combined_row.tolist() + [current_fight_date]
                )

            processed_count += 1

        print(f"Processed {processed_count} matchups (including {partial_data_count} with partial fight history)")
        print(f"Skipped {skipped_count} matchups where at least one fighter had no previous fights")
        return matchup_data

    def _process_fight_odds(self, odds_a: float, odds_b: float) -> Tuple[List[float], float, float]:
        """Process betting odds for a fight."""
        return self.odds_utils.process_odds_pair(odds_a, odds_b)

    def _process_elo_stats(self, current_fight: pd.Series) -> Tuple[List[float], float]:
        """Process Elo rating statistics."""
        elo_a = current_fight['pre_fight_elo']
        elo_b = current_fight['pre_fight_elo_b']
        elo_diff = current_fight['pre_fight_elo_diff']

        # Calculate win probabilities based on Elo
        a_win_prob = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        b_win_prob = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))

        elo_stats = [elo_a, elo_b, elo_diff, a_win_prob, b_win_prob]
        elo_ratio = self.utils.safe_divide(elo_a, elo_b)

        return elo_stats, elo_ratio

    def _process_other_stats(self, current_fight: pd.Series) -> List[float]:
        """Process other fighter statistics."""
        # Win/loss streak stats
        win_streak_a = current_fight['win_streak']
        win_streak_b = current_fight['win_streak_b']
        win_streak_diff = win_streak_a - win_streak_b
        win_streak_ratio = self.utils.safe_divide(win_streak_a, win_streak_b)

        loss_streak_a = current_fight['loss_streak']
        loss_streak_b = current_fight['loss_streak_b']
        loss_streak_diff = loss_streak_a - loss_streak_b
        loss_streak_ratio = self.utils.safe_divide(loss_streak_a, loss_streak_b)

        # Experience stats
        exp_a = current_fight['years_of_experience']
        exp_b = current_fight['years_of_experience_b']
        exp_diff = exp_a - exp_b
        exp_ratio = self.utils.safe_divide(exp_a, exp_b)

        # Last fight stats
        days_since_a = current_fight['days_since_last_fight']
        days_since_b = current_fight['days_since_last_fight_b']
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
            tester: int,
            include_names: bool
    ) -> List[str]:
        """Generate column names for the matchup DataFrame."""
        # Results columns
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

    def split_train_val_test(
            self,
            matchup_data_file: str,
            start_date: str,
            end_date: str,
            years_back: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split matchup data into training, validation, and test sets.

        Args:
            matchup_data_file: Path to matchup data CSV
            start_date: Start date for test set (YYYY-MM-DD)
            end_date: End date for test set (YYYY-MM-DD)
            years_back: Number of years of data to include

        Returns:
            Tuple of (train_data, val_data, test_data) DataFrames
        """
        print(f"Splitting data from {start_date} to {end_date} with {years_back} years history...")
        matchup_df = self.fight_processor._load_csv(matchup_data_file)

        # Remove highly correlated features
        matchup_df, removed_features = self.utils.remove_correlated_features(
            matchup_df,
            correlation_threshold=0.95,
            protected_columns=['current_fight_open_odds_diff', 'current_fight_closing_range_end_b',
                               'current_fight_closing_odds_diff']
        )

        # Convert dates
        matchup_df['current_fight_date'] = pd.to_datetime(matchup_df['current_fight_date'])
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        years_before = start_date - pd.DateOffset(years=years_back)

        # Split data
        test_data = matchup_df[
            (matchup_df['current_fight_date'] >= start_date) &
            (matchup_df['current_fight_date'] <= end_date)
            ].copy()

        remaining_data = matchup_df[
            (matchup_df['current_fight_date'] >= years_before) &
            (matchup_df['current_fight_date'] < start_date)
            ].copy()

        # Sort remaining data and split into train/val
        remaining_data = remaining_data.sort_values(by='current_fight_date', ascending=True)
        split_index = int(len(remaining_data) * 0.8)
        train_data = remaining_data.iloc[:split_index].copy()
        val_data = remaining_data.iloc[split_index:].copy()

        # Remove duplicate fights
        val_data = self._remove_duplicate_fights(val_data)
        test_data = self._remove_duplicate_fights(test_data)

        # Sort datasets
        sort_cols = ['current_fight_date', 'fighter_a', 'fighter_b']
        train_data = train_data.sort_values(by=sort_cols, ascending=True)
        val_data = val_data.sort_values(by=sort_cols, ascending=True)
        test_data = test_data.sort_values(by=sort_cols, ascending=True)

        # Save datasets
        self.fight_processor._save_csv(train_data, 'train test data/train_data.csv')
        self.fight_processor._save_csv(val_data, 'train test data/val_data.csv')
        self.fight_processor._save_csv(test_data, 'train test data/test_data.csv')

        # Save removed features
        with open(os.path.join(self.fight_processor.data_dir, 'train test data/removed_features.txt'), 'w') as file:
            file.write(','.join(removed_features))

        print(
            f"Train set size: {len(train_data)}, Validation set size: {len(val_data)}, Test set size: {len(test_data)}")
        print(f"Removed {len(removed_features)} correlated features")

        return train_data, val_data, test_data

    def _remove_duplicate_fights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate fights, keeping only one row per fight pair."""
        df = df.copy()
        df['fight_pair'] = df.apply(
            lambda row: tuple(sorted([row['fighter_a'], row['fighter_b']])), axis=1
        )
        df = df.drop_duplicates(subset=['fight_pair'], keep='first')
        return df.drop(columns=['fight_pair']).reset_index(drop=True)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    # Initialize processors
    fight_processor = FightDataProcessor()
    matchup_processor = MatchupProcessor()

    # Uncomment the functions you want to run
    fight_processor.combine_rounds_stats('../data/ufc_fight_processed.csv')
    calculate_elo_ratings('../data/combined_rounds.csv')
    fight_processor.combine_fighters_stats("../data/combined_rounds.csv")
    matchup_processor.create_matchup_data("../data/combined_sorted_fighter_stats.csv", 3, True)
    matchup_processor.split_train_val_test(
        '../data/matchup data/matchup_data_3_avg_name.csv',
        '2024-01-01',
        '2025-12-31',
        10
    )


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"Total runtime: {end_time - start_time}")

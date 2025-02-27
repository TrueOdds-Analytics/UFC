"""
UFC Data Analysis Utilities

This module contains utility classes and functions for processing UFC fight data.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union, Any


class DataUtils:
    """General data processing utilities."""

    @staticmethod
    def safe_divide(
        numerator: Union[float, np.ndarray, pd.Series],
        denominator: Union[float, np.ndarray, pd.Series],
        default: float = 0
    ) -> Union[float, np.ndarray, pd.Series]:
        """
        Safely divide with protection against division by zero.

        Args:
            numerator: Value or array to divide
            denominator: Value or array to divide by
            default: Default value when denominator is zero

        Returns:
            Result of division with zeros replaced by default
        """
        if isinstance(numerator, (pd.Series, pd.DataFrame)) or isinstance(denominator, (pd.Series, pd.DataFrame)):
            # For pandas objects, use the div method and handle NaN/inf values
            result = pd.Series(numerator).div(pd.Series(denominator))
            return result.fillna(default).replace([np.inf, -np.inf], default)
        elif isinstance(numerator, np.ndarray) and isinstance(denominator, np.ndarray):
            # For numpy arrays, use masking
            result = np.zeros_like(numerator, dtype=float)
            mask = denominator != 0
            result[mask] = numerator[mask] / denominator[mask]
            result[~mask] = default
            return result
        else:
            # For scalar values
            return numerator / denominator if denominator != 0 else default

    def preprocess_data(self, ufc_stats: pd.DataFrame, fighter_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the UFC and fighter stats dataframes.

        Args:
            ufc_stats: DataFrame with UFC fight statistics
            fighter_stats: DataFrame with fighter biographical data

        Returns:
            Preprocessed DataFrame
        """
        # Standardize fighter names and dates
        ufc_stats['fighter'] = ufc_stats['fighter'].astype(str).str.lower()
        ufc_stats['fight_date'] = pd.to_datetime(ufc_stats['fight_date'])
        fighter_stats['name'] = fighter_stats['FIGHTER'].astype(str).str.lower().str.strip()
        fighter_stats['dob'] = fighter_stats['DOB'].replace(['--', '', 'NA', 'N/A'], np.nan).apply(DateUtils.parse_date)

        # Merge fighter stats and calculate age
        ufc_stats = pd.merge(
            ufc_stats,
            fighter_stats[['name', 'dob']],
            left_on='fighter', right_on='name',
            how='left'
        )
        ufc_stats['age'] = (ufc_stats['fight_date'] - ufc_stats['dob']).dt.days / 365.25
        ufc_stats['age'] = ufc_stats['age'].fillna(np.nan).round().astype(float)
        ufc_stats.loc[ufc_stats['age'] < 0, 'age'] = np.nan

        # Clean data and drop unwanted columns
        ufc_stats = ufc_stats.drop(['round', 'location', 'name'], axis=1)
        ufc_stats = ufc_stats[~ufc_stats['weight_class'].str.contains("Women's")]

        # Convert time strings ('MM:SS') to seconds
        ufc_stats['time'] = (
                pd.to_datetime(ufc_stats['time'], format='%M:%S').dt.minute * 60 +
                pd.to_datetime(ufc_stats['time'], format='%M:%S').dt.second
        )

        return ufc_stats

    def rename_columns_general(self, col: str) -> str:
        """
        Rename columns for clarity.

        Args:
            col: Column name to rename

        Returns:
            Renamed column
        """
        if 'fighter' in col and not col.startswith('fighter'):
            if 'b_fighter_b' in col:
                return col.replace('b_fighter_b', 'fighter_b_opponent')
            elif 'b_fighter' in col:
                return col.replace('b_fighter', 'fighter_a_opponent')
            elif 'fighter' in col and 'fighter_b' not in col:
                return col.replace('fighter', 'fighter_a')
        return col

    def get_opponent(self, fighter: str, fight_id: str, ufc_stats: pd.DataFrame) -> Optional[str]:
        """
        Get a fighter's opponent for a specific fight.

        Args:
            fighter: Fighter name
            fight_id: Fight ID
            ufc_stats: UFC stats DataFrame

        Returns:
            Opponent name or None if not found
        """
        fight_fighters = ufc_stats[ufc_stats['id'] == fight_id]['fighter'].unique()
        if len(fight_fighters) < 2:
            return None
        return fight_fighters[0] if fight_fighters[0] != fighter else fight_fighters[1]

    def remove_correlated_features(
        self,
        df: pd.DataFrame,
        correlation_threshold: float = 0.95,
        protected_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features from DataFrame.

        Args:
            df: Input DataFrame
            correlation_threshold: Correlation threshold for removal
            protected_columns: Columns to never remove

        Returns:
            Tuple of (DataFrame without correlated features, list of removed columns)
        """
        protected_columns = protected_columns or []

        # Select numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_df = df[numeric_columns]

        # Calculate correlation matrix
        corr_matrix = numeric_df.corr().abs()

        # Get upper triangle of correlation matrix
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find columns to drop (excluding protected columns)
        columns_to_drop = [
            column for column in upper_tri.columns
            if any(upper_tri[column] > correlation_threshold)
            and column not in protected_columns
        ]

        # Drop columns
        cleaned_df = df.drop(columns=columns_to_drop)

        return cleaned_df, columns_to_drop


class OddsUtils:
    """Utilities for processing betting odds."""

    @staticmethod
    def round_to_nearest_1(x: float) -> int:
        """Round to nearest integer."""
        return round(x)

    @staticmethod
    def calculate_complementary_odd(odd: float) -> float:
        """
        Calculate complementary betting odd.

        Args:
            odd: Original betting odd

        Returns:
            Complementary betting odd
        """
        if odd > 0:
            prob = 100 / (odd + 100)
        else:
            prob = abs(odd) / (abs(odd) + 100)

        complementary_prob = 1.045 - prob  # Make probabilities sum to 104.5%

        if complementary_prob >= 0.5:
            complementary_odd = -100 * complementary_prob / (1 - complementary_prob)
        else:
            complementary_odd = 100 * (1 - complementary_prob) / complementary_prob

        return OddsUtils.round_to_nearest_1(complementary_odd)

    def process_odds_pair(
        self,
        odds_a: Optional[float],
        odds_b: Optional[float]
    ) -> Tuple[List[float], float, float]:
        """
        Process a pair of betting odds.

        Args:
            odds_a: First fighter's odds (or None/NaN)
            odds_b: Second fighter's odds (or None/NaN)

        Returns:
            Tuple of (odds list, odds difference, odds ratio)
        """
        utils = DataUtils()

        if pd.notna(odds_a) and pd.notna(odds_b):
            odds_list = [odds_a, odds_b]
            odds_diff = odds_a - odds_b
            odds_ratio = utils.safe_divide(odds_a, odds_b)
        elif pd.notna(odds_a):
            odds_a_rounded = self.round_to_nearest_1(odds_a)
            odds_b_calc = self.calculate_complementary_odd(odds_a_rounded)
            odds_list = [odds_a_rounded, odds_b_calc]
            odds_diff = odds_a_rounded - odds_b_calc
            odds_ratio = utils.safe_divide(odds_a_rounded, odds_b_calc)
        elif pd.notna(odds_b):
            odds_b_rounded = self.round_to_nearest_1(odds_b)
            odds_a_calc = self.calculate_complementary_odd(odds_b_rounded)
            odds_list = [odds_a_calc, odds_b_rounded]
            odds_diff = odds_a_calc - odds_b_rounded
            odds_ratio = utils.safe_divide(odds_a_calc, odds_b_rounded)
        else:
            odds_list = [-111, -111]
            odds_diff = 0
            odds_ratio = 1

        return odds_list, odds_diff, odds_ratio

    def process_odds_data(self, final_stats: pd.DataFrame) -> pd.DataFrame:
        """
        Process and merge betting odds data with fight statistics.

        Args:
            final_stats: DataFrame containing fight statistics

        Returns:
            DataFrame with merged odds data
        """
        # Copy and remove duplicate columns
        final_stats = final_stats.copy().loc[:, ~final_stats.columns.duplicated()]

        # Read odds data
        odds_df = pd.read_csv('../data/odds data/cleaned_fight_odds.csv')

        # Standardize fighter names
        final_stats['fighter'] = final_stats['fighter'].str.lower().str.strip()
        odds_df['Matchup'] = odds_df['Matchup'].str.lower().str.strip()

        # Rename odds_df column so both DataFrames share the same key
        odds_df.rename(columns={'Matchup': 'fighter'}, inplace=True)

        # Convert dates to datetime
        final_stats['fight_date'] = pd.to_datetime(final_stats['fight_date'])
        odds_df['Date'] = pd.to_datetime(odds_df['Date'], format='%Y-%m-%d')

        # Sort both DataFrames by their respective date columns (required for merge_asof)
        final_stats.sort_values('fight_date', inplace=True)
        odds_df.sort_values('Date', inplace=True)

        # Merge using merge_asof with a grouping key and tolerance of 1 day
        merged_df = pd.merge_asof(
            final_stats,
            odds_df,
            left_on='fight_date',
            right_on='Date',
            by='fighter',
            tolerance=pd.Timedelta("1D"),
            direction='nearest'
        )

        # Drop the extra Date column from the odds DataFrame
        merged_df.drop(columns=['Date'], inplace=True)

        # Rename odds columns for clarity
        merged_df.rename(
            columns={
                'Open': 'open_odds',
                'Closing Range Start': 'closing_range_start',
                'Closing Range End': 'closing_range_end',
                'Movement': 'odds_movement'
            },
            inplace=True
        )

        return merged_df


class FighterUtils:
    """Utilities for processing fighter statistics."""

    def __init__(self):
        """Initialize with DataUtils instance."""
        self.utils = DataUtils()

    def aggregate_fighter_stats(self, group: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        """
        Calculate cumulative career statistics for a fighter.

        Args:
            group: DataFrame group for a single fighter
            numeric_columns: Numeric columns to aggregate

        Returns:
            DataFrame with added career statistics
        """
        group = group.sort_values('fight_date')
        cumulative_stats = group[numeric_columns].cumsum(skipna=True)
        fight_count = group.groupby('fighter').cumcount() + 1

        # Calculate career and career average stats
        for col in numeric_columns:
            group[f"{col}_career"] = cumulative_stats[col]
            group[f"{col}_career_avg"] = self.utils.safe_divide(cumulative_stats[col], fight_count)

        # Calculate career rate stats
        group['significant_strikes_rate_career'] = self.utils.safe_divide(
            cumulative_stats['significant_strikes_landed'],
            cumulative_stats['significant_strikes_attempted']
        )
        group['takedown_rate_career'] = self.utils.safe_divide(
            cumulative_stats['takedown_successful'],
            cumulative_stats['takedown_attempted']
        )
        group['total_strikes_rate_career'] = self.utils.safe_divide(
            cumulative_stats['total_strikes_landed'],
            cumulative_stats['total_strikes_attempted']
        )
        group["combined_success_rate_career"] = (group["takedown_rate_career"] + group["total_strikes_rate_career"]) / 2

        return group

    def calculate_experience_and_days(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate fighter experience and days between fights.

        Args:
            group: DataFrame group for a single fighter

        Returns:
            DataFrame with added experience metrics
        """
        group = group.sort_values('fight_date')
        group['years_of_experience'] = (group['fight_date'] - group['fight_date'].iloc[0]).dt.days / 365.25
        group['days_since_last_fight'] = (group['fight_date'] - group['fight_date'].shift()).dt.days
        return group

    def update_streaks(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate win and loss streaks for a fighter.

        Args:
            group: DataFrame group for a single fighter

        Returns:
            DataFrame with updated streak columns
        """
        group = group.sort_values('fight_date')

        # Create a copy of the group to avoid SettingWithCopyWarning
        group_copy = group.copy()

        # Initialize streak columns
        group_copy['win_streak'] = 0
        group_copy['loss_streak'] = 0

        # For the first fight, no update needed

        # Calculate streaks for subsequent fights
        for i in range(1, len(group_copy)):
            if group_copy.iloc[i-1]['winner'] == 1:  # Win
                group_copy.iloc[i, group_copy.columns.get_loc('win_streak')] = group_copy.iloc[i-1]['win_streak'] + 1
                group_copy.iloc[i, group_copy.columns.get_loc('loss_streak')] = 0
            else:  # Loss
                group_copy.iloc[i, group_copy.columns.get_loc('win_streak')] = 0
                group_copy.iloc[i, group_copy.columns.get_loc('loss_streak')] = group_copy.iloc[i-1]['loss_streak'] + 1

        return group_copy

    def calculate_time_based_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate time-normalized statistics.

        Args:
            df: Fighter stats DataFrame

        Returns:
            DataFrame with added time-based statistics
        """
        # Convert career time (seconds) to minutes
        df['time_career_minutes'] = df['time_career'] / 60

        # Calculate takedowns and knockdowns per 15 minutes
        df['takedowns_per_15min'] = self.utils.safe_divide(
            df['takedown_successful_career'], df['time_career_minutes']
        ) * 15

        df['knockdowns_per_15min'] = self.utils.safe_divide(
            df['knockdowns_career'], df['time_career_minutes']
        ) * 15

        return df

    def calculate_total_fight_stats(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cumulative fight statistics.

        Args:
            group: DataFrame group for a single fighter

        Returns:
            DataFrame with added fight outcome statistics
        """
        # Sort by date and reset index
        group = group.sort_values('fight_date').reset_index(drop=True)

        # Calculate fight counts
        group['total_fights'] = range(1, len(group) + 1)
        group['total_wins'] = group['winner'].cumsum()
        group['total_losses'] = group['total_fights'] - group['total_wins']

        # Create outcome type masks
        ko_mask = group['result'].isin([0, 3])
        submission_mask = group['result'] == 1
        decision_mask = group['result'].isin([2, 4])

        win_mask = group['winner'] == 1
        loss_mask = ~win_mask

        # Calculate win types using vectorized operations
        group['wins_by_ko'] = (ko_mask & win_mask).cumsum()
        group['wins_by_submission'] = (submission_mask & win_mask).cumsum()
        group['wins_by_decision'] = (decision_mask & win_mask).cumsum()

        # Calculate loss types using vectorized operations
        group['losses_by_ko'] = (ko_mask & loss_mask).cumsum()
        group['losses_by_submission'] = (submission_mask & loss_mask).cumsum()
        group['losses_by_decision'] = (decision_mask & loss_mask).cumsum()

        # Calculate win/loss rates
        for outcome in ['ko', 'submission', 'decision']:
            group[f'win_rate_by_{outcome}'] = self.utils.safe_divide(
                group[f'wins_by_{outcome}'], group['total_wins']
            )
            group[f'loss_rate_by_{outcome}'] = self.utils.safe_divide(
                group[f'losses_by_{outcome}'], group['total_losses']
            )

        return group


class DateUtils:
    """Utilities for date processing."""

    @staticmethod
    def parse_date(date_str: Any) -> pd.Timestamp:
        """
        Parse date string in various formats.

        Args:
            date_str: Date string to parse

        Returns:
            Parsed timestamp or NaT if parsing fails
        """
        if pd.isna(date_str):
            return pd.NaT
        try:
            return pd.to_datetime(date_str, format='%d-%b-%y')
        except ValueError:
            try:
                return pd.to_datetime(date_str, format='%b %d, %Y')
            except ValueError:
                return pd.NaT
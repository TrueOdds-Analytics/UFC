"""
Utility functions for UFC fighter matchup processing
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional

# Import from the original module
try:
    from src.data_processing.cleaning.data_cleaner import DataUtils, OddsUtils

    # Use the imported utilities
    data_utils = DataUtils()
    odds_utils = OddsUtils()
    ORIGINAL_UTILS_AVAILABLE = True
except ImportError:
    ORIGINAL_UTILS_AVAILABLE = False
    # Define fallback implementations if original imports are not available
    print("WARNING: Original utility modules not found. Using fallback implementations.")


class FallbackDataUtils:
    """Fallback implementation of DataUtils if original is not available"""

    def safe_divide(self, a, b):
        """
        Safely divide a by b, returning 1 if b is 0 or if either value is NaN

        Args:
            a: Numerator
            b: Denominator

        Returns:
            Result of a/b or 1 if division not possible
        """
        if pd.isna(a) or pd.isna(b) or b == 0:
            return 1.0
        return a / b

    def rename_columns_general(self, col):
        """
        Basic column name standardization

        Args:
            col: Column name to rename

        Returns:
            Standardized column name
        """
        # Convert to lowercase and replace spaces with underscores
        new_col = str(col).lower().replace(' ', '_')
        return new_col


class FallbackOddsUtils:
    """Fallback implementation of OddsUtils if original is not available"""

    def process_odds_pair(self, odds_a: float, odds_b: float) -> Tuple[List[float], float, float]:
        """
        Process a pair of American odds to return decimal odds, difference, and ratio

        Args:
            odds_a: American odds for fighter A
            odds_b: American odds for fighter B

        Returns:
            Tuple of ([decimal_odds_a, decimal_odds_b], difference, ratio)
        """
        # Convert American odds to decimal
        decimal_a = (odds_a / 100) + 1 if odds_a > 0 else (100 / abs(odds_a)) + 1
        decimal_b = (odds_b / 100) + 1 if odds_b > 0 else (100 / abs(odds_b)) + 1

        # Calculate difference and ratio
        diff = decimal_a - decimal_b
        ratio = decimal_a / decimal_b if decimal_b != 0 else 1.0

        return [decimal_a, decimal_b], diff, ratio


# Create appropriate utility objects based on import success
if not ORIGINAL_UTILS_AVAILABLE:
    data_utils = FallbackDataUtils()
    odds_utils = FallbackOddsUtils()


def safe_divide(a, b):
    """
    Safely divide a by b, handling edge cases

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Result of a/b or 1 if division not possible
    """
    return data_utils.safe_divide(a, b)


def process_odds_pair(odds_a: float, odds_b: float) -> Tuple[List[float], float, float]:
    """
    Process a pair of American odds

    Args:
        odds_a: American odds for fighter A
        odds_b: American odds for fighter B

    Returns:
        Tuple of ([decimal_odds_a, decimal_odds_b], difference, ratio)
    """
    return odds_utils.process_odds_pair(odds_a, odds_b)


def rename_column(col):
    """
    Standardize column name

    Args:
        col: Column name to rename

    Returns:
        Standardized column name
    """
    return data_utils.rename_columns_general(col)


def resolve_fight_date(fight_date):
    """
    Process the fight date to ensure a valid datetime object

    Args:
        fight_date: Date string in YYYY-MM-DD format or datetime object

    Returns:
        datetime object representing the fight date
    """
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
    """
    Generate a filename for a matchup between two fighters

    Args:
        fighter_a: Name of first fighter
        fighter_b: Name of second fighter

    Returns:
        Formatted filename
    """
    return f"{fighter_a.replace(' ', '_')}_vs_{fighter_b.replace(' ', '_')}_matchup.csv"


def ensure_directory_exists(directory):
    """
    Ensure a directory exists, creating it if necessary

    Args:
        directory: Directory path to verify

    Returns:
        Absolute path to the directory
    """
    abs_path = os.path.abspath(directory)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def create_result_columns(n_past_fights, tester):
    """
    Generate column names for fight result columns

    Args:
        n_past_fights: Number of past fights to consider
        tester: Value used for calculating result columns (typically 6 - n_past_fights)

    Returns:
        List of result column names
    """
    results_columns = []
    for i in range(1, tester + 1):
        results_columns += [
            f"result_fight_{i}", f"winner_fight_{i}", f"weight_class_fight_{i}", f"scheduled_rounds_fight_{i}",
            f"result_b_fight_{i}", f"winner_b_fight_{i}", f"weight_class_b_fight_{i}",
            f"scheduled_rounds_b_fight_{i}"
        ]
    return results_columns
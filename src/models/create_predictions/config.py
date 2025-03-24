"""
Configuration settings for UFC fighter matchup processing
"""
import os

# Default number of past fights to consider for statistics
DEFAULT_PAST_FIGHTS = 3

# Default directory paths (will be resolved to absolute paths when used)
DEFAULT_DATA_DIR = "data"

# File paths relative to data directory
PATHS = {
    'processed_fighter_stats': "processed/combined_sorted_fighter_stats.csv",
    'test_data': "train_test/test_data.csv",
    'removed_features': "train_test/removed_features.txt",
    'live_data': "live_data",
    'matchup_data': "matchup data"
}

# Important columns that should always be preserved in the output
IMPORTANT_COLUMNS = [
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
    'current_fight_closing_open_diff_a', 'current_fight_closing_open_diff_b'
]

# Columns to exclude from features
EXCLUDED_COLUMNS = [
    'fighter', 'fighter_lower', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
    'result', 'winner', 'weight_class', 'scheduled_rounds',
    'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b'
]

# Debug mode - set to True for detailed logging
DEBUG = True


def get_absolute_path(base_dir, rel_path):
    """
    Get absolute path from relative path and base directory

    Args:
        base_dir: Base directory
        rel_path: Relative path

    Returns:
        Absolute path
    """
    return os.path.join(os.path.abspath(base_dir), rel_path)


def resolve_data_dir():
    """
    Attempt to find the data directory from various common locations

    Returns:
        Resolved data directory path or None if not found
    """
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Try several possible locations
    possible_dirs = [
        os.path.join(current_dir, "data"),
        os.path.join(os.path.dirname(current_dir), "data"),
        "data",  # Try relative to working directory
    ]

    # Add user home directory if available
    if 'HOME' in os.environ:
        possible_dirs.append(os.path.join(os.environ['HOME'], "UFC/data"))

    # Try Windows user directory if running on Windows
    if 'USERPROFILE' in os.environ:
        possible_dirs.append(os.path.join(os.environ['USERPROFILE'], "PycharmProjects/UFC/data"))

    # Check each possible location
    for path in possible_dirs:
        if os.path.exists(path):
            return os.path.abspath(path)

    # If no valid path found, return default
    return os.path.abspath(DEFAULT_DATA_DIR)
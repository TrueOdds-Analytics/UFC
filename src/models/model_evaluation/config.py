"""
Configuration settings for MMA betting model evaluation
"""
import os

# Default configuration values
DEFAULT_CONFIG = {
    # Threshold for bet confidence
    'manual_threshold': 0.5,

    # Calibration settings
    'use_calibration': True,
    'calibration_type': 'isotonic',  # Options: 'range_based', 'isotonic', or False

    # Betting strategy parameters
    'initial_bankroll': 10000,
    'kelly_fraction': 0.5,  # Fraction of Kelly criterion to use (0.5 = half Kelly)
    'fixed_bet_fraction': 0.1,  # Fraction of bankroll for fixed bets
    'max_bet_percentage': 0.1,  # Maximum percentage of bankroll to bet

    # Odds settings
    'min_odds': -300,  # Minimum odds to place a bet
    'max_underdog_odds': 200,  # Maximum underdog odds to place a bet

    # Model settings
    'use_ensemble': True,  # Whether to use ensemble of models

    # Odds type to use for evaluation
    'odds_type': 'close',  # Options: 'open', 'close', 'average'

    # Evaluation mode
    'evaluate_directory': True,  # Whether to evaluate all models in a directory
    'single_model_name': None,  # Name of a specific model to evaluate
}

# Model files for ensemble
MODEL_FILES = [
    'model_0.7009_auc_diff_0.0161.json',
    'model_0.7002_auc_diff_0.0410.json',
    'model_0.7025_auc_diff_0.0221.json',
    'model_0.7009_auc_diff_0.0293.json',
    'model_0.7009_auc_diff_0.0038.json'
]

# Range-based calibration settings
RANGE_CALIBRATION_RANGES = [0.25, 0.45, 0.65, 0.85]  # Creates 5 regions

# Data file paths
DATA_PATHS = {
    'val_data': '../../../data/train_test/val_data.csv',
    'test_data': '../../../data/train_test/test_data.csv',
    'live_data': '../../../data/live_data/all_matchups.csv',
    'encoder_path': '../../../saved_models/encoders/category_encoder.pkl',
    'model_base_path': '../../../saved_models/xgboost/jan2024-dec2025/dynamicmatchup sorted 425/'
}

# Display columns for output
DISPLAY_COLUMNS = ['current_fight_date', 'fighter_a', 'fighter_b']

# Output directory for plots
OUTPUT_DIR = '../outputs/calibration_plots'


# Ensure the existence of required directories
def ensure_directories():
    """Ensure that necessary directories exist."""
    os.makedirs(os.path.dirname(DATA_PATHS['encoder_path']), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

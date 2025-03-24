"""
Entry point for MMA betting model evaluation
"""
import os
import sys
# Add the parent directory to path if needed
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from main import main
from config import DEFAULT_CONFIG

if __name__ == "__main__":
    """
    Run MMA betting model evaluation with specified configuration

    This script serves as the main entry point and allows for different evaluation modes:
    1. Evaluate a single model by name
    2. Evaluate all models in a directory
    3. Run the default ensemble model evaluation

    Change the parameters below to customize the evaluation settings.
    """

    # Alternative: Use the default configuration from config.py
    main(**DEFAULT_CONFIG)

    # You can set the mode here:
    # 1. Evaluate all models in directory (EVALUATE_ALL_MODELS = True)
    # 2. Evaluate a single model by name (SINGLE_MODEL_NAME = "model_name")
    # 3. Run original functionality (both False)
    # EVALUATE_ALL_MODELS = True
    # SINGLE_MODEL_NAME = None  # Set to model name to evaluate just one model
    #
    # # Calibration options: 'isotonic', 'range_based', or False (for uncalibrated)
    # CALIBRATION_TYPE = 'isotonic'
    # USE_CALIBRATION = True if CALIBRATION_TYPE else False
    #
    # if SINGLE_MODEL_NAME:
    #     # Evaluate just one model
    #     main(
    #         use_calibration=USE_CALIBRATION,
    #         calibration_type=CALIBRATION_TYPE,
    #         initial_bankroll=10000,
    #         kelly_fraction=0.5,
    #         fixed_bet_fraction=0.1,
    #         max_bet_percentage=0.1,
    #         min_odds=-300,
    #         max_underdog_odds=200,
    #         odds_type='close',
    #         single_model_name=SINGLE_MODEL_NAME
    #     )
    # elif EVALUATE_ALL_MODELS:
    #     # Evaluate all models
    #     main(
    #         use_calibration=USE_CALIBRATION,
    #         calibration_type=CALIBRATION_TYPE,
    #         initial_bankroll=10000,
    #         kelly_fraction=0.5,
    #         fixed_bet_fraction=0.1,
    #         max_bet_percentage=0.1,
    #         min_odds=-300,
    #         max_underdog_odds=200,
    #         odds_type='close',
    #         evaluate_directory=True
    #     )
    # else:
    #     # Original functionality - evaluate an ensemble of models
    #     main(
    #         manual_threshold=0.5,
    #         use_calibration=USE_CALIBRATION,
    #         calibration_type=CALIBRATION_TYPE,
    #         initial_bankroll=10000,
    #         kelly_fraction=0.5,
    #         fixed_bet_fraction=0.1,
    #         max_bet_percentage=0.1,
    #         min_odds=-300,
    #         max_underdog_odds=200,
    #         use_ensemble=True,  # Set to True to use ensemble of models
    #         odds_type='close'  # Options: 'open', 'close', 'average'
    #     )

"""
Entry point for MMA betting analysis
"""
# Import directly from sibling modules (no package prefix needed)
from main import main
from config import DEFAULT_CONFIG

if __name__ == "__main__":
    # Run with the default configuration
    main(**DEFAULT_CONFIG)

    # Alternatively, you can customize parameters:
    """
    main(
        manual_threshold=0.55,
        use_calibration=True,
        calibration_type='range_based',
        initial_bankroll=5000,
        kelly_fraction=0.25,
        fixed_bet_fraction=0.05,
        max_bet_percentage=0.05,
        min_odds=-200,
        use_ensemble=True,
        odds_type='average'
    )
    """
"""
Entry point for creating UFC fighter matchups
"""
import os
from datetime import datetime
import argparse

from creator import UFCMatchupCreator
from config import resolve_data_dir

# Example matchup definitions - used when running without arguments
EXAMPLE_MATCHUPS = [
    {
        'fighter_a': "Austin Hubbard",
        'fighter_b': "MarQuel Mederos",
        'open_odds_a': 160,
        'open_odds_b': -192,
        'closing_odds_a': 171,
        'closing_odds_b': -202,
        'fight_date': "2025-03-29"
    },
    {
        'fighter_a': "Brandon Moreno",
        'fighter_b': "Steve Erceg",
        'open_odds_a': -188,
        'open_odds_b': 150,
        'closing_odds_a': -195,
        'closing_odds_b': 166,
        'fight_date': "2025-03-29"
    },
    {
        'fighter_a': "Christian Rodriguez",
        'fighter_b': "Melquizael Costa",
        'open_odds_a': -175,
        'open_odds_b': 130,
        'closing_odds_a': -145,
        'closing_odds_b': 126,
        'fight_date': "2025-03-29"
    },
    {
        'fighter_a': "CJ Vergara",
        'fighter_b': "Edgar Chairez",
        'open_odds_a': 175,
        'open_odds_b': -250,
        'closing_odds_a': 256,
        'closing_odds_b': -302,
        'fight_date': "2025-03-29"
    },
    {
        'fighter_a': "David Martinez",
        'fighter_b': "Saimon Oliveira",
        'open_odds_a': -175,
        'open_odds_b': 145,
        'closing_odds_a': -185,
        'closing_odds_b': 175,
        'fight_date': "2025-03-29"
    },
    {
        'fighter_a': "Drew Dober",
        'fighter_b': "Manuel Torres",
        'open_odds_a': -120,
        'open_odds_b': -110,
        'closing_odds_a': 106,
        'closing_odds_b': 122,
        'fight_date': "2025-03-29"
    },
    {
        'fighter_a': "Gabriel Miranda",
        'fighter_b': "Jamall Emmers",
        'open_odds_a': 180,
        'open_odds_b': -218,
        'closing_odds_a': 276,
        'closing_odds_b': -347,
        'fight_date': "2025-03-29"
    },
    {
        'fighter_a': "Kevin Borjas",
        'fighter_b': "Ronaldo Rodriguez",
        'open_odds_a': 110,
        'open_odds_b': -150,
        'closing_odds_a': 128,
        'closing_odds_b': -148,
        'fight_date': "2025-03-29"
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
            # Option 1: Create a single matchup
            print("\nCreating a single matchup:")
            matchup = EXAMPLE_MATCHUPS[0]  # Take the first matchup
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
            # Option 2: Create multiple matchups
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
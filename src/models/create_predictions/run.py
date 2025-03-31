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
        'fighter_a': "Brad Tavares",
        'fighter_b': "Gerald Meerschaert",
        'open_odds_a': -210,
        'open_odds_b': 162,
        'closing_odds_a': -260,
        'closing_odds_b': 215,
        'fight_date': "2025-04-05"
    },
    {
        'fighter_a': "ChangHo Lee",
        'fighter_b': "Cortavious Romious",
        'open_odds_a': 205,
        'open_odds_b': -250,
        'closing_odds_a': -160,
        'closing_odds_b': 138,
        'fight_date': "2025-04-05"
    },
    {
        'fighter_a': "Daniel Frunza",
        'fighter_b': "Rhys McKee",
        'open_odds_a': -200,
        'open_odds_b': 150,
        'closing_odds_a': -161,
        'closing_odds_b': 139,
        'fight_date': "2025-04-05"
    },
    {
        'fighter_a': "Daniel Santos",
        'fighter_b': "Davey Grant",
        'open_odds_a': -225,
        'open_odds_b': 163,
        'closing_odds_a': -159,
        'closing_odds_b': 137,
        'fight_date': "2025-04-05"
    },
    {
        'fighter_a': "Joanderson Brito",
        'fighter_b': "Pat Sabatini",
        'open_odds_a': -265,
        'open_odds_b': 200,
        'closing_odds_a': -207,
        'closing_odds_b': 175,
        'fight_date': "2025-04-05"
    },
    {
        'fighter_a': "Luis Gurule",
        'fighter_b': "Ode Osbourne",
        'open_odds_a': -250,
        'open_odds_b': 175,
        'closing_odds_a': -220,
        'closing_odds_b': 185,
        'fight_date': "2025-04-05"
    },
    {
        'fighter_a': "Pedro Falcao",
        'fighter_b': "Victor Henry",
        'open_odds_a': 110,
        'open_odds_b': -150,
        'closing_odds_a': 175,
        'closing_odds_b': -211,
        'fight_date': "2025-04-05"
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
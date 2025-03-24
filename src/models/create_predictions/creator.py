"""
Creator module for generating UFC matchups
"""
import os
import pandas as pd
from typing import List, Dict

from predictor import FighterMatchupPredictor
from config import PATHS, DEBUG
from utils import ensure_directory_exists


class UFCMatchupCreator:
    """Creates matchup data for UFC fights."""

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the matchup creator with a data directory.

        Args:
            data_dir: Path to the data directory
        """
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

    def create_multiple_matchups(
            self,
            matchups: List[Dict],
            output_filename: str = "all_matchups.csv"
    ) -> pd.DataFrame:
        """
        Create multiple matchups for UFC fights and combine them into a single DataFrame,
        sorted alphabetically by fighter_a.

        Args:
            matchups: List of dictionaries, each containing matchup data with keys:
                     'fighter_a', 'fighter_b', 'open_odds_a', 'open_odds_b',
                     'closing_odds_a', 'closing_odds_b', and optionally 'fight_date'
            output_filename: Name of the output CSV file (default: "all_matchups.csv")

        Returns:
            Combined DataFrame with all matchup data, sorted by fighter_a
        """
        all_dfs = []
        for i, matchup in enumerate(matchups):
            if DEBUG:
                print(f"\nProcessing matchup {i + 1} of {len(matchups)}")

            # Extract matchup data with defaults for optional parameters
            fighter_a = matchup['fighter_a']
            fighter_b = matchup['fighter_b']
            open_odds_a = matchup['open_odds_a']
            open_odds_b = matchup['open_odds_b']
            closing_odds_a = matchup['closing_odds_a']
            closing_odds_b = matchup['closing_odds_b']
            fight_date = matchup.get('fight_date', None)  # Optional parameter

            try:
                # Use the internal method with save_individual_file=False to avoid individual file saving
                df = self.matchup_predictor.create_fighter_matchup(
                    fighter_a, fighter_b, open_odds_a, open_odds_b,
                    closing_odds_a, closing_odds_b, fight_date,
                    save_individual_file=False
                )
                all_dfs.append(df)
            except Exception as e:
                print(f"Error creating matchup {fighter_a} vs {fighter_b}: {str(e)}")

        # Combine all DataFrames
        if not all_dfs:
            print("No matchups were successfully created")
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Sort by fighter_a alphabetically
        if 'fighter_a' in combined_df.columns:
            combined_df = combined_df.sort_values(by='fighter_a')

        # Create matchup data directory if it doesn't exist
        matchup_dir = os.path.join(self.data_dir, PATHS['matchup_data'])
        ensure_directory_exists(matchup_dir)

        # Save the combined data
        output_path = os.path.join(matchup_dir, output_filename)
        self.matchup_predictor.fight_processor._save_csv(combined_df, output_path)
        if DEBUG:
            print(f"\nCreated {len(all_dfs)} matchups successfully out of {len(matchups)} requested")
            print(f"Combined output saved to: {output_path}")

        return combined_df
"""Helpers for processing betting odds data."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

from .data_utils import OddsPair, PathLike, resolve_data_directory


logger = logging.getLogger(__name__)


class OddsUtils:
    """Utility methods for processing betting odds information."""

    def __init__(
        self,
        *,
        data_dir: Optional[Union[str, Path]] = None,
        odds_filename: Union[str, Path] = "processed/cleaned_fight_odds.csv",
    ) -> None:
        module_path = Path(__file__).resolve()
        self._data_dir = resolve_data_directory(data_dir, module_path, default_subdir="data")
        self._odds_filename = Path(odds_filename)

    def _resolve_odds_path(self, odds_filepath: Optional[Union[str, Path]] = None) -> Path:
        if odds_filepath is not None:
            candidate = Path(odds_filepath).expanduser()
            if not candidate.is_absolute():
                candidate = self._data_dir / candidate
        else:
            candidate = self._data_dir / self._odds_filename

        candidate = candidate.expanduser()
        if not candidate.exists():
            raise FileNotFoundError(
                f"Odds data file not found at {candidate}. "
                "Provide a valid path or ensure the data directory is correct."
            )

        return candidate

    @staticmethod
    def round_to_nearest_1(value: float) -> int:
        return round(value)

    @staticmethod
    def calculate_complementary_odd(odd: float) -> float:
        if odd > 0:
            prob = 100 / (odd + 100)
        else:
            prob = abs(odd) / (abs(odd) + 100)

        complementary_prob = 1.045 - prob

        if complementary_prob >= 0.5:
            complementary_odd = -100 * complementary_prob / (1 - complementary_prob)
        else:
            complementary_odd = 100 * (1 - complementary_prob) / complementary_prob

        return OddsUtils.round_to_nearest_1(complementary_odd)

    def process_odds_pair(
        self,
        odds_a: Optional[float],
        odds_b: Optional[float],
    ) -> Tuple[List[float], float, float]:
        if pd.notna(odds_a) and pd.notna(odds_b):
            pair = OddsPair(float(odds_a), float(odds_b))
            return [pair.fighter_a, pair.fighter_b], pair.diff, pair.ratio

        if pd.notna(odds_a):
            rounded_a = self.round_to_nearest_1(float(odds_a))
            inferred_b = self.calculate_complementary_odd(rounded_a)
            pair = OddsPair(float(rounded_a), float(inferred_b))
            return [pair.fighter_a, pair.fighter_b], pair.diff, pair.ratio

        if pd.notna(odds_b):
            rounded_b = self.round_to_nearest_1(float(odds_b))
            inferred_a = self.calculate_complementary_odd(rounded_b)
            pair = OddsPair(float(inferred_a), float(rounded_b))
            return [pair.fighter_a, pair.fighter_b], pair.diff, pair.ratio

        fallback = OddsPair(-111.0, -111.0)
        return [fallback.fighter_a, fallback.fighter_b], fallback.diff, fallback.ratio

    def process_odds_data(
        self,
        final_stats: pd.DataFrame,
        odds_filepath: Optional[PathLike] = None,
    ) -> pd.DataFrame:
        final_stats = final_stats.copy().loc[:, ~final_stats.columns.duplicated()]

        odds_path = self._resolve_odds_path(odds_filepath)
        try:
            odds_df = pd.read_csv(odds_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.exception("Failed to load odds data from %s", odds_path)
            raise RuntimeError(f"Failed to load odds data from {odds_path}") from exc

        final_stats["fighter"] = final_stats["fighter"].str.lower().str.strip()
        odds_df["Matchup"] = odds_df["Matchup"].str.lower().str.strip()
        odds_df.rename(columns={"Matchup": "fighter"}, inplace=True)

        final_stats["fight_date"] = pd.to_datetime(final_stats["fight_date"])
        odds_df["Date"] = pd.to_datetime(odds_df["Date"], format="%Y-%m-%d")

        final_stats.sort_values("fight_date", inplace=True)
        odds_df.sort_values("Date", inplace=True)

        merged_df = pd.merge_asof(
            final_stats,
            odds_df,
            left_on="fight_date",
            right_on="Date",
            by="fighter",
            tolerance=pd.Timedelta("1D"),
            direction="nearest",
        )

        merged_df.drop(columns=["Date"], inplace=True)
        merged_df.rename(
            columns={
                "Open": "open_odds",
                "Closing Range Start": "closing_range_start",
                "Closing Range End": "closing_range_end",
                "Movement": "odds_movement",
            },
            inplace=True,
        )

        return merged_df


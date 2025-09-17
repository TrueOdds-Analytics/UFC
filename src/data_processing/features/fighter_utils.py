"""Utilities for aggregating fighter-level statistics."""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

from .data_utils import DataUtils


logger = logging.getLogger(__name__)


class FighterUtils:
    """Helper functions for computing cumulative fighter statistics."""

    def __init__(self, enable_verification: bool = True) -> None:
        self.utils = DataUtils()
        self.enable_verification = enable_verification
        self.verification_results: List[tuple[str, str, bool]] = []

    def aggregate_fighter_stats(self, group: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        group = group.sort_values("fight_date")
        cumulative_stats = group[numeric_columns].cumsum(skipna=True)
        fight_count = group.groupby("fighter").cumcount() + 1

        for column in numeric_columns:
            group[f"{column}_career"] = cumulative_stats[column]
            group[f"{column}_career_avg"] = self.utils.safe_divide(
                cumulative_stats[column], fight_count
            )

        if self.enable_verification and len(group) > 0:
            self._verify_career_stats(group, numeric_columns)

        group["significant_strikes_rate_career"] = self.utils.safe_divide(
            cumulative_stats.get("significant_strikes_landed", 0),
            cumulative_stats.get("significant_strikes_attempted", 1),
        )
        group["takedown_rate_career"] = self.utils.safe_divide(
            cumulative_stats.get("takedown_successful", 0),
            cumulative_stats.get("takedown_attempted", 1),
        )
        group["total_strikes_rate_career"] = self.utils.safe_divide(
            cumulative_stats.get("total_strikes_landed", 0),
            cumulative_stats.get("total_strikes_attempted", 1),
        )
        group["combined_success_rate_career"] = (
            group["takedown_rate_career"] + group["total_strikes_rate_career"]
        ) / 2

        return group

    def calculate_experience_and_days(self, group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("fight_date")
        group["years_of_experience"] = (
            group["fight_date"] - group["fight_date"].iloc[0]
        ).dt.days / 365.25
        group["days_since_last_fight"] = (
            group["fight_date"] - group["fight_date"].shift()
        ).dt.days
        return group

    def update_streaks(self, group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("fight_date").copy()
        group["win_streak"] = 0
        group["loss_streak"] = 0

        for idx in range(1, len(group)):
            previous = group.iloc[idx - 1]
            if previous["winner"] == 1:
                group.iat[idx, group.columns.get_loc("win_streak")] = previous["win_streak"] + 1
                group.iat[idx, group.columns.get_loc("loss_streak")] = 0
            else:
                group.iat[idx, group.columns.get_loc("win_streak")] = 0
                group.iat[idx, group.columns.get_loc("loss_streak")] = previous["loss_streak"] + 1

        if self.enable_verification and len(group) > 0:
            self._verify_streaks(group)

        return group

    def calculate_time_based_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["time_career_minutes"] = df["time_career"] / 60
        df["takedowns_per_15min"] = self.utils.safe_divide(
            df["takedown_successful_career"], df["time_career_minutes"]
        ) * 15
        df["knockdowns_per_15min"] = self.utils.safe_divide(
            df["knockdowns_career"], df["time_career_minutes"]
        ) * 15
        return df

    def calculate_total_fight_stats(self, group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("fight_date").reset_index(drop=True)
        group["total_fights"] = range(1, len(group) + 1)
        group["total_wins"] = group["winner"].cumsum()
        group["total_losses"] = group["total_fights"] - group["total_wins"]

        if self.enable_verification and len(group) > 0:
            self._verify_total_fights(group)

        ko_mask = group["result"].isin([0, 3])
        submission_mask = group["result"] == 1
        decision_mask = group["result"].isin([2, 4])

        win_mask = group["winner"] == 1
        loss_mask = ~win_mask

        group["wins_by_ko"] = (ko_mask & win_mask).cumsum()
        group["wins_by_submission"] = (submission_mask & win_mask).cumsum()
        group["wins_by_decision"] = (decision_mask & win_mask).cumsum()

        group["losses_by_ko"] = (ko_mask & loss_mask).cumsum()
        group["losses_by_submission"] = (submission_mask & loss_mask).cumsum()
        group["losses_by_decision"] = (decision_mask & loss_mask).cumsum()

        for outcome in ("ko", "submission", "decision"):
            group[f"win_rate_by_{outcome}"] = self.utils.safe_divide(
                group[f"wins_by_{outcome}"], group["total_wins"]
            )
            group[f"loss_rate_by_{outcome}"] = self.utils.safe_divide(
                group[f"losses_by_{outcome}"], group["total_losses"]
            )

        return group

    def print_verification_summary(self) -> None:
        if not self.verification_results:
            return

        passed = sum(1 for _, _, result in self.verification_results if result)
        total = len(self.verification_results)

        logger.info("LEAKAGE VERIFICATION SUMMARY")
        logger.info("Checks passed: %s/%s", passed, total)
        if passed != total:
            logger.warning("Some verification checks failed – review the log output above")

    # ------------------------------------------------------------------
    # Internal verification helpers
    # ------------------------------------------------------------------

    def _verify_career_stats(self, group: pd.DataFrame, numeric_columns: List[str]) -> None:
        fighter_name = group["fighter"].iloc[0]
        verification_passed = True

        for fight_idx in range(min(3, len(group))):
            fight = group.iloc[fight_idx]
            if "knockdowns" in numeric_columns and "knockdowns_career" in group.columns:
                expected_knockdowns = group["knockdowns"].iloc[: fight_idx + 1].sum()
                actual_knockdowns = fight.get("knockdowns_career", 0)
                if abs(expected_knockdowns - actual_knockdowns) > 0.01:
                    logger.error(
                        "LEAKAGE CHECK #1 failed for %s (fight %s): expected %s knockdowns but found %s",
                        fighter_name,
                        fight_idx + 1,
                        expected_knockdowns,
                        actual_knockdowns,
                    )
                    verification_passed = False
                    break

        if verification_passed:
            self.verification_results.append(("career_stats", fighter_name, True))

    def _verify_streaks(self, group: pd.DataFrame) -> None:
        fighter_name = group["fighter"].iloc[0]
        first_fight = group.iloc[0]

        if first_fight["win_streak"] != 0 or first_fight["loss_streak"] != 0:
            logger.error(
                "LEAKAGE CHECK #2 failed for %s: first fight streaks should be zero",
                fighter_name,
            )
            self.verification_results.append(("streaks", fighter_name, False))
            return

        streak_valid = True
        for idx in range(1, min(3, len(group))):
            previous = group.iloc[idx - 1]
            current = group.iloc[idx]
            if previous["winner"] == 1:
                expected_win = previous["win_streak"] + 1
                if current["win_streak"] != expected_win or current["loss_streak"] != 0:
                    streak_valid = False
                    break
            else:
                expected_loss = previous["loss_streak"] + 1
                if current["loss_streak"] != expected_loss or current["win_streak"] != 0:
                    streak_valid = False
                    break

        if streak_valid:
            self.verification_results.append(("streaks", fighter_name, True))
        else:
            logger.error("LEAKAGE CHECK #2 streak continuity error for %s", fighter_name)

    def _verify_total_fights(self, group: pd.DataFrame) -> None:
        fighter_name = group["fighter"].iloc[0]
        verification_passed = True

        first_fight = group.iloc[0]
        if first_fight["total_fights"] != 1:
            logger.error(
                "LEAKAGE CHECK #3 failed for %s: first fight total_fights should equal 1",
                fighter_name,
            )
            verification_passed = False

        for idx in range(min(3, len(group))):
            fight = group.iloc[idx]
            if fight["total_fights"] != idx + 1:
                logger.error(
                    "LEAKAGE CHECK #3 progression error for %s (fight %s)",
                    fighter_name,
                    idx + 1,
                )
                verification_passed = False
                break

        if verification_passed:
            self.verification_results.append(("total_fights", fighter_name, True))

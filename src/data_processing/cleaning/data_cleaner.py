"""End-to-end preprocessing pipeline for UFC fight data."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.data_processing.features import (
    DataUtils,
    FighterUtils,
    OddsUtils,
    resolve_data_directory,
)
from src.data_processing.features.Elo import calculate_elo_ratings


logger = logging.getLogger(__name__)


PathLike = Path | str


class FightDataProcessor:
    """Load, clean and enrich fighter round statistics."""

    def __init__(self, data_dir: PathLike | None = "../../../data", *, enable_verification: bool = True) -> None:
        module_path = Path(__file__).resolve()
        self.data_dir = resolve_data_directory(data_dir, module_path, default_subdir="data")
        self.utils = DataUtils()
        self.odds_utils = OddsUtils(data_dir=self.data_dir)
        self.fighter_utils = FighterUtils(enable_verification=enable_verification)
        self.enable_verification = enable_verification

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, relative_path: PathLike) -> Path:
        path = Path(relative_path)
        if not path.is_absolute():
            path = (self.data_dir / path).resolve()
        return path

    def _load_csv(self, relative_path: PathLike) -> pd.DataFrame:
        path = self._resolve_path(relative_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found at {path}")
        return pd.read_csv(path)

    def _save_csv(self, df: pd.DataFrame, relative_path: PathLike) -> Path:
        path = self._resolve_path(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved %s", path)
        return path

    # ------------------------------------------------------------------
    # Round aggregation pipeline
    # ------------------------------------------------------------------

    def combine_rounds_stats(self, file_path: PathLike) -> pd.DataFrame:
        logger.info("Loading and preprocessing round statistics from %s", file_path)
        rounds = self._load_csv(file_path)
        fighter_stats = self._load_csv("raw/ufc_fighter_tott.csv")
        rounds = self.utils.preprocess_data(rounds, fighter_stats)

        numeric_columns = self._get_numeric_columns(rounds)
        aggregated = self._aggregate_round_statistics(rounds, numeric_columns)
        metadata = self._build_round_metadata(rounds)

        combined = aggregated.merge(metadata, on=["id", "fighter"], how="left")
        combined = self._finalise_fighter_careers(combined, numeric_columns)
        combined = self.odds_utils.process_odds_data(combined)
        combined = self._drop_duplicate_columns(combined)

        combined = combined.sort_values(["fighter", "fight_date"])
        combined = combined.groupby("fighter", group_keys=False).apply(
            self.fighter_utils.calculate_experience_and_days
        )
        combined = combined.groupby("fighter", group_keys=False).apply(
            self.fighter_utils.update_streaks
        )
        combined["days_since_last_fight"] = combined["days_since_last_fight"].fillna(0)
        combined = self.fighter_utils.calculate_time_based_stats(combined)
        combined = combined.groupby("fighter", group_keys=False).apply(
            self.fighter_utils.calculate_total_fight_stats
        )

        if self.enable_verification:
            self.fighter_utils.print_verification_summary()

        self._save_csv(combined, "processed/combined_rounds.csv")
        return combined

    def _aggregate_round_statistics(
        self, rounds: pd.DataFrame, numeric_columns: Sequence[str]
    ) -> pd.DataFrame:
        aggregated = (
            rounds.groupby(["id", "fighter"], as_index=False)[list(numeric_columns)].sum()
        )
        return self._calculate_basic_rates(aggregated)

    def _build_round_metadata(self, rounds: pd.DataFrame) -> pd.DataFrame:
        max_round_data = (
            rounds.groupby("id")[["last_round", "time"]].max().reset_index()
        )

        non_numeric_columns = (
            rounds.select_dtypes(exclude=[np.number]).columns.difference(["id", "fighter"])
        )
        metadata = (
            rounds.drop_duplicates(subset=["id", "fighter"])[
                ["id", "fighter", "age", *list(non_numeric_columns)]
            ]
        )

        return metadata.merge(max_round_data, on="id", how="left")

    def _finalise_fighter_careers(
        self, combined: pd.DataFrame, numeric_columns: Sequence[str]
    ) -> pd.DataFrame:
        combined = combined.groupby("fighter", group_keys=False).apply(
            self.fighter_utils.aggregate_fighter_stats, numeric_columns=list(numeric_columns)
        )
        combined = self._calculate_per_minute_stats(combined)
        combined = self._calculate_additional_rates(combined)
        combined = self._filter_unwanted_results(combined)
        combined = self._factorize_categorical_columns(combined)
        return combined

    def _drop_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        duplicate_columns = df.columns[df.columns.duplicated()]
        if len(duplicate_columns) > 0:
            logger.info("Dropped duplicate columns: %s", sorted(set(duplicate_columns)))
        return df.loc[:, ~df.columns.duplicated()]

    def _get_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col not in {"id", "last_round", "age"}]
        if "time" not in numeric_columns:
            numeric_columns.append("time")
        return numeric_columns

    def _calculate_basic_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        df["significant_strikes_rate"] = self.utils.safe_divide(
            df["significant_strikes_landed"], df["significant_strikes_attempted"]
        )
        df["takedown_rate"] = self.utils.safe_divide(
            df["takedown_successful"], df["takedown_attempted"]
        )
        return df

    def _calculate_per_minute_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        df["fight_duration_minutes"] = self.utils.safe_divide(df["time"], 60)
        per_minute_columns = [
            "significant_strikes_landed",
            "significant_strikes_attempted",
            "total_strikes_landed",
            "total_strikes_attempted",
        ]
        for column in per_minute_columns:
            df[f"{column}_per_min"] = self.utils.safe_divide(
                df[column], df["fight_duration_minutes"]
            )
        return df

    def _calculate_additional_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        df["total_strikes_rate"] = self.utils.safe_divide(
            df["total_strikes_landed"], df["total_strikes_attempted"]
        )
        df["combined_success_rate"] = (
            df["takedown_rate"] + df["total_strikes_rate"]
        ) / 2
        return df

    @staticmethod
    def _filter_unwanted_results(df: pd.DataFrame) -> pd.DataFrame:
        df = df[~df["winner"].isin(["NC/NC", "D/D"])].copy()
        df = df[~df["result"].isin(["DQ", "DQ ", "Could Not Continue ", "Overturned ", "Other "])].copy()
        return df

    def _factorize_categorical_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in ["result", "winner", "scheduled_rounds"]:
            df[column], unique = pd.factorize(df[column])
            mapping = {index: label for index, label in enumerate(unique)}
            logger.debug("Mapping for %s: %s", column, mapping)
        return df

    # ------------------------------------------------------------------
    # Fighter pairing
    # ------------------------------------------------------------------

    def combine_fighters_stats(self, file_path: PathLike) -> pd.DataFrame:
        df = self._load_csv(file_path)
        df = df.drop(columns=[col for col in df.columns if "event" in col.lower()])
        df = df.sort_values(by=["id", "fighter"])

        paired_rows: list[pd.Series] = []
        skipped = 0
        for _, group in df.groupby("id", sort=False):
            if len(group) != 2:
                skipped += 1
                continue
            paired_rows.extend(self._create_fight_pair_rows(group))

        if skipped:
            logger.warning("Skipped %s fights with missing fighter data", skipped)

        final_combined_df = pd.DataFrame(paired_rows).reset_index(drop=True)
        final_combined_df = self._calculate_differential_and_ratio_features(final_combined_df)
        final_combined_df = final_combined_df[~final_combined_df["winner"].isin(["NC", "D"])]
        final_combined_df["fight_date"] = pd.to_datetime(final_combined_df["fight_date"])
        final_combined_df = final_combined_df.sort_values(by=["fighter", "fight_date"], ascending=True)

        self._save_csv(final_combined_df, "processed/combined_sorted_fighter_stats.csv")
        return final_combined_df

    def _create_fight_pair_rows(self, group: pd.DataFrame) -> List[pd.Series]:
        fighters = group.sort_values("fighter").reset_index(drop=True)
        fighter_a = fighters.iloc[0]
        fighter_b = fighters.iloc[1]
        original = pd.concat([fighter_a, fighter_b.add_suffix("_b")])
        mirrored = pd.concat([fighter_b, fighter_a.add_suffix("_b")])
        return [original, mirrored]

    def _calculate_differential_and_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        base_columns = [
            "knockdowns",
            "significant_strikes_landed",
            "significant_strikes_attempted",
            "significant_strikes_rate",
            "total_strikes_landed",
            "total_strikes_attempted",
            "takedown_successful",
            "takedown_attempted",
            "takedown_rate",
            "submission_attempt",
            "reversals",
            "head_landed",
            "head_attempted",
            "body_landed",
            "body_attempted",
            "leg_landed",
            "leg_attempted",
            "distance_landed",
            "distance_attempted",
            "clinch_landed",
            "clinch_attempted",
            "ground_landed",
            "ground_attempted",
        ]
        other_columns = [
            "open_odds",
            "closing_range_start",
            "closing_range_end",
            "pre_fight_elo",
            "years_of_experience",
            "win_streak",
            "loss_streak",
            "days_since_last_fight",
            "significant_strikes_landed_per_min",
            "significant_strikes_attempted_per_min",
            "total_strikes_landed_per_min",
            "total_strikes_attempted_per_min",
            "takedowns_per_15min",
            "knockdowns_per_15min",
            "total_fights",
            "total_wins",
            "total_losses",
            "wins_by_ko",
            "losses_by_ko",
            "wins_by_submission",
            "losses_by_submission",
            "wins_by_decision",
            "losses_by_decision",
            "win_rate_by_ko",
            "loss_rate_by_ko",
            "win_rate_by_submission",
            "loss_rate_by_submission",
            "win_rate_by_decision",
            "loss_rate_by_decision",
        ]

        columns_to_process = (
            base_columns
            + [f"{col}_career" for col in base_columns]
            + [f"{col}_career_avg" for col in base_columns]
            + other_columns
        )

        diff_features = {}
        ratio_features = {}

        for column in columns_to_process:
            column_b = f"{column}_b"
            if column in df.columns and column_b in df.columns:
                diff_features[f"{column}_diff"] = df[column] - df[column_b]
                ratio_features[f"{column}_ratio"] = self.utils.safe_divide(df[column], df[column_b])

        return pd.concat([df, pd.DataFrame(diff_features), pd.DataFrame(ratio_features)], axis=1)

class MatchupProcessor:
    """Generate matchup-level features for modelling."""

    def __init__(self, data_dir: PathLike | None = "../../../data", *, enable_verification: bool = True) -> None:
        self.fight_processor = FightDataProcessor(data_dir=data_dir, enable_verification=enable_verification)
        self.data_dir = self.fight_processor.data_dir
        self.utils = DataUtils()
        self.odds_utils = OddsUtils(data_dir=self.data_dir)
        self.enable_verification = enable_verification
        self.leakage_warnings: list[str] = []

    # ------------------------------------------------------------------
    # Matchup creation
    # ------------------------------------------------------------------

    def create_matchup_data(
        self,
        file_path: PathLike,
        tester: int,
        include_names: bool = False,
    ) -> pd.DataFrame:
        logger.info("Creating matchup data from %s", file_path)
        df = self.fight_processor._load_csv(file_path)
        df["fight_date"] = pd.to_datetime(df["fight_date"])
        df = df.sort_values("fight_date")

        n_past_fights = 6 - tester
        columns_to_exclude = {
            "fighter",
            "id",
            "fighter_b",
            "fight_date",
            "fight_date_b",
            "result",
            "winner",
            "weight_class",
            "scheduled_rounds",
            "result_b",
            "winner_b",
            "weight_class_b",
            "scheduled_rounds_b",
        }
        features_to_include = [
            column
            for column in df.columns
            if column not in columns_to_exclude and column != "age" and not column.endswith("_age")
        ]
        method_columns = ["winner"]

        matchup_rows = self._process_matchups(
            df,
            features_to_include,
            method_columns,
            n_past_fights,
            tester,
            include_names,
        )

        column_names = self._generate_column_names(
            features_to_include,
            method_columns,
            n_past_fights,
            tester,
            include_names,
        )
        matchup_df = pd.DataFrame(matchup_rows, columns=column_names)
        matchup_df = matchup_df.drop(columns=["fight_date"], errors="ignore")
        matchup_df.columns = [self.utils.rename_columns_general(column) for column in matchup_df.columns]
        matchup_df = self._calculate_matchup_features(matchup_df, features_to_include, n_past_fights)

        if self.enable_verification and self.leakage_warnings:
            logger.warning("MATCHUP DATA LEAKAGE WARNINGS:")
            for warning in self.leakage_warnings[:10]:
                logger.warning("%s", warning)
            if len(self.leakage_warnings) > 10:
                logger.warning("... and %s more warnings", len(self.leakage_warnings) - 10)

        suffix = "_name" if include_names else ""
        output_filename = f"matchup data/matchup_data_{n_past_fights}_avg{suffix}.csv"
        self.fight_processor._save_csv(matchup_df, output_filename)
        return matchup_df

    def _process_matchups(
        self,
        df: pd.DataFrame,
        features_to_include: Sequence[str],
        method_columns: Sequence[str],
        n_past_fights: int,
        tester: int,
        include_names: bool,
    ) -> List[List]:
        fighter_histories = {
            fighter: group.sort_values("fight_date")
            for fighter, group in df.groupby("fighter")
        }

        matchup_rows: List[List] = []
        skipped = 0
        partial = 0
        verification_sample_size = 5

        for idx, current_fight in df.iterrows():
            fighter_a = current_fight["fighter"]
            fighter_b = current_fight["fighter_b"]

            history_a = self._recent_fights(
                fighter_histories.get(fighter_a, pd.DataFrame()),
                current_fight["fight_date"],
                n_past_fights,
            )
            history_b = self._recent_fights(
                fighter_histories.get(fighter_b, pd.DataFrame()),
                current_fight["fight_date"],
                n_past_fights,
            )

            if self.enable_verification and idx < verification_sample_size:
                self._verify_recent_fights(fighter_a, fighter_b, history_a, history_b, current_fight)

            if history_a.empty or history_b.empty:
                skipped += 1
                continue

            if len(history_a) < n_past_fights or len(history_b) < n_past_fights:
                partial += 1

            matchup_row = self._build_matchup_row(
                current_fight,
                history_a,
                history_b,
                features_to_include,
                method_columns,
                tester,
            )

            most_recent_date = max(history_a["fight_date"].max(), history_b["fight_date"].max())
            current_fight_date = current_fight["fight_date"]

            if include_names:
                prefix = [fighter_a, fighter_b, most_recent_date]
            else:
                prefix = [most_recent_date]

            matchup_rows.append(prefix + matchup_row + [current_fight_date])

        logger.info(
            "Processed %s matchups (including %s with partial fight history); skipped %s with no history",
            len(matchup_rows),
            partial,
            skipped,
        )
        return matchup_rows

    def _recent_fights(
        self, history: pd.DataFrame, fight_date: pd.Timestamp, limit: int
    ) -> pd.DataFrame:
        if history.empty:
            return history
        prior_fights = history[history["fight_date"] < fight_date]
        if prior_fights.empty:
            return prior_fights
        return prior_fights.sort_values("fight_date", ascending=False).head(limit)

    def _verify_recent_fights(
        self,
        fighter_a: str,
        fighter_b: str,
        history_a: pd.DataFrame,
        history_b: pd.DataFrame,
        current_fight: pd.Series,
    ) -> None:
        for fighter_name, history in ((fighter_a, history_a), (fighter_b, history_b)):
            if history.empty:
                warning = f"❌ CRITICAL LEAKAGE: Fighter {fighter_name} has no past fights before {current_fight['fight_date']}"
                self.leakage_warnings.append(warning)
                continue
            latest_past_fight = history.iloc[0]
            if latest_past_fight["fight_date"] >= current_fight["fight_date"]:
                warning = (
                    f"❌ CRITICAL LEAKAGE: Fighter {fighter_name} uses future fight data"
                )
                self.leakage_warnings.append(warning)

    def _build_matchup_row(
        self,
        current_fight: pd.Series,
        history_a: pd.DataFrame,
        history_b: pd.DataFrame,
        features_to_include: Sequence[str],
        method_columns: Sequence[str],
        tester: int,
    ) -> List:
        fighter_a_features = history_a[features_to_include].mean(numeric_only=True).to_numpy()
        fighter_b_features = history_b[features_to_include].mean(numeric_only=True).to_numpy()

        results_a = self._extract_recent_results(
            history_a[["result", "winner", "weight_class", "scheduled_rounds"]],
            tester,
        )
        results_b = self._extract_recent_results(
            history_b[["result_b", "winner_b", "weight_class_b", "scheduled_rounds_b"]],
            tester,
        )

        open_odds, open_diff, open_ratio = self._process_fight_odds(
            current_fight["open_odds"], current_fight["open_odds_b"]
        )
        closing_odds, closing_diff, closing_ratio = self._process_fight_odds(
            current_fight["closing_range_end"], current_fight["closing_range_end_b"]
        )
        closing_open_diff = [
            current_fight["closing_range_end"] - current_fight["open_odds"],
            current_fight["closing_range_end_b"] - current_fight["open_odds_b"],
        ]

        ages = [current_fight["age"], current_fight["age_b"]]
        age_diff = current_fight["age"] - current_fight["age_b"]
        age_ratio = self.utils.safe_divide(current_fight["age"], current_fight["age_b"])

        elo_stats, elo_ratio = self._process_elo_stats(current_fight)
        other_stats = self._process_other_stats(current_fight)
        labels = current_fight[list(method_columns)].to_list()

        return (
            list(fighter_a_features)
            + list(fighter_b_features)
            + list(results_a)
            + list(results_b)
            + list(open_odds)
            + [open_diff, open_ratio]
            + list(closing_odds)
            + [closing_diff, closing_ratio, *closing_open_diff]
            + ages
            + [age_diff, age_ratio]
            + list(elo_stats)
            + [elo_ratio]
            + other_stats
            + labels
        )

    def _extract_recent_results(self, results: pd.DataFrame, tester: int) -> np.ndarray:
        available = results.head(min(len(results), tester)).values.flatten()
        pad_length = tester * results.shape[1] - len(available)
        if pad_length <= 0:
            return available
        return np.pad(available, (0, pad_length), constant_values=np.nan)

    def _process_fight_odds(self, odds_a: float, odds_b: float) -> Tuple[List[float], float, float]:
        return self.odds_utils.process_odds_pair(odds_a, odds_b)

    def _process_elo_stats(self, current_fight: pd.Series) -> Tuple[List[float], float]:
        elo_a = current_fight["pre_fight_elo"]
        elo_b = current_fight["pre_fight_elo_b"]
        elo_diff = current_fight["pre_fight_elo_diff"]

        a_win_prob = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        b_win_prob = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))

        elo_stats = [elo_a, elo_b, elo_diff, a_win_prob, b_win_prob]
        elo_ratio = self.utils.safe_divide(elo_a, elo_b)
        return elo_stats, elo_ratio

    def _process_other_stats(self, current_fight: pd.Series) -> List[float]:
        win_streak_a = current_fight["win_streak"]
        win_streak_b = current_fight["win_streak_b"]
        loss_streak_a = current_fight["loss_streak"]
        loss_streak_b = current_fight["loss_streak_b"]

        exp_a = current_fight["years_of_experience"]
        exp_b = current_fight["years_of_experience_b"]

        days_since_a = current_fight["days_since_last_fight"]
        days_since_b = current_fight["days_since_last_fight_b"]

        return [
            win_streak_a,
            win_streak_b,
            win_streak_a - win_streak_b,
            self.utils.safe_divide(win_streak_a, win_streak_b),
            loss_streak_a,
            loss_streak_b,
            loss_streak_a - loss_streak_b,
            self.utils.safe_divide(loss_streak_a, loss_streak_b),
            exp_a,
            exp_b,
            exp_a - exp_b,
            self.utils.safe_divide(exp_a, exp_b),
            days_since_a,
            days_since_b,
            days_since_a - days_since_b,
            self.utils.safe_divide(days_since_a, days_since_b),
        ]

    def _generate_column_names(
        self,
        features_to_include: Sequence[str],
        method_columns: Sequence[str],
        n_past_fights: int,
        tester: int,
        include_names: bool,
    ) -> List[str]:
        results_columns = []
        for i in range(1, tester + 1):
            results_columns.extend(
                [
                    f"result_fight_{i}",
                    f"winner_fight_{i}",
                    f"weight_class_fight_{i}",
                    f"scheduled_rounds_fight_{i}",
                    f"result_b_fight_{i}",
                    f"winner_b_fight_{i}",
                    f"weight_class_b_fight_{i}",
                    f"scheduled_rounds_b_fight_{i}",
                ]
            )

        new_columns = [
            "current_fight_pre_fight_elo_a",
            "current_fight_pre_fight_elo_b",
            "current_fight_pre_fight_elo_diff",
            "current_fight_pre_fight_elo_a_win_chance",
            "current_fight_pre_fight_elo_b_win_chance",
            "current_fight_pre_fight_elo_ratio",
            "current_fight_win_streak_a",
            "current_fight_win_streak_b",
            "current_fight_win_streak_diff",
            "current_fight_win_streak_ratio",
            "current_fight_loss_streak_a",
            "current_fight_loss_streak_b",
            "current_fight_loss_streak_diff",
            "current_fight_loss_streak_ratio",
            "current_fight_years_experience_a",
            "current_fight_years_experience_b",
            "current_fight_years_experience_diff",
            "current_fight_years_experience_ratio",
            "current_fight_days_since_last_a",
            "current_fight_days_since_last_b",
            "current_fight_days_since_last_diff",
            "current_fight_days_since_last_ratio",
        ]

        base_columns = ["fight_date"] if not include_names else ["fighter_a", "fighter_b", "fight_date"]
        feature_columns = [
            *(f"{feature}_fighter_avg_last_{n_past_fights}" for feature in features_to_include),
            *(f"{feature}_fighter_b_avg_last_{n_past_fights}" for feature in features_to_include),
        ]
        odds_age_columns = [
            "current_fight_open_odds",
            "current_fight_open_odds_b",
            "current_fight_open_odds_diff",
            "current_fight_open_odds_ratio",
            "current_fight_closing_odds",
            "current_fight_closing_odds_b",
            "current_fight_closing_odds_diff",
            "current_fight_closing_odds_ratio",
            "current_fight_closing_open_diff_a",
            "current_fight_closing_open_diff_b",
            "current_fight_age",
            "current_fight_age_b",
            "current_fight_age_diff",
            "current_fight_age_ratio",
        ]

        return (
            base_columns
            + feature_columns
            + results_columns
            + odds_age_columns
            + new_columns
            + [str(method) for method in method_columns]
            + ["current_fight_date"]
        )

    def _calculate_matchup_features(
        self,
        df: pd.DataFrame,
        features_to_include: Sequence[str],
        n_past_fights: int,
    ) -> pd.DataFrame:
        diff_columns = {}
        ratio_columns = {}

        for feature in features_to_include:
            col_a = f"{feature}_fighter_a_avg_last_{n_past_fights}"
            col_b = f"{feature}_fighter_b_avg_last_{n_past_fights}"
            if col_a in df.columns and col_b in df.columns:
                diff_columns[f"matchup_{feature}_diff_avg_last_{n_past_fights}"] = df[col_a] - df[col_b]
                ratio_columns[f"matchup_{feature}_ratio_avg_last_{n_past_fights}"] = self.utils.safe_divide(
                    df[col_a], df[col_b]
                )

        return pd.concat([df, pd.DataFrame(diff_columns), pd.DataFrame(ratio_columns)], axis=1)

    # ------------------------------------------------------------------
    # Temporal splits
    # ------------------------------------------------------------------

    def split_train_val_test(
        self,
        matchup_data_file: PathLike,
        start_date: str,
        end_date: str,
        years_back: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info(
            "Splitting matchup data from %s to %s with %s years lookback",
            start_date,
            end_date,
            years_back,
        )
        matchup_df = self.fight_processor._load_csv(matchup_data_file)
        matchup_df["current_fight_date"] = pd.to_datetime(matchup_df["current_fight_date"])

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        history_start = start_dt - pd.DateOffset(years=years_back)

        test_data = matchup_df[
            (matchup_df["current_fight_date"] >= start_dt)
            & (matchup_df["current_fight_date"] <= end_dt)
        ].copy()

        remaining = matchup_df[
            (matchup_df["current_fight_date"] >= history_start)
            & (matchup_df["current_fight_date"] < start_dt)
        ].copy()

        remaining = remaining.sort_values("current_fight_date")
        unique_dates = sorted(remaining["current_fight_date"].unique())
        split_index = int(len(unique_dates) * 0.8)

        if split_index < len(unique_dates):
            cutoff = unique_dates[split_index]
            train_data = remaining[remaining["current_fight_date"] < cutoff].copy()
            val_data = remaining[remaining["current_fight_date"] >= cutoff].copy()
            logger.info("Temporal split cutoff date: %s", cutoff)
        else:
            train_data = remaining.copy()
            val_data = pd.DataFrame()
            logger.info("Not enough unique dates for validation split; validation set left empty")

        test_data = self._remove_duplicate_fights(test_data, random=False)
        train_data = train_data.sort_values("current_fight_date")
        if not val_data.empty:
            val_data = val_data.sort_values("current_fight_date")
        test_data = test_data.sort_values(["current_fight_date", "fighter_a"], ascending=[True, True])

        removed_features: List[str] = []
        if not train_data.empty:
            train_data, removed_features = self.utils.remove_correlated_features(
                train_data,
                correlation_threshold=0.95,
                protected_columns=[
                    "winner",
                    "current_fight_open_odds_diff",
                    "current_fight_closing_range_end_b",
                    "current_fight_closing_odds_diff",
                ],
            )
            if removed_features:
                val_data = val_data.drop(columns=removed_features, errors="ignore")
                test_data = test_data.drop(columns=removed_features, errors="ignore")
                logger.info("Removed %s correlated features", len(removed_features))

        if self.enable_verification:
            self._verify_split(train_data, val_data, test_data)

        self.fight_processor._save_csv(train_data, "train_test/train_data.csv")
        self.fight_processor._save_csv(val_data, "train_test/val_data.csv")
        self.fight_processor._save_csv(test_data, "train_test/test_data.csv")

        removed_path = self.data_dir / "train_test" / "removed_features.txt"
        removed_path.parent.mkdir(parents=True, exist_ok=True)
        removed_path.write_text(",".join(removed_features))

        logger.info(
            "Train size: %s, Validation size: %s, Test size: %s",
            len(train_data),
            len(val_data),
            len(test_data),
        )
        return train_data, val_data, test_data

    def _verify_split(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> None:
        logger.info("LEAKAGE CHECK #5: Train/Val/Test date ranges")
        if not train_data.empty:
            logger.info(
                "Train date range: %s – %s",
                train_data["current_fight_date"].min(),
                train_data["current_fight_date"].max(),
            )
        if not val_data.empty:
            logger.info(
                "Validation date range: %s – %s",
                val_data["current_fight_date"].min(),
                val_data["current_fight_date"].max(),
            )
        if not test_data.empty:
            logger.info(
                "Test date range: %s – %s",
                test_data["current_fight_date"].min(),
                test_data["current_fight_date"].max(),
            )

        if not train_data.empty and not val_data.empty:
            if train_data["current_fight_date"].max() >= val_data["current_fight_date"].min():
                logger.error("❌ LEAKAGE: Train and validation dates overlap")
        if not val_data.empty and not test_data.empty:
            if val_data["current_fight_date"].max() >= test_data["current_fight_date"].min():
                logger.error("❌ LEAKAGE: Validation and test dates overlap")

        if not train_data.empty and not val_data.empty:
            train_dates = set(train_data["current_fight_date"].unique())
            val_dates = set(val_data["current_fight_date"].unique())
            common_dates = sorted(train_dates & val_dates)
            if common_dates:
                logger.warning(
                    "⚠️ %s dates appear in both train and validation sets (e.g. %s)",
                    len(common_dates),
                    common_dates[:5],
                )

    # ------------------------------------------------------------------
    # Fight de-duplication
    # ------------------------------------------------------------------

    def _remove_duplicate_fights(self, df: pd.DataFrame, random: bool = True) -> pd.DataFrame:
        df = df.copy()
        df["fight_pair"] = df.apply(
            lambda row: tuple(sorted([row["fighter_a"], row["fighter_b"]])), axis=1
        )

        if random:
            df = df.sample(frac=1, random_state=42)
            df = df.drop_duplicates(subset=["fight_pair"], keep="first")
        else:
            selected_rows = []
            for _, group in df.groupby("fight_pair"):
                alpha_rows = group[group["fighter_a"] <= group["fighter_b"]]
                if not alpha_rows.empty:
                    selected_rows.append(alpha_rows.iloc[0])
                else:
                    selected_rows.append(group.iloc[0])
            df = pd.DataFrame(selected_rows)
            df = df.sort_values(by=["current_fight_date", "fighter_a"], ascending=[True, True])

        return df.drop(columns=["fight_pair"]).reset_index(drop=True)


# =============================================================================
# Comprehensive Data Integrity Verification
# =============================================================================

def verify_data_integrity(data_dir: PathLike = "../../../data", sample_size: int = 5) -> bool:
    logger.info("Running comprehensive data leakage verification")
    data_dir_path = resolve_data_directory(data_dir, Path(__file__).resolve(), default_subdir="data")

    issues_found: List[str] = []
    try:
        combined_rounds = pd.read_csv(data_dir_path / "processed/combined_rounds.csv")

        test_fighter = combined_rounds["fighter"].value_counts().index[0]
        fighter_data = combined_rounds[combined_rounds["fighter"] == test_fighter].sort_values("fight_date")
        logger.info("Fighter %s has %s fights", test_fighter, len(fighter_data))

        for idx in range(min(3, len(fighter_data))):
            fight = fighter_data.iloc[idx]
            expected_total = idx + 1
            if fight.get("total_fights", expected_total) != expected_total:
                issues_found.append(f"Fighter {test_fighter}: total_fights mismatch in fight {idx + 1}")

        date_issues = 0
        for fighter, group in combined_rounds.groupby("fighter"):
            dates = pd.to_datetime(group["fight_date"]).sort_values()
            if not dates.is_monotonic_increasing:
                date_issues += 1
                if date_issues <= 3:
                    issues_found.append(f"Fighter {fighter}: unordered dates")
        if date_issues:
            logger.warning("Detected %s fighters with unordered dates", date_issues)

        train_data = pd.read_csv(data_dir_path / "train_test/train_data.csv")
        val_data = pd.read_csv(data_dir_path / "train_test/val_data.csv")
        test_data = pd.read_csv(data_dir_path / "train_test/test_data.csv")

        train_end = train_data["current_fight_date"].max()
        val_start = val_data["current_fight_date"].min()
        val_end = val_data["current_fight_date"].max()
        test_start = test_data["current_fight_date"].min()

        if pd.to_datetime(train_end) >= pd.to_datetime(val_start):
            issues_found.append("Train/validation date overlap")
        if pd.to_datetime(val_end) >= pd.to_datetime(test_start):
            issues_found.append("Validation/test date overlap")

        for idx in range(min(sample_size, len(test_data))):
            row = test_data.iloc[idx]
            if "total_fights" in row and pd.isna(row.get("total_fights")):
                issues_found.append(f"Sample {idx + 1}: missing total_fights")

    except FileNotFoundError as exc:
        issues_found.append(f"Missing file: {exc}")
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("Unexpected error during verification")
        issues_found.append(f"Unexpected error: {exc}")

    if not issues_found:
        logger.info("✅ Data integrity checks passed")
    else:
        logger.error("❌ Data integrity issues detected (%s)", len(issues_found))
        for issue in issues_found[:10]:
            logger.error(" - %s", issue)
        if len(issues_found) > 10:
            logger.error("   ... and %s more issues", len(issues_found) - 10)

    return not issues_found


# =============================================================================
# Main Execution
# =============================================================================

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    fight_processor = FightDataProcessor(enable_verification=True)
    matchup_processor = MatchupProcessor(data_dir=str(fight_processor.data_dir), enable_verification=True)

    logger.info("Starting UFC data processing pipeline with leakage verification")

    fight_processor.combine_rounds_stats("processed/ufc_fight_processed.csv")
    combined_rounds_path = fight_processor.data_dir / "processed" / "combined_rounds.csv"
    calculate_elo_ratings(str(combined_rounds_path))
    fight_processor.combine_fighters_stats("processed/combined_rounds.csv")

    matchup_processor.create_matchup_data("processed/combined_sorted_fighter_stats.csv", tester=3, include_names=True)
    matchup_processor.split_train_val_test(
        "matchup data/matchup_data_3_avg_name.csv",
        "2025-01-01",
        "2025-12-31",
        10,
    )

    integrity_passed = verify_data_integrity(fight_processor.data_dir)
    if integrity_passed:
        logger.info("✅ Data processing completed successfully with no leakage detected")
    else:
        logger.warning("⚠️ Potential leakage issues detected – review the log output")


if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    logger.info("Total runtime: %s", end_time - start_time)

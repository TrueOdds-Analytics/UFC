"""Utility helpers for UFC data processing pipelines.

This module houses general-purpose data utilities that are shared across the
cleaning and feature engineering steps.  The goal is to keep the helpers small
and composable so downstream code remains easy to follow and test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


PathLike = Union[str, Path]


def resolve_data_directory(
    data_dir: Optional[PathLike],
    module_file: Path,
    default_subdir: str = "data",
) -> Path:
    """Resolve ``data_dir`` to an absolute :class:`Path`.

    The original project mixed several different working-directory assumptions
    which made the pipeline brittle.  This helper normalises the behaviour by
    supporting three resolution strategies, in priority order:

    1. Absolute paths are returned unchanged.
    2. Relative paths are resolved against the module's directory (allowing the
       processors to work when invoked as scripts).
    3. Relative paths are resolved against the repository root (supporting the
       previous ``../../../data`` style).

    Parameters
    ----------
    data_dir:
        Requested data directory, if any.
    module_file:
        ``__file__`` path of the module instantiating the processor.
    default_subdir:
        Fallback directory name relative to the repo root when ``data_dir`` is
        ``None``.

    Returns
    -------
    :class:`Path`
        An absolute path that may not yet exist but is safe to create.
    """

    module_path = Path(module_file).resolve()
    repo_root = module_path.parents[2]

    if data_dir is None:
        candidate = repo_root / default_subdir
    else:
        candidate = Path(data_dir).expanduser()
        if not candidate.is_absolute():
            module_relative = (module_path.parent / candidate).resolve(strict=False)
            repo_relative = (repo_root / candidate).resolve(strict=False)
            if module_relative.exists():
                candidate = module_relative
            elif repo_relative.exists():
                candidate = repo_relative
            else:
                # ``resolve`` above already normalised the path, so even if the
                # directory does not currently exist we still return an absolute
                # location to keep downstream path handling simple.
                candidate = module_relative

    return candidate


class DataUtils:
    """General-purpose helpers for pandas and numpy operations."""

    @staticmethod
    def safe_divide(
        numerator: Union[float, np.ndarray, pd.Series, pd.DataFrame],
        denominator: Union[float, np.ndarray, pd.Series, pd.DataFrame],
        *,
        default: float = 0.0,
    ) -> Union[float, np.ndarray, pd.Series]:
        """Safely divide two values while guarding against division by zero.

        ``pandas`` changed its handling of division in recent releases which can
        easily surface ``ZeroDivisionError`` or ``inf`` values when the input
        arrays contain zeros.  This implementation standardises the behaviour by
        returning ``default`` for undefined results regardless of the input
        container type.
        """

        if isinstance(numerator, (pd.Series, pd.DataFrame)) or isinstance(
            denominator, (pd.Series, pd.DataFrame)
        ):
            result = pd.Series(numerator).div(pd.Series(denominator))
            cleaned = result.replace([np.inf, -np.inf], np.nan).fillna(default)
            return cleaned

        if isinstance(numerator, np.ndarray) or isinstance(denominator, np.ndarray):
            numerator_arr = np.asarray(numerator, dtype=float)
            denominator_arr = np.asarray(denominator, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                result = np.divide(
                    numerator_arr,
                    denominator_arr,
                    out=np.full_like(numerator_arr, default, dtype=float),
                    where=denominator_arr != 0,
                )
            result[np.isnan(result)] = default
            return result

        # Scalars fall back to Python's division semantics with an explicit check
        return default if denominator == 0 else numerator / denominator

    def preprocess_data(self, ufc_stats: pd.DataFrame, fighter_stats: pd.DataFrame) -> pd.DataFrame:
        """Clean and enrich round-level statistics with fighter metadata."""

        processed = ufc_stats.copy()
        processed["fighter"] = processed["fighter"].astype(str).str.lower()
        processed["fight_date"] = pd.to_datetime(processed["fight_date"], errors="coerce")

        fighters = fighter_stats.copy()
        fighters["name"] = fighters["FIGHTER"].astype(str).str.lower().str.strip()
        fighters["dob"] = (
            fighters["DOB"].replace(["--", "", "NA", "N/A"], np.nan).apply(DateUtils.parse_date)
        )

        processed = processed.merge(
            fighters[["name", "dob"]],
            left_on="fighter",
            right_on="name",
            how="left",
        )

        age_days = (processed["fight_date"] - processed["dob"]).dt.days
        processed["age"] = (age_days / 365.25).round().astype(float)
        processed.loc[processed["age"] < 0, "age"] = np.nan

        processed.drop(columns=["round", "location", "name"], inplace=True, errors="ignore")
        processed = processed[~processed["weight_class"].str.contains("Women's", na=False)]

        time_delta = pd.to_timedelta(processed["time"].fillna("00:00"))
        processed["time"] = time_delta.dt.total_seconds()

        return processed

    @staticmethod
    def rename_columns_general(column: str) -> str:
        """Apply small readability fixes for matchup column names."""

        if "fighter" in column and not column.startswith("fighter"):
            if "b_fighter_b" in column:
                return column.replace("b_fighter_b", "fighter_b_opponent")
            if "b_fighter" in column:
                return column.replace("b_fighter", "fighter_a_opponent")
            if "fighter" in column and "fighter_b" not in column:
                return column.replace("fighter", "fighter_a")
        return column

    @staticmethod
    def get_opponent(fighter: str, fight_id: str, ufc_stats: pd.DataFrame) -> Optional[str]:
        """Return a fighter's opponent for a given fight, if known."""

        fight_fighters = ufc_stats.loc[ufc_stats["id"] == fight_id, "fighter"].unique()
        if len(fight_fighters) < 2:
            return None
        return fight_fighters[0] if fight_fighters[0] != fighter else fight_fighters[1]

    def remove_correlated_features(
        self,
        df: pd.DataFrame,
        *,
        correlation_threshold: float = 0.95,
        protected_columns: Optional[Sequence[str]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Drop highly correlated numeric features."""

        protected = set(protected_columns or [])
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_df = df[numeric_columns]

        if numeric_df.empty:
            return df, []

        corr_matrix = numeric_df.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        columns_to_drop = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > correlation_threshold) and column not in protected
        ]

        cleaned_df = df.drop(columns=columns_to_drop, errors="ignore")
        return cleaned_df, columns_to_drop


class DateUtils:
    """Convenience helpers for parsing inconsistent date strings."""

    @staticmethod
    def parse_date(date_str: Any) -> pd.Timestamp:
        """Parse a date in one of several known formats.

        ``pandas.to_datetime`` is deliberately invoked with ``errors='coerce'``
        after checking the two observed formats.  This gives us predictable
        ``NaT`` results for malformed values instead of raising ``ValueError``.
        """

        if pd.isna(date_str):
            return pd.NaT

        for fmt in ("%d-%b-%y", "%b %d, %Y"):
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                continue

        return pd.to_datetime(date_str, errors="coerce")


@dataclass(frozen=True)
class OddsPair:
    """Simple structure for representing a pair of betting odds."""

    fighter_a: float
    fighter_b: float

    @property
    def diff(self) -> float:
        return self.fighter_a - self.fighter_b

    @property
    def ratio(self) -> float:
        return DataUtils.safe_divide(self.fighter_a, self.fighter_b)  # type: ignore[arg-type]


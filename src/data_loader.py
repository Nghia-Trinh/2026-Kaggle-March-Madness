"""
Data loader for the 2026 Kaggle March Machine Learning Mania competition.

Loads and validates CSV files provided by the competition, returning
clean DataFrames ready for feature engineering and modelling.
"""

import os
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# File names used in the Kaggle data bundle
# ---------------------------------------------------------------------------
_REQUIRED_FILES = {
    "teams": "MTeams.csv",
    "seasons": "MSeasons.csv",
    "regular_season": "MRegularSeasonCompactResults.csv",
    "tourney_results": "MNCAATourneyCompactResults.csv",
    "tourney_seeds": "MNCAATourneySeeds.csv",
    "tourney_slots": "MNCAATourneySlots.csv",
    "sample_submission": "MSampleSubmission.csv",
}

_OPTIONAL_FILES = {
    "regular_season_detailed": "MRegularSeasonDetailedResults.csv",
    "tourney_detailed": "MNCAATourneyDetailedResults.csv",
    "massey_ordinals": "MMasseyOrdinals.csv",
}


def load_data(data_dir: str | os.PathLike) -> dict[str, pd.DataFrame]:
    """Load all available competition CSV files from *data_dir*.

    Parameters
    ----------
    data_dir:
        Directory containing the Kaggle data files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of logical name → DataFrame for every file found.
        Required files raise ``FileNotFoundError`` when missing; optional
        files are silently skipped.
    """
    data_dir = Path(data_dir)
    result: dict[str, pd.DataFrame] = {}

    for key, filename in _REQUIRED_FILES.items():
        filepath = data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(
                f"Required data file not found: {filepath}"
            )
        result[key] = pd.read_csv(filepath)

    for key, filename in _OPTIONAL_FILES.items():
        filepath = data_dir / filename
        if filepath.exists():
            result[key] = pd.read_csv(filepath)

    return result


def parse_submission_ids(submission_df: pd.DataFrame) -> pd.DataFrame:
    """Parse the ``ID`` column of a submission DataFrame.

    The Kaggle submission format uses IDs of the form
    ``<Season>_<Team1ID>_<Team2ID>`` (Team1ID < Team2ID).

    Parameters
    ----------
    submission_df:
        DataFrame that has at least an ``ID`` column.

    Returns
    -------
    pd.DataFrame
        Original columns plus ``Season``, ``Team1ID``, ``Team2ID``
        (all integer).
    """
    df = submission_df.copy()
    parts = df["ID"].str.split("_", expand=True)
    df["Season"] = parts[0].astype(int)
    df["Team1ID"] = parts[1].astype(int)
    df["Team2ID"] = parts[2].astype(int)
    return df


def get_season_games(
    regular_season: pd.DataFrame,
    tourney_results: pd.DataFrame,
    season: int,
    include_tourney: bool = True,
) -> pd.DataFrame:
    """Return all games for a specific *season*.

    Parameters
    ----------
    regular_season:
        MRegularSeasonCompactResults DataFrame.
    tourney_results:
        MNCAATourneyCompactResults DataFrame.
    season:
        The season year to filter on.
    include_tourney:
        If ``True``, tournament games from *prior* seasons are included
        alongside regular-season games.

    Returns
    -------
    pd.DataFrame
        Columns: ``Season``, ``WTeamID``, ``LTeamID``, ``WScore``,
        ``LScore``, ``WLoc``.
    """
    cols = ["Season", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"]

    reg = regular_season[regular_season["Season"] <= season][cols].copy()

    if include_tourney:
        t = tourney_results[tourney_results["Season"] < season][cols].copy()
        return pd.concat([reg, t], ignore_index=True)

    return reg

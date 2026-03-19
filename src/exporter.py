"""
CSV export utilities for the 2026 Kaggle March Machine Learning Mania model.

Provides :func:`export_predictions` which combines raw win-probability
predictions with team metadata (names, seeds, Elo ratings, season stats)
and writes a human-readable results CSV file.
"""

import os
from pathlib import Path

import pandas as pd


# Column order in the detailed output file
_DETAIL_COLUMNS = [
    "Season",
    "Team1ID",
    "Team1Name",
    "Team1Seed",
    "Team1Region",
    "Team1Elo",
    "Team2ID",
    "Team2Name",
    "Team2Seed",
    "Team2Region",
    "Team2Elo",
    "EloDiff",
    "Team1WinProbability",
    "Team2WinProbability",
]


def build_results_df(
    predictions: pd.DataFrame,
    season: int,
    teams: pd.DataFrame,
    seed_features: pd.DataFrame,
    elo_ratings: dict[int, float],
    initial_elo: float = 1500.0,
) -> pd.DataFrame:
    """Build an enriched results DataFrame from raw win-probability predictions.

    Parameters
    ----------
    predictions:
        DataFrame with columns ``Team1ID``, ``Team2ID``,
        ``Team1WinProbability`` *or* ``WinProbability``.
        ``Team1ID < Team2ID`` is assumed (Kaggle convention).
    season:
        The competition season year (used to filter *seed_features*).
    teams:
        MTeams DataFrame with at least ``TeamID`` and ``TeamName``.
    seed_features:
        Output of :func:`src.feature_engineering.build_seed_features`
        with columns ``Season``, ``TeamID``, ``Seed`` (int), ``Region``.
    elo_ratings:
        Mapping ``team_id → float`` for the target season (output of
        :func:`src.ratings.get_pregame_ratings`).
    initial_elo:
        Fallback Elo for teams not present in *elo_ratings*.

    Returns
    -------
    pd.DataFrame
        One row per matchup with the columns listed in
        :data:`_DETAIL_COLUMNS`.  Rows are sorted by ``Team1Seed``,
        ``Team2Seed`` when seed data are available.
    """
    df = predictions.copy()

    # Normalise the probability column name
    if "WinProbability" in df.columns and "Team1WinProbability" not in df.columns:
        df = df.rename(columns={"WinProbability": "Team1WinProbability"})

    df["Season"] = season
    df["Team2WinProbability"] = 1.0 - df["Team1WinProbability"]

    # ----- team names -----
    name_map: dict[int, str] = {}
    if "TeamName" in teams.columns:
        name_map = dict(zip(teams["TeamID"].astype(int), teams["TeamName"].astype(str)))

    df["Team1Name"] = df["Team1ID"].map(name_map).fillna("Unknown")
    df["Team2Name"] = df["Team2ID"].map(name_map).fillna("Unknown")

    # ----- seed + region -----
    season_seeds = seed_features[seed_features["Season"] == season]
    seed_map: dict[int, int] = {}
    region_map: dict[int, str] = {}
    if not season_seeds.empty:
        seed_map = dict(zip(season_seeds["TeamID"].astype(int), season_seeds["Seed"].astype(int)))
        region_map = dict(zip(season_seeds["TeamID"].astype(int), season_seeds["Region"].astype(str)))

    df["Team1Seed"] = df["Team1ID"].map(seed_map)
    df["Team2Seed"] = df["Team2ID"].map(seed_map)
    df["Team1Region"] = df["Team1ID"].map(region_map).fillna("")
    df["Team2Region"] = df["Team2ID"].map(region_map).fillna("")

    # ----- Elo ratings -----
    df["Team1Elo"] = df["Team1ID"].map(elo_ratings).fillna(initial_elo).round(2)
    df["Team2Elo"] = df["Team2ID"].map(elo_ratings).fillna(initial_elo).round(2)
    df["EloDiff"] = (df["Team1Elo"] - df["Team2Elo"]).round(2)

    # ----- round probabilities -----
    df["Team1WinProbability"] = df["Team1WinProbability"].round(6)
    df["Team2WinProbability"] = df["Team2WinProbability"].round(6)

    # ----- sort by seed when available -----
    if not season_seeds.empty:
        df = df.sort_values(
            ["Team1Seed", "Team2Seed"],
            na_position="last",
        ).reset_index(drop=True)

    # Ensure all expected columns exist (fill missing ones with None)
    for col in _DETAIL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[_DETAIL_COLUMNS]


def export_predictions(
    predictions: pd.DataFrame,
    output_path: str | os.PathLike,
    season: int,
    teams: pd.DataFrame,
    seed_features: pd.DataFrame,
    elo_ratings: dict[int, float],
    initial_elo: float = 1500.0,
) -> pd.DataFrame:
    """Build an enriched results DataFrame and write it to *output_path*.

    This is a thin wrapper around :func:`build_results_df` that also
    handles directory creation and file writing.

    Parameters
    ----------
    predictions:
        DataFrame with columns ``Team1ID``, ``Team2ID``,
        ``Team1WinProbability`` *or* ``WinProbability``.
    output_path:
        Destination CSV path.  Parent directories are created if needed.
    season:
        Competition season year.
    teams:
        MTeams DataFrame.
    seed_features:
        Output of :func:`src.feature_engineering.build_seed_features`.
    elo_ratings:
        Mapping ``team_id → float`` for the target season.
    initial_elo:
        Fallback Elo for unknown teams.

    Returns
    -------
    pd.DataFrame
        The enriched results DataFrame (also written to *output_path*).
    """
    output_path = Path(output_path)
    results_df = build_results_df(
        predictions=predictions,
        season=season,
        teams=teams,
        seed_features=seed_features,
        elo_ratings=elo_ratings,
        initial_elo=initial_elo,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    return results_df

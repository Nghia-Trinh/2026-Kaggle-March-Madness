"""
Feature engineering for the 2026 Kaggle March Machine Learning Mania model.

Transforms raw game results and seed data into per-team features used
by the rating system and Monte Carlo simulation.
"""

import numpy as np
import pandas as pd


def seed_to_int(seed_str: str) -> int:
    """Convert a seed string like ``'W01'`` or ``'Z16a'`` to an integer.

    Parameters
    ----------
    seed_str:
        Raw seed string from ``MNCAATourneySeeds.csv``.

    Returns
    -------
    int
        Numeric seed value (1–16).
    """
    return int(seed_str[1:3])


def build_seed_features(tourney_seeds: pd.DataFrame) -> pd.DataFrame:
    """Extract a numeric seed and region for each team per season.

    Parameters
    ----------
    tourney_seeds:
        MNCAATourneySeeds DataFrame with columns ``Season``, ``Seed``,
        ``TeamID``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Season``, ``TeamID``, ``Seed`` (int), ``Region``
        (single character).
    """
    df = tourney_seeds.copy()
    df["SeedInt"] = df["Seed"].apply(seed_to_int)
    df["Region"] = df["Seed"].str[0]
    return df[["Season", "TeamID", "SeedInt", "Region"]].rename(
        columns={"SeedInt": "Seed"}
    )


def compute_season_stats(games: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team aggregated stats from compact game results.

    Parameters
    ----------
    games:
        DataFrame with columns ``Season``, ``WTeamID``, ``LTeamID``,
        ``WScore``, ``LScore``.

    Returns
    -------
    pd.DataFrame
        Index: ``(Season, TeamID)`` with columns:
        ``Games``, ``Wins``, ``Losses``, ``WinPct``, ``AvgPointDiff``,
        ``AvgPointsFor``, ``AvgPointsAgainst``.
    """
    wins = (
        games.groupby(["Season", "WTeamID"])
        .agg(
            Wins=("WScore", "count"),
            PointsFor=("WScore", "sum"),
            PointsAgainst=("LScore", "sum"),
        )
        .rename_axis(["Season", "TeamID"])
    )

    losses = (
        games.groupby(["Season", "LTeamID"])
        .agg(
            Losses=("LScore", "count"),
            PointsFor=("LScore", "sum"),
            PointsAgainst=("WScore", "sum"),
        )
        .rename_axis(["Season", "TeamID"])
    )

    stats = wins.add(losses, fill_value=0)
    stats["Wins"] = stats["Wins"].fillna(0)
    stats["Losses"] = stats["Losses"].fillna(0)
    stats["Games"] = stats["Wins"] + stats["Losses"]
    stats["WinPct"] = stats["Wins"] / stats["Games"].replace(0, np.nan)
    stats["AvgPointsFor"] = stats["PointsFor"] / stats["Games"].replace(
        0, np.nan
    )
    stats["AvgPointsAgainst"] = stats["PointsAgainst"] / stats[
        "Games"
    ].replace(0, np.nan)
    stats["AvgPointDiff"] = stats["AvgPointsFor"] - stats["AvgPointsAgainst"]

    return stats[
        [
            "Games",
            "Wins",
            "Losses",
            "WinPct",
            "AvgPointDiff",
            "AvgPointsFor",
            "AvgPointsAgainst",
        ]
    ].reset_index()


def build_matchup_features(
    team1_id: int,
    team2_id: int,
    season: int,
    elo_ratings: dict[tuple[int, int], float],
    season_stats: pd.DataFrame,
    seed_features: pd.DataFrame,
) -> dict:
    """Build a feature dictionary for a single team1 vs. team2 matchup.

    Parameters
    ----------
    team1_id:
        Lower team ID (as in the Kaggle submission format).
    team2_id:
        Higher team ID.
    season:
        Competition season year.
    elo_ratings:
        Mapping of ``(season, team_id)`` → Elo rating.
    season_stats:
        Output of :func:`compute_season_stats`.
    seed_features:
        Output of :func:`build_seed_features`.

    Returns
    -------
    dict
        Feature values with keys prefixed by ``t1_`` and ``t2_`` for
        each team, plus difference features (``delta_*``).
    """
    features: dict = {}

    elo1 = elo_ratings.get((season, team1_id), 1500.0)
    elo2 = elo_ratings.get((season, team2_id), 1500.0)
    features["t1_elo"] = elo1
    features["t2_elo"] = elo2
    features["delta_elo"] = elo1 - elo2

    stats = season_stats[season_stats["Season"] == season]
    for tid, prefix in [(team1_id, "t1"), (team2_id, "t2")]:
        row = stats[stats["TeamID"] == tid]
        if not row.empty:
            features[f"{prefix}_win_pct"] = float(row["WinPct"].iloc[0])
            features[f"{prefix}_avg_diff"] = float(
                row["AvgPointDiff"].iloc[0]
            )
        else:
            features[f"{prefix}_win_pct"] = 0.5
            features[f"{prefix}_avg_diff"] = 0.0

    features["delta_win_pct"] = (
        features["t1_win_pct"] - features["t2_win_pct"]
    )
    features["delta_avg_diff"] = (
        features["t1_avg_diff"] - features["t2_avg_diff"]
    )

    seeds = seed_features[seed_features["Season"] == season]
    seed1_row = seeds[seeds["TeamID"] == team1_id]
    seed2_row = seeds[seeds["TeamID"] == team2_id]
    seed1 = int(seed1_row["Seed"].iloc[0]) if not seed1_row.empty else 16
    seed2 = int(seed2_row["Seed"].iloc[0]) if not seed2_row.empty else 16
    features["t1_seed"] = seed1
    features["t2_seed"] = seed2
    features["delta_seed"] = seed2 - seed1

    return features

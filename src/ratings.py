"""
Elo-based team rating system for the March Madness model.

Elo ratings are updated after every game using the standard Elo formula.
An optional margin-of-victory multiplier (à la FiveThirtyEight) can be
enabled to make ratings more sensitive to blowout wins.
"""

import math
from typing import Iterable

import pandas as pd


# Default Elo hyper-parameters
_DEFAULT_K = 20.0
_DEFAULT_INITIAL = 1500.0
_DEFAULT_HOME_ADVANTAGE = 100.0  # Elo points for home court
_MOV_MULTIPLIER = True  # apply margin-of-victory multiplier by default


def _expected_score(rating_a: float, rating_b: float) -> float:
    """Return the expected win probability for team A vs. team B."""
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def _mov_multiplier(margin: int, elo_diff: float) -> float:
    """Margin-of-victory multiplier from FiveThirtyEight's NBA Elo model.

    Adapted for college basketball:  ``ln(|margin| + 1) * 2.2 / (elo_diff * 0.001 + 2.2)``.
    """
    return math.log(abs(margin) + 1) * 2.2 / (elo_diff * 0.001 + 2.2)


def compute_elo_ratings(
    games: pd.DataFrame,
    k: float = _DEFAULT_K,
    initial_rating: float = _DEFAULT_INITIAL,
    home_advantage: float = _DEFAULT_HOME_ADVANTAGE,
    use_mov: bool = _MOV_MULTIPLIER,
    season_reset_pct: float = 0.25,
) -> dict[tuple[int, int], float]:
    """Compute Elo ratings for every team in every season.

    Ratings are initialised once and carried forward across seasons,
    with a partial regression to the mean between seasons.

    Parameters
    ----------
    games:
        DataFrame with columns ``Season``, ``WTeamID``, ``LTeamID``,
        ``WScore``, ``LScore``, ``WLoc``.  Must be sorted by ``Season``
        (and ideally by ``DayNum`` if available).
    k:
        K-factor controlling update speed.
    initial_rating:
        Starting Elo for teams with no history.
    home_advantage:
        Elo-point bonus awarded to the home team.
    use_mov:
        Whether to apply a margin-of-victory multiplier.
    season_reset_pct:
        Fraction of the gap between current and initial rating to
        regress to mean between seasons (0 = no regression, 1 = full
        reset).

    Returns
    -------
    dict[tuple[int, int], float]
        ``(season, team_id)`` → Elo rating at the **end** of that
        season (including any tournament games in the same season).
    """
    ratings: dict[int, float] = {}  # current rating per team_id
    history: dict[tuple[int, int], float] = {}  # (season, team_id) → end-of-season rating
    current_season: int | None = None

    for _, row in games.iterrows():
        season = int(row["Season"])

        # Season transition: store end-of-last-season ratings and apply
        # partial mean regression before the new season starts.
        if season != current_season:
            if current_season is not None:
                # Store ratings at end of previous season
                for tid, r in ratings.items():
                    history[(current_season, tid)] = r
                # Partial regression to the mean
                ratings = {
                    tid: r + season_reset_pct * (initial_rating - r)
                    for tid, r in ratings.items()
                }
            current_season = season

        w_id = int(row["WTeamID"])
        l_id = int(row["LTeamID"])
        w_score = int(row["WScore"])
        l_score = int(row["LScore"])
        loc = str(row.get("WLoc", "N"))

        # Initialise ratings on first encounter
        if w_id not in ratings:
            ratings[w_id] = initial_rating
        if l_id not in ratings:
            ratings[l_id] = initial_rating

        r_w = ratings[w_id]
        r_l = ratings[l_id]

        # Apply home-court advantage
        if loc == "H":
            r_w_adj = r_w + home_advantage
        elif loc == "A":
            r_w_adj = r_w - home_advantage
        else:
            r_w_adj = r_w

        expected_w = _expected_score(r_w_adj, r_l)
        expected_l = 1.0 - expected_w

        margin = w_score - l_score
        mov = _mov_multiplier(margin, r_w_adj - r_l) if use_mov else 1.0

        update_w = k * mov * (1.0 - expected_w)
        update_l = k * mov * (0.0 - expected_l)

        ratings[w_id] = r_w + update_w
        ratings[l_id] = r_l + update_l

    # Store ratings for the final season in the data
    if current_season is not None:
        for tid, r in ratings.items():
            history[(current_season, tid)] = r

    return history


def get_pregame_ratings(
    elo_history: dict[tuple[int, int], float],
    season: int,
    initial_rating: float = _DEFAULT_INITIAL,
) -> dict[int, float]:
    """Return ratings to use at the *start* of tournament play for *season*.

    The returned ratings are the end-of-regular-season values (i.e.
    those recorded just before any tournament games of that season are
    played).  In practice this is the most recent entry in
    *elo_history* for each team at or before *season*.

    Parameters
    ----------
    elo_history:
        Output of :func:`compute_elo_ratings`.
    season:
        The target season.
    initial_rating:
        Fallback for teams with no recorded history.

    Returns
    -------
    dict[int, float]
        ``team_id`` → Elo rating.
    """
    result: dict[int, float] = {}
    for (s, tid), r in elo_history.items():
        if s == season:
            result[tid] = r
    return result

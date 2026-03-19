"""
Monte Carlo simulation engine for the 2026 Kaggle March Madness competition.

The simulator estimates win probabilities for every possible matchup by
running *n_simulations* independent tournament brackets and counting
how often each pairing occurs and who wins.  A calibrated logistic
model converts team-strength differences into win probabilities for
each individual game.
"""

import math
from typing import Callable

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid / logistic function


# ---------------------------------------------------------------------------
# Win-probability models
# ---------------------------------------------------------------------------

def elo_win_probability(elo_a: float, elo_b: float) -> float:
    """Return the probability that team A beats team B given their Elo ratings.

    Uses the standard Elo formula with a 400-point scale.

    Parameters
    ----------
    elo_a:
        Elo rating of team A.
    elo_b:
        Elo rating of team B.

    Returns
    -------
    float
        Probability in [0, 1] that team A wins.
    """
    return 1.0 / (1.0 + math.pow(10.0, (elo_b - elo_a) / 400.0))


def logistic_win_probability(
    delta: float,
    scale: float = 30.0,
) -> float:
    """Return a win probability via a logistic function of a strength delta.

    Parameters
    ----------
    delta:
        Strength difference (team A minus team B).  May be Elo points,
        point differential, or any composite score.
    scale:
        Logistic scale parameter.  Larger values flatten the curve
        (smaller differences map to probabilities closer to 0.5).

    Returns
    -------
    float
        Probability in [0, 1] that team A wins.
    """
    return float(expit(delta / scale))


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------

class MonteCarloBracketSimulator:
    """Simulate full March Madness brackets using Monte Carlo sampling.

    Parameters
    ----------
    teams:
        List of team IDs participating in the tournament.
    win_prob_fn:
        Callable ``(team_a_id, team_b_id) → float`` returning the
        probability that *team_a* beats *team_b*.  This function is
        called for every individual game in every simulated bracket.
    n_simulations:
        Number of full brackets to simulate.
    random_seed:
        Optional seed for reproducibility.
    """

    def __init__(
        self,
        teams: list[int],
        win_prob_fn: Callable[[int, int], float],
        n_simulations: int = 10_000,
        random_seed: int | None = 42,
    ) -> None:
        self.teams = list(teams)
        self.win_prob_fn = win_prob_fn
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(random_seed)

    def simulate_game(self, team_a: int, team_b: int) -> int:
        """Simulate a single game and return the winner's ID."""
        p = self.win_prob_fn(team_a, team_b)
        return team_a if self.rng.random() < p else team_b

    def simulate_bracket(self, seeded_teams: list[int]) -> list[tuple[int, int]]:
        """Simulate one full bracket starting from *seeded_teams*.

        Teams are matched in order: index 0 vs. 1, 2 vs. 3, etc.,
        recursively until a champion is determined.

        Parameters
        ----------
        seeded_teams:
            Ordered list of team IDs; the bracket structure follows a
            power-of-two single-elimination format.  If the list length
            is not a power of two the last teams receive byes.

        Returns
        -------
        list[tuple[int, int]]
            All ``(winner, loser)`` game results from this bracket.
        """
        results: list[tuple[int, int]] = []
        current_round = list(seeded_teams)

        while len(current_round) > 1:
            next_round: list[int] = []
            # Pad to even length: the last team receives a bye (advances automatically)
            bye_team: int | None = None
            if len(current_round) % 2 == 1:
                bye_team = current_round[-1]
                current_round = current_round[:-1]

            for i in range(0, len(current_round), 2):
                a = current_round[i]
                b = current_round[i + 1]
                winner = self.simulate_game(a, b)
                loser = b if winner == a else a
                results.append((winner, loser))
                next_round.append(winner)

            if bye_team is not None:
                next_round.append(bye_team)

            current_round = next_round

        return results

    def run(self, seeded_teams: list[int]) -> pd.DataFrame:
        """Run *n_simulations* brackets and aggregate win counts.

        Parameters
        ----------
        seeded_teams:
            Ordered list of team IDs in seeding order (best seed first).

        Returns
        -------
        pd.DataFrame
            Columns: ``Team1ID``, ``Team2ID``, ``WinCount``,
            ``MatchupCount``, ``WinProbability``.
            One row per unique ordered pair where ``Team1ID < Team2ID``.
        """
        win_counts: dict[tuple[int, int], int] = {}
        matchup_counts: dict[tuple[int, int], int] = {}

        for _ in range(self.n_simulations):
            game_results = self.simulate_bracket(seeded_teams)
            for winner, loser in game_results:
                key = (min(winner, loser), max(winner, loser))
                matchup_counts[key] = matchup_counts.get(key, 0) + 1
                win_key = (winner, loser)
                normalized_key = (min(winner, loser), max(winner, loser))
                # Track wins for the lower-ID team (Team1 in submission format).
                # If the higher-ID team won, Team1 (lower ID) did not win; no increment.
                if winner < loser:
                    win_counts[normalized_key] = (
                        win_counts.get(normalized_key, 0) + 1
                    )
                else:
                    win_counts[normalized_key] = win_counts.get(
                        normalized_key, 0
                    )

        rows = []
        for key, total in matchup_counts.items():
            t1, t2 = key
            wins_t1 = win_counts.get(key, 0)
            rows.append(
                {
                    "Team1ID": t1,
                    "Team2ID": t2,
                    "WinCount": wins_t1,
                    "MatchupCount": total,
                    "WinProbability": wins_t1 / total if total > 0 else 0.5,
                }
            )

        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Head-to-head probability matrix (no bracket structure)
# ---------------------------------------------------------------------------

def compute_pairwise_probabilities(
    team_ids: list[int],
    win_prob_fn: Callable[[int, int], float],
) -> pd.DataFrame:
    """Compute a symmetric win-probability matrix for all pairs of teams.

    Unlike the bracket simulator this function does **not** simulate
    full tournaments; it simply evaluates *win_prob_fn* for every
    ordered pair once.  Use this when the Kaggle submission requires
    probabilities for all possible matchups rather than only those that
    can occur in the bracket.

    Parameters
    ----------
    team_ids:
        All team IDs to include.
    win_prob_fn:
        Callable ``(team_a_id, team_b_id) → float``.

    Returns
    -------
    pd.DataFrame
        Columns: ``Team1ID``, ``Team2ID``, ``WinProbability``
        (Team1ID < Team2ID always).
    """
    rows = []
    ids = sorted(team_ids)
    for i, t1 in enumerate(ids):
        for t2 in ids[i + 1 :]:
            p = win_prob_fn(t1, t2)
            rows.append({"Team1ID": t1, "Team2ID": t2, "WinProbability": p})
    return pd.DataFrame(rows)


def blend_probabilities(
    mc_df: pd.DataFrame,
    direct_df: pd.DataFrame,
    mc_weight: float = 0.5,
) -> pd.DataFrame:
    """Blend Monte Carlo bracket probabilities with direct pairwise probabilities.

    Teams that never meet in simulated brackets receive probabilities
    only from *direct_df*.  The blend weight controls the contribution
    of Monte Carlo results where both are available.

    Parameters
    ----------
    mc_df:
        Output of :meth:`MonteCarloBracketSimulator.run`.
    direct_df:
        Output of :func:`compute_pairwise_probabilities`.
    mc_weight:
        Weight in [0, 1] for Monte Carlo probabilities (``1 - mc_weight``
        is applied to direct probabilities).

    Returns
    -------
    pd.DataFrame
        Columns: ``Team1ID``, ``Team2ID``, ``WinProbability``.
    """
    mc_map = {
        (int(r["Team1ID"]), int(r["Team2ID"])): float(r["WinProbability"])
        for _, r in mc_df.iterrows()
    }

    rows = []
    for _, row in direct_df.iterrows():
        key = (int(row["Team1ID"]), int(row["Team2ID"]))
        direct_p = float(row["WinProbability"])
        mc_p = mc_map.get(key)
        if mc_p is not None:
            blended = mc_weight * mc_p + (1.0 - mc_weight) * direct_p
        else:
            blended = direct_p
        rows.append(
            {"Team1ID": key[0], "Team2ID": key[1], "WinProbability": blended}
        )
    return pd.DataFrame(rows)

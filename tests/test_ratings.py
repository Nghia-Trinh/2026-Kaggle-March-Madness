"""
Tests for src/ratings.py
"""

import pytest
import pandas as pd

from src.ratings import (
    compute_elo_ratings,
    get_pregame_ratings,
    _DEFAULT_INITIAL,
)


def _make_games(records: list[dict]) -> pd.DataFrame:
    """Helper to build a minimal games DataFrame."""
    return pd.DataFrame(
        records,
        columns=["Season", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"],
    )


class TestComputeEloRatings:
    def test_returns_dict(self):
        games = _make_games(
            [{"Season": 2024, "WTeamID": 1, "LTeamID": 2, "WScore": 80, "LScore": 70, "WLoc": "N"}]
        )
        result = compute_elo_ratings(games)
        assert isinstance(result, dict)

    def test_winner_gains_elo(self):
        games = _make_games(
            [{"Season": 2024, "WTeamID": 1, "LTeamID": 2, "WScore": 80, "LScore": 70, "WLoc": "N"}]
        )
        result = compute_elo_ratings(games)
        assert result[(2024, 1)] > _DEFAULT_INITIAL
        assert result[(2024, 2)] < _DEFAULT_INITIAL

    def test_loser_drops_proportionally(self):
        """The loser's Elo loss should mirror the winner's Elo gain."""
        games = _make_games(
            [{"Season": 2024, "WTeamID": 1, "LTeamID": 2, "WScore": 80, "LScore": 70, "WLoc": "N"}]
        )
        result = compute_elo_ratings(games)
        winner_gain = result[(2024, 1)] - _DEFAULT_INITIAL
        loser_loss = _DEFAULT_INITIAL - result[(2024, 2)]
        assert winner_gain == pytest.approx(loser_loss, abs=1.0)

    def test_elo_sum_conserved(self):
        """Total Elo across two teams should remain approximately constant."""
        games = _make_games(
            [{"Season": 2024, "WTeamID": 1, "LTeamID": 2, "WScore": 80, "LScore": 70, "WLoc": "N"}]
        )
        result = compute_elo_ratings(games)
        total = result[(2024, 1)] + result[(2024, 2)]
        assert total == pytest.approx(2 * _DEFAULT_INITIAL, abs=1.0)

    def test_multiple_seasons_keys_present(self):
        games = _make_games(
            [
                {"Season": 2024, "WTeamID": 1, "LTeamID": 2, "WScore": 80, "LScore": 70, "WLoc": "N"},
                {"Season": 2025, "WTeamID": 2, "LTeamID": 1, "WScore": 75, "LScore": 70, "WLoc": "N"},
            ]
        )
        result = compute_elo_ratings(games)
        assert (2024, 1) in result
        assert (2024, 2) in result
        assert (2025, 1) in result
        assert (2025, 2) in result

    def test_home_advantage_applied(self):
        """Home team should have a higher expected score → smaller update when winning."""
        home_game = _make_games(
            [{"Season": 2024, "WTeamID": 1, "LTeamID": 2, "WScore": 80, "LScore": 70, "WLoc": "H"}]
        )
        neutral_game = _make_games(
            [{"Season": 2024, "WTeamID": 1, "LTeamID": 2, "WScore": 80, "LScore": 70, "WLoc": "N"}]
        )
        home_result = compute_elo_ratings(home_game)
        neutral_result = compute_elo_ratings(neutral_game)
        # Home win should gain *less* Elo than neutral win (expected more from home team)
        assert home_result[(2024, 1)] < neutral_result[(2024, 1)]

    def test_consistent_winner_has_higher_rating(self):
        games = _make_games(
            [
                {"Season": 2024, "WTeamID": 1, "LTeamID": 2, "WScore": 80, "LScore": 70, "WLoc": "N"},
                {"Season": 2024, "WTeamID": 1, "LTeamID": 2, "WScore": 80, "LScore": 70, "WLoc": "N"},
                {"Season": 2024, "WTeamID": 1, "LTeamID": 2, "WScore": 80, "LScore": 70, "WLoc": "N"},
            ]
        )
        result = compute_elo_ratings(games)
        assert result[(2024, 1)] > result[(2024, 2)]


class TestGetPregameRatings:
    def test_returns_dict_for_target_season(self):
        elo_history = {(2024, 1): 1520.0, (2024, 2): 1480.0, (2025, 1): 1530.0}
        result = get_pregame_ratings(elo_history, season=2024)
        assert isinstance(result, dict)
        assert 1 in result
        assert 2 in result

    def test_only_target_season_returned(self):
        elo_history = {(2024, 1): 1520.0, (2025, 1): 1530.0}
        result = get_pregame_ratings(elo_history, season=2024)
        assert 1 in result
        # Season 2025 should NOT bleed into result
        # (both seasons have team 1; we only want season 2024)
        assert result[1] == pytest.approx(1520.0)

    def test_empty_when_no_data_for_season(self):
        elo_history = {(2024, 1): 1520.0}
        result = get_pregame_ratings(elo_history, season=2025)
        assert result == {}

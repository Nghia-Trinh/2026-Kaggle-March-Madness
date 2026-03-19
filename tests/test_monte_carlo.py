"""
Tests for src/monte_carlo.py
"""

import pytest
import pandas as pd

from src.monte_carlo import (
    elo_win_probability,
    logistic_win_probability,
    compute_pairwise_probabilities,
    MonteCarloBracketSimulator,
    blend_probabilities,
)


# ---------------------------------------------------------------------------
# elo_win_probability
# ---------------------------------------------------------------------------

class TestEloWinProbability:
    def test_equal_ratings_returns_half(self):
        assert elo_win_probability(1500.0, 1500.0) == pytest.approx(0.5)

    def test_higher_rating_wins_more(self):
        assert elo_win_probability(1600.0, 1500.0) > 0.5

    def test_lower_rating_wins_less(self):
        assert elo_win_probability(1400.0, 1500.0) < 0.5

    def test_symmetry(self):
        p = elo_win_probability(1600.0, 1400.0)
        assert elo_win_probability(1400.0, 1600.0) == pytest.approx(1.0 - p)

    def test_probability_bounds(self):
        for delta in [-800, -400, 0, 400, 800]:
            p = elo_win_probability(1500.0 + delta, 1500.0)
            assert 0.0 < p < 1.0


# ---------------------------------------------------------------------------
# logistic_win_probability
# ---------------------------------------------------------------------------

class TestLogisticWinProbability:
    def test_zero_delta_returns_half(self):
        assert logistic_win_probability(0.0) == pytest.approx(0.5)

    def test_positive_delta_greater_than_half(self):
        assert logistic_win_probability(30.0) > 0.5

    def test_negative_delta_less_than_half(self):
        assert logistic_win_probability(-30.0) < 0.5

    def test_symmetry(self):
        p = logistic_win_probability(60.0)
        assert logistic_win_probability(-60.0) == pytest.approx(1.0 - p, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_pairwise_probabilities
# ---------------------------------------------------------------------------

class TestComputePairwiseProbabilities:
    def _always_half(self, a, b):
        return 0.5

    def test_returns_dataframe(self):
        result = compute_pairwise_probabilities([1, 2, 3], self._always_half)
        assert isinstance(result, pd.DataFrame)

    def test_correct_number_of_pairs(self):
        # C(4, 2) = 6
        result = compute_pairwise_probabilities([1, 2, 3, 4], self._always_half)
        assert len(result) == 6

    def test_team1_always_less_than_team2(self):
        result = compute_pairwise_probabilities([10, 20, 30], self._always_half)
        assert (result["Team1ID"] < result["Team2ID"]).all()

    def test_probabilities_in_range(self):
        def asymmetric(a, b):
            return 0.6 if a < b else 0.4

        result = compute_pairwise_probabilities([1, 2, 3], asymmetric)
        assert (result["WinProbability"] >= 0).all()
        assert (result["WinProbability"] <= 1).all()

    def test_single_team_returns_empty(self):
        result = compute_pairwise_probabilities([1], self._always_half)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# MonteCarloBracketSimulator
# ---------------------------------------------------------------------------

class TestMonteCarloBracketSimulator:
    def _always_first_wins(self, a, b):
        return 1.0 if a < b else 0.0

    def _always_half(self, a, b):
        return 0.5

    def test_simulate_game_deterministic_always_wins(self):
        sim = MonteCarloBracketSimulator(
            teams=[1, 2], win_prob_fn=self._always_first_wins, n_simulations=10
        )
        winner = sim.simulate_game(1, 2)
        assert winner == 1

    def test_simulate_bracket_returns_list_of_tuples(self):
        sim = MonteCarloBracketSimulator(
            teams=[1, 2, 3, 4],
            win_prob_fn=self._always_half,
            n_simulations=10,
        )
        results = sim.simulate_bracket([1, 2, 3, 4])
        assert isinstance(results, list)
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_simulate_bracket_four_teams_three_games(self):
        sim = MonteCarloBracketSimulator(
            teams=[1, 2, 3, 4],
            win_prob_fn=self._always_first_wins,
            n_simulations=1,
        )
        results = sim.simulate_bracket([1, 2, 3, 4])
        # 4 teams → 3 games (semis + final)
        assert len(results) == 3

    def test_run_returns_dataframe_with_required_columns(self):
        sim = MonteCarloBracketSimulator(
            teams=[1, 2, 3, 4],
            win_prob_fn=self._always_half,
            n_simulations=100,
        )
        result = sim.run([1, 2, 3, 4])
        for col in ["Team1ID", "Team2ID", "WinProbability"]:
            assert col in result.columns

    def test_win_probability_in_valid_range(self):
        sim = MonteCarloBracketSimulator(
            teams=[1, 2, 3, 4],
            win_prob_fn=self._always_half,
            n_simulations=500,
        )
        result = sim.run([1, 2, 3, 4])
        assert (result["WinProbability"] >= 0).all()
        assert (result["WinProbability"] <= 1).all()

    def test_deterministic_bracket_always_lower_wins(self):
        """When win prob is 1 for lower-ID teams, they always win."""
        sim = MonteCarloBracketSimulator(
            teams=[1, 2, 3, 4],
            win_prob_fn=self._always_first_wins,
            n_simulations=50,
        )
        result = sim.run([1, 2, 3, 4])
        # Lower ID team should win every matchup it appears in
        # The key (1, 2) should have WinProbability = 1.0 (team 1 always wins)
        row = result[(result["Team1ID"] == 1) & (result["Team2ID"] == 2)]
        if not row.empty:
            assert row["WinProbability"].iloc[0] == pytest.approx(1.0)

    def test_team1_id_always_less_than_team2_id(self):
        sim = MonteCarloBracketSimulator(
            teams=[10, 20, 30, 40],
            win_prob_fn=self._always_half,
            n_simulations=50,
        )
        result = sim.run([10, 20, 30, 40])
        if not result.empty:
            assert (result["Team1ID"] < result["Team2ID"]).all()

    def test_reproducibility_with_same_seed(self):
        teams = list(range(1, 9))

        def prob_fn(a, b):
            return 0.5

        sim1 = MonteCarloBracketSimulator(
            teams=teams, win_prob_fn=prob_fn, n_simulations=200, random_seed=7
        )
        sim2 = MonteCarloBracketSimulator(
            teams=teams, win_prob_fn=prob_fn, n_simulations=200, random_seed=7
        )
        r1 = sim1.run(teams)
        r2 = sim2.run(teams)
        pd.testing.assert_frame_equal(
            r1.sort_values(["Team1ID", "Team2ID"]).reset_index(drop=True),
            r2.sort_values(["Team1ID", "Team2ID"]).reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# blend_probabilities
# ---------------------------------------------------------------------------

class TestBlendProbabilities:
    def _make_direct_df(self):
        return pd.DataFrame(
            {
                "Team1ID": [1, 1, 2],
                "Team2ID": [2, 3, 3],
                "WinProbability": [0.6, 0.7, 0.4],
            }
        )

    def _make_mc_df(self):
        return pd.DataFrame(
            {
                "Team1ID": [1],
                "Team2ID": [2],
                "WinProbability": [0.8],
                "WinCount": [40],
                "MatchupCount": [50],
            }
        )

    def test_returns_dataframe(self):
        result = blend_probabilities(self._make_mc_df(), self._make_direct_df())
        assert isinstance(result, pd.DataFrame)

    def test_blended_value_for_observed_matchup(self):
        mc_df = self._make_mc_df()
        direct_df = self._make_direct_df()
        result = blend_probabilities(mc_df, direct_df, mc_weight=0.5)
        row = result[(result["Team1ID"] == 1) & (result["Team2ID"] == 2)]
        assert not row.empty
        expected = 0.5 * 0.8 + 0.5 * 0.6
        assert row["WinProbability"].iloc[0] == pytest.approx(expected)

    def test_direct_only_for_unobserved_matchup(self):
        mc_df = self._make_mc_df()
        direct_df = self._make_direct_df()
        result = blend_probabilities(mc_df, direct_df, mc_weight=0.5)
        row = result[(result["Team1ID"] == 1) & (result["Team2ID"] == 3)]
        assert not row.empty
        assert row["WinProbability"].iloc[0] == pytest.approx(0.7)

    def test_output_length_matches_direct(self):
        mc_df = self._make_mc_df()
        direct_df = self._make_direct_df()
        result = blend_probabilities(mc_df, direct_df)
        assert len(result) == len(direct_df)

    def test_probabilities_in_zero_one_range(self):
        mc_df = self._make_mc_df()
        direct_df = self._make_direct_df()
        result = blend_probabilities(mc_df, direct_df, mc_weight=0.5)
        assert (result["WinProbability"] >= 0).all()
        assert (result["WinProbability"] <= 1).all()

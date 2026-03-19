"""
Tests for src/feature_engineering.py
"""

import pytest
import pandas as pd
import numpy as np

from src.feature_engineering import (
    seed_to_int,
    build_seed_features,
    compute_season_stats,
    build_matchup_features,
)


class TestSeedToInt:
    def test_standard_seed_01(self):
        assert seed_to_int("W01") == 1

    def test_standard_seed_16(self):
        assert seed_to_int("Z16") == 16

    def test_seed_with_play_in_suffix(self):
        assert seed_to_int("W16a") == 16

    def test_different_regions(self):
        for region in ["W", "X", "Y", "Z"]:
            assert seed_to_int(f"{region}08") == 8


class TestBuildSeedFeatures:
    def _make_seeds(self):
        return pd.DataFrame(
            {
                "Season": [2025, 2025, 2025],
                "Seed": ["W01", "X08", "Y16a"],
                "TeamID": [1101, 1102, 1103],
            }
        )

    def test_returns_dataframe(self):
        result = build_seed_features(self._make_seeds())
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        result = build_seed_features(self._make_seeds())
        for col in ["Season", "TeamID", "Seed", "Region"]:
            assert col in result.columns

    def test_seed_values_are_integers(self):
        result = build_seed_features(self._make_seeds())
        assert result["Seed"].dtype in (int, "int64", "int32")

    def test_seed_values_correct(self):
        result = build_seed_features(self._make_seeds())
        seeds = result.sort_values("TeamID")["Seed"].tolist()
        assert seeds == [1, 8, 16]

    def test_region_extracted(self):
        result = build_seed_features(self._make_seeds())
        regions = dict(zip(result["TeamID"], result["Region"]))
        assert regions[1101] == "W"
        assert regions[1102] == "X"
        assert regions[1103] == "Y"


class TestComputeSeasonStats:
    def _make_games(self):
        return pd.DataFrame(
            {
                "Season": [2025, 2025, 2025],
                "WTeamID": [1, 2, 1],
                "LTeamID": [2, 1, 3],
                "WScore": [80, 75, 90],
                "LScore": [70, 65, 60],
                "WLoc": ["N", "N", "N"],
            }
        )

    def test_returns_dataframe(self):
        result = compute_season_stats(self._make_games())
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self):
        result = compute_season_stats(self._make_games())
        for col in ["Season", "TeamID", "Games", "Wins", "Losses", "WinPct", "AvgPointDiff"]:
            assert col in result.columns

    def test_win_pct_in_range(self):
        result = compute_season_stats(self._make_games())
        assert (result["WinPct"].dropna() >= 0).all()
        assert (result["WinPct"].dropna() <= 1).all()

    def test_team1_has_correct_wins(self):
        result = compute_season_stats(self._make_games())
        team1 = result[(result["Season"] == 2025) & (result["TeamID"] == 1)]
        assert team1["Wins"].iloc[0] == 2
        assert team1["Losses"].iloc[0] == 1

    def test_games_equals_wins_plus_losses(self):
        result = compute_season_stats(self._make_games())
        assert (result["Games"] == result["Wins"] + result["Losses"]).all()


class TestBuildMatchupFeatures:
    def _make_fixtures(self):
        elo_ratings = {
            (2025, 1101): 1600.0,
            (2025, 1102): 1400.0,
        }
        season_stats = pd.DataFrame(
            {
                "Season": [2025, 2025],
                "TeamID": [1101, 1102],
                "WinPct": [0.8, 0.4],
                "AvgPointDiff": [10.0, -5.0],
                "Games": [30, 30],
                "Wins": [24, 12],
                "Losses": [6, 18],
                "AvgPointsFor": [80.0, 70.0],
                "AvgPointsAgainst": [70.0, 75.0],
            }
        )
        seed_features = pd.DataFrame(
            {
                "Season": [2025, 2025],
                "TeamID": [1101, 1102],
                "Seed": [1, 16],
                "Region": ["W", "W"],
            }
        )
        return elo_ratings, season_stats, seed_features

    def test_returns_dict(self):
        elo, stats, seeds = self._make_fixtures()
        result = build_matchup_features(1101, 1102, 2025, elo, stats, seeds)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        elo, stats, seeds = self._make_fixtures()
        result = build_matchup_features(1101, 1102, 2025, elo, stats, seeds)
        for key in ["t1_elo", "t2_elo", "delta_elo", "t1_seed", "t2_seed", "delta_seed"]:
            assert key in result

    def test_elo_delta_computed_correctly(self):
        elo, stats, seeds = self._make_fixtures()
        result = build_matchup_features(1101, 1102, 2025, elo, stats, seeds)
        assert result["delta_elo"] == pytest.approx(200.0)

    def test_seed_delta_positive_for_better_team(self):
        """delta_seed = seed2 - seed1; positive means team1 is better seeded."""
        elo, stats, seeds = self._make_fixtures()
        result = build_matchup_features(1101, 1102, 2025, elo, stats, seeds)
        assert result["delta_seed"] == 15  # 16 - 1

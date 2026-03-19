"""
Tests for src/data_loader.py
"""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    load_data,
    parse_submission_ids,
    get_season_games,
    _REQUIRED_FILES,
)


# ---------------------------------------------------------------------------
# Helpers – synthetic CSV data
# ---------------------------------------------------------------------------

def _write_minimal_data(tmp_dir: Path) -> None:
    """Write the minimum set of CSV files required by load_data."""
    teams = pd.DataFrame({"TeamID": [1101, 1102], "TeamName": ["TeamA", "TeamB"]})
    seasons = pd.DataFrame(
        {"Season": [2024, 2025, 2026], "DayzeroDate": ["11/6/23", "11/4/24", "11/3/25"],
         "RegionW": ["W"] * 3, "RegionX": ["X"] * 3,
         "RegionY": ["Y"] * 3, "RegionZ": ["Z"] * 3}
    )
    reg = pd.DataFrame(
        {
            "Season": [2024, 2024, 2025],
            "DayNum": [10, 11, 10],
            "WTeamID": [1101, 1102, 1101],
            "WScore": [80, 75, 85],
            "LTeamID": [1102, 1101, 1102],
            "LScore": [70, 65, 75],
            "WLoc": ["H", "A", "N"],
            "NumOT": [0, 0, 0],
        }
    )
    tourney = pd.DataFrame(
        {
            "Season": [2024],
            "DayNum": [136],
            "WTeamID": [1101],
            "WScore": [82],
            "LTeamID": [1102],
            "LScore": [78],
            "WLoc": ["N"],
            "NumOT": [0],
        }
    )
    seeds = pd.DataFrame(
        {
            "Season": [2025, 2025],
            "Seed": ["W01", "W16"],
            "TeamID": [1101, 1102],
        }
    )
    slots = pd.DataFrame(
        {"Season": [2025], "Slot": ["W01"], "StrongSeed": ["W01"], "WeakSeed": ["W16"]}
    )
    sample_sub = pd.DataFrame(
        {"ID": ["2026_1101_1102"], "Pred": [0.5]}
    )

    for filename, df in [
        ("MTeams.csv", teams),
        ("MSeasons.csv", seasons),
        ("MRegularSeasonCompactResults.csv", reg),
        ("MNCAATourneyCompactResults.csv", tourney),
        ("MNCAATourneySeeds.csv", seeds),
        ("MNCAATourneySlots.csv", slots),
        ("MSampleSubmission.csv", sample_sub),
    ]:
        df.to_csv(tmp_dir / filename, index=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_data_returns_all_required_keys(tmp_path):
    _write_minimal_data(tmp_path)
    data = load_data(tmp_path)
    for key in _REQUIRED_FILES:
        assert key in data, f"Expected key '{key}' not in loaded data"


def test_load_data_missing_required_file_raises(tmp_path):
    _write_minimal_data(tmp_path)
    (tmp_path / "MTeams.csv").unlink()
    with pytest.raises(FileNotFoundError, match="MTeams.csv"):
        load_data(tmp_path)


def test_load_data_returns_dataframes(tmp_path):
    _write_minimal_data(tmp_path)
    data = load_data(tmp_path)
    for key, df in data.items():
        assert isinstance(df, pd.DataFrame), f"Expected DataFrame for key '{key}'"


def test_parse_submission_ids_columns():
    df = pd.DataFrame({"ID": ["2026_1101_1102", "2026_1103_1104"], "Pred": [0.5, 0.5]})
    result = parse_submission_ids(df)
    assert "Season" in result.columns
    assert "Team1ID" in result.columns
    assert "Team2ID" in result.columns


def test_parse_submission_ids_values():
    df = pd.DataFrame({"ID": ["2026_1101_1102"]})
    result = parse_submission_ids(df)
    assert result["Season"].iloc[0] == 2026
    assert result["Team1ID"].iloc[0] == 1101
    assert result["Team2ID"].iloc[0] == 1102


def test_parse_submission_ids_team1_less_than_team2():
    """Team1ID should always be lower than Team2ID in Kaggle format."""
    df = pd.DataFrame(
        {"ID": ["2026_1101_1102", "2026_1200_1250", "2026_1300_1400"]}
    )
    result = parse_submission_ids(df)
    assert (result["Team1ID"] < result["Team2ID"]).all()


def test_get_season_games_filters_correctly():
    reg = pd.DataFrame(
        {
            "Season": [2024, 2025, 2026],
            "WTeamID": [1, 2, 3],
            "LTeamID": [4, 5, 6],
            "WScore": [80, 75, 85],
            "LScore": [70, 65, 75],
            "WLoc": ["N", "N", "N"],
        }
    )
    tourney = pd.DataFrame(
        {
            "Season": [2024, 2025],
            "WTeamID": [1, 2],
            "LTeamID": [4, 5],
            "WScore": [82, 78],
            "LScore": [78, 72],
            "WLoc": ["N", "N"],
        }
    )
    games = get_season_games(reg, tourney, season=2025)
    # All regular season games up to 2025, plus tourney games from 2024
    assert all(games["Season"] <= 2025)
    # Should include 2024 regular, 2025 regular, 2024 tourney (not 2025 tourney)
    assert len(games) == 3


def test_get_season_games_excludes_tourney_when_flag_false():
    reg = pd.DataFrame(
        {
            "Season": [2024, 2025],
            "WTeamID": [1, 2],
            "LTeamID": [3, 4],
            "WScore": [80, 75],
            "LScore": [70, 65],
            "WLoc": ["N", "N"],
        }
    )
    tourney = pd.DataFrame(
        {
            "Season": [2024],
            "WTeamID": [1],
            "LTeamID": [3],
            "WScore": [82],
            "LScore": [78],
            "WLoc": ["N"],
        }
    )
    games = get_season_games(reg, tourney, season=2025, include_tourney=False)
    assert len(games) == 2

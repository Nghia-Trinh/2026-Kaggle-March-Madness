"""Tests for src/exporter.py."""

import pytest
import pandas as pd

from src.exporter import build_results_df, export_predictions, _DETAIL_COLUMNS


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_predictions(pairs: list[tuple[int, int]], probs: list[float]) -> pd.DataFrame:
    """Build a minimal predictions DataFrame."""
    return pd.DataFrame(
        [
            {"Team1ID": t1, "Team2ID": t2, "Team1WinProbability": p}
            for (t1, t2), p in zip(pairs, probs)
        ]
    )


def _make_teams(*team_ids: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TeamID": list(team_ids),
            "TeamName": [f"Team_{tid}" for tid in team_ids],
        }
    )


def _make_seed_features(season: int, mapping: dict[int, tuple[int, str]]) -> pd.DataFrame:
    """mapping: {team_id: (seed_int, region)}"""
    return pd.DataFrame(
        [
            {"Season": season, "TeamID": tid, "Seed": seed, "Region": region}
            for tid, (seed, region) in mapping.items()
        ]
    )


SEASON = 2026
TEAMS = _make_teams(1101, 1102, 1103, 1104)
SEEDS = _make_seed_features(
    SEASON,
    {1101: (1, "W"), 1102: (16, "W"), 1103: (2, "X"), 1104: (15, "X")},
)
ELO = {1101: 1650.0, 1102: 1350.0, 1103: 1620.0, 1104: 1380.0}
PREDS = _make_predictions(
    [(1101, 1102), (1101, 1103), (1102, 1104)],
    [0.85, 0.55, 0.45],
)


# ---------------------------------------------------------------------------
# build_results_df
# ---------------------------------------------------------------------------

class TestBuildResultsDf:
    def test_returns_dataframe(self):
        result = build_results_df(PREDS, SEASON, TEAMS, SEEDS, ELO)
        assert isinstance(result, pd.DataFrame)

    def test_has_all_required_columns(self):
        result = build_results_df(PREDS, SEASON, TEAMS, SEEDS, ELO)
        for col in _DETAIL_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_row_count_matches_input(self):
        result = build_results_df(PREDS, SEASON, TEAMS, SEEDS, ELO)
        assert len(result) == len(PREDS)

    def test_season_column_filled(self):
        result = build_results_df(PREDS, SEASON, TEAMS, SEEDS, ELO)
        assert (result["Season"] == SEASON).all()

    def test_team_names_populated(self):
        result = build_results_df(PREDS, SEASON, TEAMS, SEEDS, ELO)
        row = result[(result["Team1ID"] == 1101) & (result["Team2ID"] == 1102)].iloc[0]
        assert row["Team1Name"] == "Team_1101"
        assert row["Team2Name"] == "Team_1102"

    def test_seed_values_populated(self):
        result = build_results_df(PREDS, SEASON, TEAMS, SEEDS, ELO)
        row = result[(result["Team1ID"] == 1101) & (result["Team2ID"] == 1102)].iloc[0]
        assert row["Team1Seed"] == 1
        assert row["Team1Region"] == "W"

    def test_elo_values_populated(self):
        result = build_results_df(PREDS, SEASON, TEAMS, SEEDS, ELO)
        row = result[(result["Team1ID"] == 1101) & (result["Team2ID"] == 1102)].iloc[0]
        assert row["Team1Elo"] == pytest.approx(1650.0)
        assert row["Team2Elo"] == pytest.approx(1350.0)

    def test_elo_diff_computed_correctly(self):
        result = build_results_df(PREDS, SEASON, TEAMS, SEEDS, ELO)
        row = result[(result["Team1ID"] == 1101) & (result["Team2ID"] == 1102)].iloc[0]
        assert row["EloDiff"] == pytest.approx(1650.0 - 1350.0)

    def test_team2_win_prob_is_complement(self):
        result = build_results_df(PREDS, SEASON, TEAMS, SEEDS, ELO)
        for _, row in result.iterrows():
            assert row["Team1WinProbability"] + row["Team2WinProbability"] == pytest.approx(1.0, abs=1e-6)

    def test_win_prob_values_preserved(self):
        result = build_results_df(PREDS, SEASON, TEAMS, SEEDS, ELO)
        row = result[result["Team1ID"] == 1101]
        row = row[row["Team2ID"] == 1102].iloc[0]
        assert row["Team1WinProbability"] == pytest.approx(0.85, abs=1e-5)

    def test_fallback_elo_for_unknown_team(self):
        preds = _make_predictions([(1101, 9999)], [0.5])
        teams_extra = pd.concat([TEAMS, pd.DataFrame({"TeamID": [9999], "TeamName": ["Unknown"]})])
        seeds_extra = pd.concat([SEEDS, pd.DataFrame({"Season": [SEASON], "TeamID": [9999], "Seed": [16], "Region": ["Z"]})])
        result = build_results_df(preds, SEASON, teams_extra, seeds_extra, ELO, initial_elo=1500.0)
        row = result.iloc[0]
        assert row["Team2Elo"] == pytest.approx(1500.0)

    def test_win_probability_normalised_when_alias_used(self):
        """Accepts WinProbability as an alias for Team1WinProbability."""
        preds_alias = pd.DataFrame(
            [{"Team1ID": 1101, "Team2ID": 1102, "WinProbability": 0.75}]
        )
        result = build_results_df(preds_alias, SEASON, TEAMS, SEEDS, ELO)
        assert "Team1WinProbability" in result.columns
        assert result["Team1WinProbability"].iloc[0] == pytest.approx(0.75)

    def test_empty_predictions_returns_empty_df(self):
        empty = pd.DataFrame(columns=["Team1ID", "Team2ID", "Team1WinProbability"])
        result = build_results_df(empty, SEASON, TEAMS, SEEDS, ELO)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_missing_seed_data_uses_empty_strings(self):
        empty_seeds = pd.DataFrame(columns=["Season", "TeamID", "Seed", "Region"])
        result = build_results_df(PREDS, SEASON, TEAMS, empty_seeds, ELO)
        assert result["Team1Region"].iloc[0] == ""
        assert result["Team2Region"].iloc[0] == ""


# ---------------------------------------------------------------------------
# export_predictions (file I/O)
# ---------------------------------------------------------------------------

class TestExportPredictions:
    def test_writes_csv_file(self, tmp_path):
        out = tmp_path / "results.csv"
        export_predictions(PREDS, out, SEASON, TEAMS, SEEDS, ELO)
        assert out.exists()

    def test_csv_has_correct_columns(self, tmp_path):
        out = tmp_path / "results.csv"
        export_predictions(PREDS, out, SEASON, TEAMS, SEEDS, ELO)
        df = pd.read_csv(out)
        for col in _DETAIL_COLUMNS:
            assert col in df.columns, f"Missing column in CSV: {col}"

    def test_csv_row_count_matches_input(self, tmp_path):
        out = tmp_path / "results.csv"
        export_predictions(PREDS, out, SEASON, TEAMS, SEEDS, ELO)
        df = pd.read_csv(out)
        assert len(df) == len(PREDS)

    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "results.csv"
        export_predictions(PREDS, out, SEASON, TEAMS, SEEDS, ELO)
        assert out.exists()

    def test_returns_dataframe(self, tmp_path):
        out = tmp_path / "results.csv"
        result = export_predictions(PREDS, out, SEASON, TEAMS, SEEDS, ELO)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(PREDS)

    def test_csv_values_match_dataframe(self, tmp_path):
        out = tmp_path / "results.csv"
        df_returned = export_predictions(PREDS, out, SEASON, TEAMS, SEEDS, ELO)
        df_read = pd.read_csv(out)
        assert len(df_returned) == len(df_read)
        assert list(df_returned.columns) == list(df_read.columns)

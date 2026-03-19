"""
End-to-end prediction pipeline for the 2026 Kaggle March Machine
Learning Mania competition.

Usage
-----
    python run_prediction.py --data-dir /path/to/kaggle/data \
                             --output submission.csv \
                             --season 2026 \
                             --simulations 50000
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_data, parse_submission_ids, get_season_games
from src.feature_engineering import (
    build_seed_features,
    compute_season_stats,
)
from src.ratings import compute_elo_ratings, get_pregame_ratings
from src.monte_carlo import (
    elo_win_probability,
    compute_pairwise_probabilities,
    MonteCarloBracketSimulator,
    blend_probabilities,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Clipping bounds to avoid log-loss / Brier score edge-case penalties.
# Values of 0.025 and 0.975 prevent extreme probabilities while still
# allowing the model to express strong confidence, balancing prediction
# accuracy with robustness against overconfident predictions.
_PROB_CLIP_LOW = 0.025
_PROB_CLIP_HIGH = 0.975


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def build_win_prob_fn(
    elo_ratings: dict[int, float],
    initial_rating: float = 1500.0,
):
    """Return a closure that computes win probability from Elo ratings."""

    def win_prob(team_a: int, team_b: int) -> float:
        r_a = elo_ratings.get(team_a, initial_rating)
        r_b = elo_ratings.get(team_b, initial_rating)
        return elo_win_probability(r_a, r_b)

    return win_prob


def run_pipeline(
    data_dir: str | Path,
    output_path: str | Path,
    season: int = 2026,
    n_simulations: int = 50_000,
    mc_blend_weight: float = 0.5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Run the full prediction pipeline and write a submission CSV.

    Parameters
    ----------
    data_dir:
        Directory containing the Kaggle competition data files.
    output_path:
        Path where the submission CSV will be written.
    season:
        Competition year.
    n_simulations:
        Number of Monte Carlo bracket simulations.
    mc_blend_weight:
        Weight for bracket-simulation probabilities when blending with
        direct pairwise Elo probabilities.
    random_seed:
        RNG seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        The submission DataFrame (also written to *output_path*).
    """
    data_dir = Path(data_dir)
    output_path = Path(output_path)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    log.info("Loading data from %s", data_dir)
    data = load_data(data_dir)

    reg = data["regular_season"]
    tourney = data["tourney_results"]
    seeds_raw = data["tourney_seeds"]
    submission_raw = data["sample_submission"]

    # ------------------------------------------------------------------
    # 2. Parse submission IDs to find all required matchups
    # ------------------------------------------------------------------
    log.info("Parsing submission IDs")
    submission = parse_submission_ids(submission_raw)
    target_season_sub = submission[submission["Season"] == season].copy()

    if target_season_sub.empty:
        log.warning(
            "No submission rows found for season %d; using all rows.", season
        )
        target_season_sub = submission.copy()

    # ------------------------------------------------------------------
    # 3. Build historical games for Elo computation
    # ------------------------------------------------------------------
    log.info("Building historical game set")
    # Include all regular season up to and including target season,
    # plus tournament results from prior seasons only.
    cols = ["Season", "DayNum", "WTeamID", "LTeamID", "WScore", "LScore", "WLoc"]
    available_cols = [c for c in cols if c in reg.columns]

    reg_filtered = reg[reg["Season"] <= season][available_cols].copy()
    tourney_prior = tourney[tourney["Season"] < season][available_cols].copy()
    all_games = pd.concat([reg_filtered, tourney_prior], ignore_index=True)

    sort_keys = [k for k in ["Season", "DayNum"] if k in all_games.columns]
    if sort_keys:
        all_games = all_games.sort_values(sort_keys).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Compute Elo ratings
    # ------------------------------------------------------------------
    log.info("Computing Elo ratings")
    elo_history = compute_elo_ratings(all_games)
    current_elo = get_pregame_ratings(elo_history, season)
    log.info("Elo ratings computed for %d teams in season %d", len(current_elo), season)

    # ------------------------------------------------------------------
    # 5. Build seed features
    # ------------------------------------------------------------------
    log.info("Building seed features")
    seed_features = build_seed_features(seeds_raw)
    season_seeds = seed_features[seed_features["Season"] == season]

    # Seeded teams ordered by seed (best seed first, used for bracket)
    seeded_teams_df = season_seeds.sort_values("Seed")
    seeded_team_ids = seeded_teams_df["TeamID"].tolist()

    if not seeded_team_ids:
        log.warning(
            "No seed data found for season %d; using all teams in submission.",
            season,
        )
        seeded_team_ids = sorted(
            set(target_season_sub["Team1ID"].tolist())
            | set(target_season_sub["Team2ID"].tolist())
        )

    # ------------------------------------------------------------------
    # 6. Monte Carlo bracket simulation
    # ------------------------------------------------------------------
    log.info("Running Monte Carlo simulation with %d brackets", n_simulations)
    win_prob_fn = build_win_prob_fn(current_elo)

    simulator = MonteCarloBracketSimulator(
        teams=seeded_team_ids,
        win_prob_fn=win_prob_fn,
        n_simulations=n_simulations,
        random_seed=random_seed,
    )
    mc_results = simulator.run(seeded_team_ids)
    log.info("Monte Carlo simulation complete; %d matchups observed", len(mc_results))

    # ------------------------------------------------------------------
    # 7. Compute direct pairwise probabilities for all required matchups
    # ------------------------------------------------------------------
    log.info("Computing direct pairwise Elo probabilities")
    all_team_ids = sorted(
        set(target_season_sub["Team1ID"].tolist())
        | set(target_season_sub["Team2ID"].tolist())
    )
    direct_results = compute_pairwise_probabilities(all_team_ids, win_prob_fn)

    # ------------------------------------------------------------------
    # 8. Blend Monte Carlo and direct probabilities
    # ------------------------------------------------------------------
    log.info("Blending probabilities (MC weight=%.2f)", mc_blend_weight)
    blended = blend_probabilities(mc_results, direct_results, mc_weight=mc_blend_weight)

    prob_map = {
        (int(r["Team1ID"]), int(r["Team2ID"])): float(r["WinProbability"])
        for _, r in blended.iterrows()
    }

    # ------------------------------------------------------------------
    # 9. Build submission
    # ------------------------------------------------------------------
    log.info("Building submission DataFrame")
    predictions = []
    for _, row in target_season_sub.iterrows():
        t1 = int(row["Team1ID"])
        t2 = int(row["Team2ID"])
        key = (min(t1, t2), max(t1, t2))
        prob = prob_map.get(key, 0.5)
        # Clip to avoid extreme Brier / log-loss penalties
        prob = float(np.clip(prob, _PROB_CLIP_LOW, _PROB_CLIP_HIGH))
        predictions.append({"ID": row["ID"], "Pred": prob})

    submission_out = pd.DataFrame(predictions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission_out.to_csv(output_path, index=False)
    log.info("Submission saved to %s (%d rows)", output_path, len(submission_out))

    return submission_out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a March Madness submission using Monte Carlo simulation."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing the Kaggle competition data files.",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Output CSV file path (default: submission.csv).",
    )
    parser.add_argument(
        "--season",
        type=int,
        default=2026,
        help="Competition season year (default: 2026).",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=50_000,
        help="Number of Monte Carlo bracket simulations (default: 50000).",
    )
    parser.add_argument(
        "--mc-weight",
        type=float,
        default=0.5,
        help="Blend weight for Monte Carlo probabilities (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args(argv)

    run_pipeline(
        data_dir=args.data_dir,
        output_path=args.output,
        season=args.season,
        n_simulations=args.simulations,
        mc_blend_weight=args.mc_weight,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()

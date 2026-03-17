"""
pipeline.py
-----------
End-to-end pipeline for March Machine Learning Mania.

Public API
----------
Constants    : MODEL_FEATURE_COLS, XGB_PARAMS, LR_PARAMS,
               XGB_BLEND_WEIGHT, LR_BLEND_WEIGHT

ELO          : build_elo_ratings(game_df, ...)

Features     : build_team_features(regular_df, seeds_df, season_elo,
                                   fitted_pca=None, fitted_kmeans=None)

Matchups     : build_matchup_dataset(game_df, team_features)

Splits       : season_walkforward_splits(seasons, min_train_seasons)
               build_weighted_train_test(regular_matchups, tourney_matchups,
                                         train_seasons, test_season, ...)

Model        : train_ensemble(X_train, y_train, w_train, X_calib, y_calib, ...)
               predict_ensemble(ensemble, X)
               tune_hyperparameters(X_train, y_train, w_train, X_calib, y_calib, n_trials)

Submission   : generate_historical_submission(sample_sub, team_features, ensemble,
                                              model_feature_cols)
               generate_2026_submission(sample_sub, team_features, ensemble,
                                        model_feature_cols)
               generate_combined_submission(sample_sub,
                                            mens_team_features, mens_ensemble,
                                            womens_team_features, womens_ensemble,
                                            model_feature_cols, year_2026_only)

CV           : cross_validate(regular_df, tourney_df, seeds_df, ...)
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from clustering import add_team_clusters
from features import get_summarized_season, build_massey_features


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_FEATURE_COLS = [
    # Elo — multiple representations so no single feature dominates
    "elo_diff",
    "tanh_elo_diff",
    "abs_elo_diff",
    "elo_diff_sq",
    # Seed
    "seed_diff",
    "team_seed",
    "opp_seed",
    # Season-level performance
    "win_pct_diff",
    "sos_diff",
    "adj_net_rating_diff",
    "off_rating_diff",
    "def_rating_diff",
    "net_rating_diff",
    "efg_diff",
    "opp_efg_diff",
    "tov_rate_diff",
    "opp_tov_rate_diff",
    "ft_rate_diff",
    "opp_ft_rate_diff",
    "orb_margin_pg_diff",
    "drb_margin_pg_diff",
    "pos_per_game_diff",
    # Late-season form (last 10 games)
    "last_n_win_pct_diff",
    "last_n_net_rating_diff",
    # Style matchup interactions
    "off_vs_def",
    "def_vs_off",
    "shooting_vs_defense",
    "ball_security_vs_pressure",
    # Cluster
    "same_cluster",
    "cluster_strength_diff",
    # Massey consensus rankings (lower rank = better)
    "massey_mean_rank_diff",
    "massey_median_rank_diff",
    "massey_pom_rank_diff",
    "massey_sag_rank_diff",
    "massey_mor_rank_diff",
]

XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1,
)

LR_PARAMS = dict(
    C=0.1,
    max_iter=1000,
    random_state=42,
)

XGB_BLEND_WEIGHT = 0.7
LR_BLEND_WEIGHT = 0.3


# ─────────────────────────────────────────────────────────────────────────────
# ELO
# ─────────────────────────────────────────────────────────────────────────────

def _expected_win_prob(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def _margin_multiplier(margin, elo_diff):
    return np.log(margin + 1) * (2.2 / (abs(elo_diff) * 0.001 + 2.2))


def build_elo_ratings(game_df, k=20, base_elo=1500, use_margin=True, carry_over=0.75):
    """
    Build end-of-season Elo ratings from regular season game results.

    Parameters
    ----------
    game_df : pd.DataFrame
        Must contain: Season, DayNum, WTeamID, LTeamID, WScore, LScore
    k : float
        Elo update sensitivity.
    base_elo : float
        Starting Elo for new teams.
    use_margin : bool
        Whether to apply a margin-of-victory multiplier.
    carry_over : float
        Fraction of prior-season Elo retained at season start (rest reverts to base_elo).

    Returns
    -------
    pd.DataFrame
        Columns: Season, TeamID, elo — one row per team per season (end of regular season).
    """
    df = game_df.sort_values(["Season", "DayNum"]).reset_index(drop=True)
    current_elo = {}
    last_season = None
    rows = []

    for _, row in df.iterrows():
        season = row["Season"]
        wt, lt = int(row["WTeamID"]), int(row["LTeamID"])
        ws, ls = row["WScore"], row["LScore"]

        if last_season is not None and season != last_season:
            for tid in list(current_elo):
                current_elo[tid] = carry_over * current_elo[tid] + (1 - carry_over) * base_elo
        last_season = season

        ew = current_elo.get(wt, base_elo)
        el = current_elo.get(lt, base_elo)
        exp_w = _expected_win_prob(ew, el)
        mult = _margin_multiplier(ws - ls, ew - el) if use_margin else 1.0

        current_elo[wt] = ew + k * mult * (1 - exp_w)
        current_elo[lt] = el + k * mult * (0 - (1 - exp_w))

        rows.append({
            "Season": season,
            "WTeamID": wt, "LTeamID": lt,
            "W_post_elo": current_elo[wt],
            "L_post_elo": current_elo[lt],
        })

    log = pd.DataFrame(rows)
    season_ending = []

    for season, grp in log.groupby("Season"):
        last_w = grp[["WTeamID", "W_post_elo"]].rename(columns={"WTeamID": "TeamID", "W_post_elo": "elo"})
        last_l = grp[["LTeamID", "L_post_elo"]].rename(columns={"LTeamID": "TeamID", "L_post_elo": "elo"})
        season_elo = pd.concat([last_w, last_l], ignore_index=True).groupby("TeamID").tail(1).copy()
        season_elo["Season"] = season
        season_ending.append(season_elo[["Season", "TeamID", "elo"]])

    return pd.concat(season_ending, ignore_index=True).sort_values(["Season", "TeamID"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Team features
# ─────────────────────────────────────────────────────────────────────────────

def build_team_features(regular_df, seeds_df, season_elo, fitted_pca=None, fitted_kmeans=None, massey_df=None):
    """
    Build the full season-level team feature table.

    Computes season stats → clusters teams → merges Elo → merges Massey rankings.

    In fit mode (fitted_pca / fitted_kmeans are None):
        Fits PCA and KMeans on the data and returns them.

    In transform mode (fitted objects provided):
        Applies pre-fitted objects — use for test seasons to prevent leakage.

    Parameters
    ----------
    regular_df : pd.DataFrame
        Regular season game results.
    seeds_df : pd.DataFrame
        Tournament seed data.
    season_elo : pd.DataFrame
        End-of-season Elo ratings (Season, TeamID, elo).
    fitted_pca : dict or None
    fitted_kmeans : KMeans or None
    massey_df : pd.DataFrame or None
        Raw MMasseyOrdinals data. If provided, ranking features are merged in.

    Returns
    -------
    team_features : pd.DataFrame   Indexed by (Season, TeamID).
    fitted_pca : dict
    fitted_kmeans : KMeans
    """
    season_stats = get_summarized_season(regular_df, seeds_df)
    team_features, fitted_pca, fitted_kmeans = add_team_clusters(
        season_stats, fitted_pca=fitted_pca, fitted_kmeans=fitted_kmeans
    )
    team_features = (
        team_features.reset_index()
        .merge(season_elo, on=["Season", "TeamID"], how="left")
        .set_index(["Season", "TeamID"])
        .sort_index()
    )

    if massey_df is not None:
        massey_feats = build_massey_features(massey_df)
        team_features = team_features.join(massey_feats, how="left")

    return team_features, fitted_pca, fitted_kmeans


# ─────────────────────────────────────────────────────────────────────────────
# Matchup dataset
# ─────────────────────────────────────────────────────────────────────────────

def build_matchup_dataset(game_df, team_features):
    """
    Build a matchup-level dataset with difference and interaction features.

    Creates two rows per game (winner perspective + loser perspective) so
    the model sees balanced classes and all head-to-head directions.

    Parameters
    ----------
    game_df : pd.DataFrame
        Must contain: Season, WTeamID, LTeamID.
    team_features : pd.DataFrame
        Indexed by (Season, TeamID) or with those as columns.

    Returns
    -------
    pd.DataFrame
        One row per team-perspective matchup with engineered features.
    """
    if isinstance(team_features.index, pd.MultiIndex):
        tf = team_features.reset_index().copy()
    else:
        tf = team_features.copy()

    keep_feature_cols = [
        "win_pct", "points_per_game",
        "fg_pct", "fg3_pct", "ft_pct",
        "off_rating", "def_rating", "net_rating",
        "efg", "opp_efg",
        "tov_rate", "opp_tov_rate",
        "ft_rate", "opp_ft_rate",
        "orb_margin_pg", "drb_margin_pg",
        "pos_per_game", "seed", "team_cluster", "elo",
        "sos", "adj_net_rating",
        "last_n_win_pct", "last_n_net_rating",
        # Massey ranking columns (present only if massey_df was passed to build_team_features)
        "massey_mean_rank", "massey_median_rank", "massey_std_rank",
        "massey_pom_rank", "massey_sag_rank", "massey_mor_rank",
        "massey_col_rank", "massey_dol_rank", "massey_ap_rank",
    ]
    keep_feature_cols = [c for c in keep_feature_cols if c in tf.columns]
    tf = tf[["Season", "TeamID"] + keep_feature_cols].copy()

    cluster_strength = (
        tf.groupby("team_cluster")["win_pct"].mean().to_dict()
        if "team_cluster" in tf.columns else {}
    )

    winners = game_df[["Season", "WTeamID", "LTeamID"]].copy()
    winners = winners.rename(columns={"WTeamID": "TeamID", "LTeamID": "OppTeamID"})
    winners["target"] = 1

    losers = game_df[["Season", "WTeamID", "LTeamID"]].copy()
    losers = losers.rename(columns={"LTeamID": "TeamID", "WTeamID": "OppTeamID"})
    losers["target"] = 0

    matchups = pd.concat([winners, losers], ignore_index=True)

    team_tf = tf.rename(columns={c: f"team_{c}" for c in keep_feature_cols})
    opp_tf = tf.rename(columns={"TeamID": "OppTeamID", **{c: f"opp_{c}" for c in keep_feature_cols}})

    matchups = matchups.merge(team_tf, on=["Season", "TeamID"], how="left")
    matchups = matchups.merge(opp_tf, on=["Season", "OppTeamID"], how="left")

    # Cluster features
    if "team_team_cluster" in matchups.columns:
        matchups = matchups.rename(columns={
            "team_team_cluster": "team_cluster",
            "opp_team_cluster": "opp_cluster",
        })
        matchups["same_cluster"] = (matchups["team_cluster"] == matchups["opp_cluster"]).astype(int)
        matchups["team_cluster_strength"] = matchups["team_cluster"].map(cluster_strength)
        matchups["opp_cluster_strength"] = matchups["opp_cluster"].map(cluster_strength)
        matchups["cluster_strength_diff"] = matchups["team_cluster_strength"] - matchups["opp_cluster_strength"]

    # Difference features
    diff_cols = [
        "win_pct", "points_per_game",
        "fg_pct", "fg3_pct", "ft_pct",
        "off_rating", "def_rating", "net_rating",
        "efg", "opp_efg",
        "tov_rate", "opp_tov_rate",
        "ft_rate", "opp_ft_rate",
        "orb_margin_pg", "drb_margin_pg",
        "pos_per_game", "seed", "elo",
        "sos", "adj_net_rating",
        "last_n_win_pct", "last_n_net_rating",
        # Massey diffs (lower rank = better, so negative diff favors team)
        "massey_mean_rank", "massey_median_rank",
        "massey_pom_rank", "massey_sag_rank", "massey_mor_rank",
    ]
    for col in diff_cols:
        tc, oc = f"team_{col}", f"opp_{col}"
        if tc in matchups.columns and oc in matchups.columns:
            matchups[f"{col}_diff"] = matchups[tc] - matchups[oc]

    # Elo nonlinear transforms
    if "elo_diff" in matchups.columns:
        matchups["abs_elo_diff"] = matchups["elo_diff"].abs()
        matchups["elo_diff_sq"] = matchups["elo_diff"] ** 2
        matchups["tanh_elo_diff"] = np.tanh(matchups["elo_diff"] / 400)

    if {"elo_diff", "seed_diff"}.issubset(matchups.columns):
        matchups["elo_vs_seed"] = matchups["elo_diff"] - matchups["seed_diff"]

    if {"elo_diff", "net_rating_diff"}.issubset(matchups.columns):
        matchups["elo_vs_net_rating"] = matchups["elo_diff"] - matchups["net_rating_diff"]

    # Style matchup interactions
    if {"team_off_rating", "opp_def_rating"}.issubset(matchups.columns):
        matchups["off_vs_def"] = matchups["team_off_rating"] - matchups["opp_def_rating"]

    if {"team_def_rating", "opp_off_rating"}.issubset(matchups.columns):
        matchups["def_vs_off"] = matchups["team_def_rating"] - matchups["opp_off_rating"]

    if {"team_efg", "opp_opp_efg"}.issubset(matchups.columns):
        matchups["shooting_vs_defense"] = matchups["team_efg"] - matchups["opp_opp_efg"]

    if {"team_tov_rate", "opp_opp_tov_rate"}.issubset(matchups.columns):
        matchups["ball_security_vs_pressure"] = matchups["team_tov_rate"] - matchups["opp_opp_tov_rate"]

    final_cols = [
        "Season", "TeamID", "OppTeamID", "target",
        "same_cluster", "cluster_strength_diff",
        "win_pct_diff", "points_per_game_diff",
        "fg_pct_diff", "fg3_pct_diff", "ft_pct_diff",
        "off_rating_diff", "def_rating_diff", "net_rating_diff",
        "efg_diff", "opp_efg_diff",
        "tov_rate_diff", "opp_tov_rate_diff",
        "ft_rate_diff", "opp_ft_rate_diff",
        "orb_margin_pg_diff", "drb_margin_pg_diff", "pos_per_game_diff",
        "seed_diff", "team_seed", "opp_seed",
        "sos_diff", "adj_net_rating_diff",
        "last_n_win_pct_diff", "last_n_net_rating_diff",
        "off_vs_def", "def_vs_off",
        "shooting_vs_defense", "ball_security_vs_pressure",
        "elo_diff", "abs_elo_diff", "elo_diff_sq", "tanh_elo_diff",
        "elo_vs_seed", "elo_vs_net_rating",
        "team_cluster", "opp_cluster",
        # Massey ranking diffs
        "massey_mean_rank_diff", "massey_median_rank_diff",
        "massey_pom_rank_diff", "massey_sag_rank_diff", "massey_mor_rank_diff",
    ]
    return matchups[[c for c in final_cols if c in matchups.columns]].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Train / test splits
# ─────────────────────────────────────────────────────────────────────────────

def season_walkforward_splits(seasons, min_train_seasons=5):
    """Return list of (train_seasons, test_season) tuples for walk-forward CV."""
    seasons = sorted(seasons)
    return [(seasons[:i], seasons[i]) for i in range(min_train_seasons, len(seasons))]


def build_weighted_train_test(
    regular_matchups, tourney_matchups,
    train_seasons, test_season,
    regular_weight=1.0, tourney_weight=2.0,
):
    """
    Build weighted train and test sets for one CV fold.

    Train: regular + tourney games from train_seasons (different sample weights).
    Test:  tourney games from test_season only.
    """
    train_reg = regular_matchups[regular_matchups["Season"].isin(train_seasons)].copy()
    train_trn = tourney_matchups[tourney_matchups["Season"].isin(train_seasons)].copy()
    test = tourney_matchups[tourney_matchups["Season"] == test_season].copy()

    train_reg["sample_weight"] = regular_weight
    train_trn["sample_weight"] = tourney_weight
    test["sample_weight"] = 1.0

    return pd.concat([train_reg, train_trn], ignore_index=True), test


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def train_ensemble(
    X_train, y_train, w_train, X_calib, y_calib,
    xgb_params=None, lr_params=None,
    xgb_weight=XGB_BLEND_WEIGHT, lr_weight=LR_BLEND_WEIGHT,
):
    """
    Train a calibrated XGBoost + Logistic Regression blend.

    XGBoost is trained with sample weights; calibrated on held-out data.
    Logistic Regression is trained on StandardScaler-normalized features.

    Returns
    -------
    dict with keys: xgb_cal, lr_cal, lr_scaler, xgb_weight, lr_weight, feature_names
    """
    xgb_params = xgb_params or XGB_PARAMS
    lr_params = lr_params or LR_PARAMS

    # XGBoost
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_train, y_train, sample_weight=w_train)

    xgb_cal = CalibratedClassifierCV(xgb, method="sigmoid", cv="prefit")
    xgb_cal.fit(X_calib, y_calib)

    # Logistic Regression (needs scaled features)
    lr_scaler = StandardScaler()
    X_train_lr = lr_scaler.fit_transform(X_train)
    X_calib_lr = lr_scaler.transform(X_calib)

    lr = LogisticRegression(**lr_params)
    lr.fit(X_train_lr, y_train, sample_weight=w_train)

    lr_cal = CalibratedClassifierCV(lr, method="sigmoid", cv="prefit")
    lr_cal.fit(X_calib_lr, y_calib)

    return {
        "xgb_cal": xgb_cal,
        "lr_cal": lr_cal,
        "lr_scaler": lr_scaler,
        "xgb_weight": xgb_weight,
        "lr_weight": lr_weight,
        "feature_names": list(X_train.columns),
    }


def tune_hyperparameters(
    X_train, y_train, w_train, X_calib, y_calib,
    n_trials=50,
    verbose=False,
):
    """
    Use Optuna to search XGBoost + blend-weight hyperparameters, optimizing
    for Brier score on the calibration set.

    Parameters
    ----------
    X_train, y_train, w_train : training data and sample weights
    X_calib, y_calib          : held-out calibration / validation set
    n_trials : int            : number of Optuna trials (default 50)
    verbose : bool            : if False, suppresses Optuna logging

    Returns
    -------
    dict   Best XGB params and blend weights, ready to pass to train_ensemble:
           {'xgb_params': {...}, 'lr_params': {...},
            'xgb_weight': float, 'lr_weight': float}
    """
    import optuna
    from sklearn.metrics import brier_score_loss as _brier

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        xgb_params = dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        lr_params = dict(
            C=trial.suggest_float("lr_C", 0.001, 10.0, log=True),
            max_iter=1000,
            random_state=42,
        )
        xgb_weight = trial.suggest_float("xgb_weight", 0.4, 0.9)
        lr_weight = 1.0 - xgb_weight

        ensemble = train_ensemble(
            X_train, y_train, w_train, X_calib, y_calib,
            xgb_params=xgb_params, lr_params=lr_params,
            xgb_weight=xgb_weight, lr_weight=lr_weight,
        )
        preds = predict_ensemble(ensemble, X_calib)
        return _brier(y_calib, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    best = study.best_params
    return {
        "xgb_params": dict(
            n_estimators=best["n_estimators"],
            max_depth=best["max_depth"],
            learning_rate=best["learning_rate"],
            subsample=best["subsample"],
            colsample_bytree=best["colsample_bytree"],
            min_child_weight=best["min_child_weight"],
            reg_alpha=best["reg_alpha"],
            reg_lambda=best["reg_lambda"],
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
        "lr_params": dict(
            C=best["lr_C"],
            max_iter=1000,
            random_state=42,
        ),
        "xgb_weight": best["xgb_weight"],
        "lr_weight": 1.0 - best["xgb_weight"],
        "best_brier": study.best_value,
    }


def predict_ensemble(ensemble, X):
    """
    Generate blended probability predictions from a trained ensemble.

    Parameters
    ----------
    ensemble : dict   Output of train_ensemble().
    X : pd.DataFrame  Feature matrix. Columns are reordered to match training order.

    Returns
    -------
    np.ndarray  Predicted win probabilities (team1 beats team2).
    """
    X = X[ensemble["feature_names"]]
    xgb_preds = ensemble["xgb_cal"].predict_proba(X)[:, 1]
    X_lr = ensemble["lr_scaler"].transform(X)
    lr_preds = ensemble["lr_cal"].predict_proba(X_lr)[:, 1]
    return ensemble["xgb_weight"] * xgb_preds + ensemble["lr_weight"] * lr_preds


def get_feature_importances(ensemble):
    """Extract mean XGBoost feature importances from a trained ensemble dict."""
    xgb = ensemble["xgb_cal"].calibrated_classifiers_[0].estimator
    return pd.Series(xgb.feature_importances_, index=ensemble["feature_names"]).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Submission
# ─────────────────────────────────────────────────────────────────────────────

def _parse_submission_ids(submission_df):
    """
    Parse 'Season_Team1_Team2' or 'MSeason_Team1_Team2' / 'WSeason_Team1_Team2' IDs.
    Strips any leading alpha prefix from the season field so both formats work.
    """
    parts = submission_df["ID"].str.split("_", expand=True)
    season_str = parts[0].str.extract(r"(\d+)")[0]  # strip M/W prefix if present
    return pd.DataFrame({
        "Season":  season_str.astype(int).values,
        "WTeamID": parts[1].astype(int).values,
        "LTeamID": parts[2].astype(int).values,
    })


def _predict_for_ids(sample_sub, team_features, ensemble, model_feature_cols):
    """
    Core prediction logic. Predicts for every ID in sample_sub.

    Parameters
    ----------
    sample_sub : pd.DataFrame   Rows from the Kaggle sample submission.
    team_features : pd.DataFrame
    ensemble : dict             Output of train_ensemble().
    model_feature_cols : list

    Returns
    -------
    pd.DataFrame   Columns: ['ID', 'Pred']
    """
    fake_games = _parse_submission_ids(sample_sub)
    matchups = build_matchup_dataset(fake_games, team_features)

    # Team1 = WTeamID = the team listed first in the ID → target == 1 rows
    sub_features = matchups[matchups["target"] == 1].reset_index(drop=True)

    available_cols = [c for c in model_feature_cols if c in sub_features.columns]
    X_sub = sub_features[available_cols].astype("float32").fillna(0.0)

    preds = predict_ensemble(ensemble, X_sub)

    result = sample_sub[["ID"]].copy()
    result["Pred"] = preds

    return result


def generate_historical_submission(sample_sub, team_features, ensemble, model_feature_cols=None):
    """
    Stage 1: Generate predictions for all historical seasons in sample_sub.

    Useful for backtesting or evaluating against past tournaments.

    Parameters
    ----------
    sample_sub : pd.DataFrame       Full Kaggle sample submission (all seasons).
    team_features : pd.DataFrame    Full team feature table, indexed by (Season, TeamID).
    ensemble : dict                 Trained ensemble from train_ensemble().
    model_feature_cols : list       Feature columns used at training time.

    Returns
    -------
    pd.DataFrame   Columns: ['ID', 'Pred']
    """
    model_feature_cols = model_feature_cols or MODEL_FEATURE_COLS
    return _predict_for_ids(sample_sub, team_features, ensemble, model_feature_cols)


def generate_2026_submission(sample_sub, team_features, ensemble, model_feature_cols=None):
    """
    Stage 2: Generate predictions for the 2026 tournament only.

    Filters sample_sub to rows containing '2026' in the ID before predicting.

    Parameters
    ----------
    sample_sub : pd.DataFrame       Kaggle sample submission (may contain all seasons).
    team_features : pd.DataFrame    Full team feature table, indexed by (Season, TeamID).
    ensemble : dict                 Trained ensemble from train_ensemble().
    model_feature_cols : list       Feature columns used at training time.

    Returns
    -------
    pd.DataFrame   Columns: ['ID', 'Pred']
    """
    model_feature_cols = model_feature_cols or MODEL_FEATURE_COLS
    sub_2026 = sample_sub[sample_sub["ID"].str.contains("2026")].reset_index(drop=True)
    return _predict_for_ids(sub_2026, team_features, ensemble, model_feature_cols)


def generate_combined_submission(
    sample_sub,
    mens_team_features, mens_ensemble,
    womens_team_features, womens_ensemble,
    model_feature_cols=None,
    year_2026_only=False,
):
    """
    Generate a single combined submission for both men's and women's tournaments.

    Men's and women's models are applied independently based on TeamID range:
      - Men's:   TeamIDs 1000–1999
      - Women's: TeamIDs 3000–3999

    Both models are then concatenated into one submission DataFrame.

    Parameters
    ----------
    sample_sub : pd.DataFrame       Full Kaggle sample submission (M + W rows).
                                    ID format: '2026_LowTeamID_HighTeamID'
    mens_team_features : pd.DataFrame
    mens_ensemble : dict            Trained ensemble for men's data.
    womens_team_features : pd.DataFrame
    womens_ensemble : dict          Trained ensemble for women's data.
    model_feature_cols : list       Shared feature columns (must be present in both).
    year_2026_only : bool           If True, restrict to 2026 rows only.

    Returns
    -------
    pd.DataFrame   Columns: ['ID', 'Pred'], men's rows followed by women's rows.
    """
    model_feature_cols = model_feature_cols or MODEL_FEATURE_COLS

    sub = sample_sub.copy()
    if year_2026_only:
        sub = sub[sub["ID"].str.contains("2026")].reset_index(drop=True)

    # Split by TeamID range: men's 1000-1999, women's 3000-3999
    # The lower TeamID is always the second field in 'Season_LowID_HighID'
    low_team_id = sub["ID"].str.split("_").str[1].astype(int)
    mens_sub = sub[low_team_id.between(1000, 1999)].reset_index(drop=True)
    womens_sub = sub[low_team_id.between(3000, 3999)].reset_index(drop=True)

    mens_preds = _predict_for_ids(mens_sub, mens_team_features, mens_ensemble, model_feature_cols)
    womens_preds = _predict_for_ids(womens_sub, womens_team_features, womens_ensemble, model_feature_cols)

    return pd.concat([mens_preds, womens_preds], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate(
    regular_df, tourney_df, seeds_df,
    model_feature_cols=None,
    min_train_seasons=5,
    regular_weight=1.0,
    tourney_weight=2.0,
    xgb_params=None,
    lr_params=None,
    verbose=True,
):
    """
    Season walk-forward cross-validation with leak-free clustering.

    Season stats and Elo are computed once. Clustering is refit each fold
    using only training-season data to prevent target leakage.

    Parameters
    ----------
    regular_df : pd.DataFrame   Full regular season game results.
    tourney_df : pd.DataFrame   Full tournament game results.
    seeds_df   : pd.DataFrame   Tournament seeds.
    model_feature_cols : list   Features to use. Defaults to MODULE-level list.
    min_train_seasons : int     Minimum seasons required before first test fold.
    regular_weight : float      Sample weight for regular season games in training.
    tourney_weight : float      Sample weight for tournament games in training.
    xgb_params : dict           XGBoost hyperparameters.
    lr_params  : dict           Logistic Regression hyperparameters.
    verbose : bool              Print fold results as they complete.

    Returns
    -------
    fold_results : pd.DataFrame
        Columns: fold, test_season, train_rows, test_rows, brier_score, log_loss
        brier_score is the competition metric (MSE of predicted probabilities).
    fold_importances : list[pd.Series]
        XGBoost feature importances per fold.
    """
    model_feature_cols = model_feature_cols or MODEL_FEATURE_COLS
    xgb_params = xgb_params or XGB_PARAMS
    lr_params = lr_params or LR_PARAMS

    # Compute Elo and season stats once (sequential — no leakage)
    season_elo = build_elo_ratings(regular_df)
    all_season_stats = get_summarized_season(regular_df, seeds_df)

    # Merge Elo into stats (no cluster labels yet)
    stats_with_elo = (
        all_season_stats.reset_index()
        .merge(season_elo, on=["Season", "TeamID"], how="left")
        .set_index(["Season", "TeamID"])
        .sort_index()
    )

    common_seasons = sorted(
        set(regular_df["Season"].unique()) & set(tourney_df["Season"].unique())
    )
    splits = season_walkforward_splits(common_seasons, min_train_seasons)

    fold_results = []
    fold_importances = []

    for fold_num, (train_seasons, test_season) in enumerate(splits, start=1):

        # Fit clusters on training seasons only
        train_stats = stats_with_elo[
            stats_with_elo.index.get_level_values("Season").isin(train_seasons)
        ]
        _, fitted_pca, fitted_kmeans = add_team_clusters(train_stats)

        # Apply pre-fitted clusters to all seasons
        all_clustered, _, _ = add_team_clusters(
            stats_with_elo, fitted_pca=fitted_pca, fitted_kmeans=fitted_kmeans
        )

        # Build matchups
        reg_matchups = build_matchup_dataset(regular_df, all_clustered)
        trn_matchups = build_matchup_dataset(tourney_df, all_clustered)

        train, test = build_weighted_train_test(
            reg_matchups, trn_matchups,
            train_seasons, test_season,
            regular_weight, tourney_weight,
        )

        train = train.dropna(subset=["target"]).copy()
        test = test.dropna(subset=["target"]).copy()

        if train.empty or test.empty:
            if verbose:
                print(f"Fold {fold_num} | Season {test_season} | skipped (empty split)")
            continue

        train["target"] = train["target"].astype(int)
        test["target"] = test["target"].astype(int)

        # Hold out the most recent training season for calibration
        calib_mask = train["Season"] == train["Season"].max()

        available_cols = [c for c in model_feature_cols if c in train.columns]

        X_train = train.loc[~calib_mask, available_cols].astype("float32")
        y_train = train.loc[~calib_mask, "target"].astype("int32")
        w_train = train.loc[~calib_mask, "sample_weight"].astype("float32")

        X_calib = train.loc[calib_mask, available_cols].astype("float32")
        y_calib = train.loc[calib_mask, "target"].astype("int32")

        X_test = test[available_cols].astype("float32")
        y_test = test["target"].astype("int32")

        # Fill NaNs using training-set medians (e.g. seed is NaN for
        # unseeded regular-season teams). Medians come from X_train only
        # so no leakage from calib or test.
        fill_values = X_train.median()
        X_train = X_train.fillna(fill_values)
        X_calib = X_calib.fillna(fill_values)
        X_test  = X_test.fillna(fill_values)

        ensemble = train_ensemble(
            X_train, y_train, w_train, X_calib, y_calib,
            xgb_params=xgb_params, lr_params=lr_params,
        )

        preds = predict_ensemble(ensemble, X_test)
        # Brier score = MSE of predicted probabilities — this is the competition metric
        brier = brier_score_loss(y_test, preds)
        ll = log_loss(y_test, preds)

        fold_importances.append(get_feature_importances(ensemble).rename(f"season_{test_season}"))

        fold_results.append({
            "fold": fold_num,
            "test_season": test_season,
            "train_rows": len(train),
            "test_rows": len(test),
            "brier_score": brier,
            "log_loss": ll,
        })

        if verbose:
            print(f"Fold {fold_num:>2} | Season {test_season} | brier = {brier:.4f} | log_loss = {ll:.4f}")

    if verbose:
        mean_brier = np.mean([r["brier_score"] for r in fold_results])
        mean_ll = np.mean([r["log_loss"] for r in fold_results])
        print(f"\nMean CV brier score: {mean_brier:.4f}  (log_loss: {mean_ll:.4f})")

    return pd.DataFrame(fold_results), fold_importances

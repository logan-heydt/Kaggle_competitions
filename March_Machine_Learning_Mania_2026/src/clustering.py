from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


OFFENSIVE_COLS = ['fg_pct', 'fg3_pct', 'ft_pct', 'off_rating', 'efg', 'tov_rate', 'ast_per_game', 'points_per_game']
DEFENSIVE_COLS = ['opp_fg_pct', 'opp_fg3_pct', 'opp_ft_pct', 'def_rating', 'opp_tov_rate', 'blk_per_game', 'stl_per_game', 'pf_per_game', 'opp_points_per_game']
REBOUNDING_COLS = ['orb_margin_pg', 'drb_margin_pg']
OVERALL_COLS = ['win_pct', 'possessions_avg'] + OFFENSIVE_COLS + DEFENSIVE_COLS + REBOUNDING_COLS

GROUPED_COLS = {
    "Offense": OFFENSIVE_COLS,
    "Defense": DEFENSIVE_COLS,
    "Rebounds": REBOUNDING_COLS,
    "Overall": OVERALL_COLS,
}


def group_pca(df, fitted_objects=None, min_var_threshold=0.8):
    """
    Apply grouped PCA to team season stats.

    Parameters
    ----------
    df : pd.DataFrame
        Team season stats.
    fitted_objects : dict or None
        If None, fit scalers and PCAs on df and return them (fit mode).
        If provided, transform df using pre-fitted objects (transform mode).
    min_var_threshold : float
        Minimum cumulative explained variance to determine n_components (fit mode only).

    Returns
    -------
    reduced : pd.DataFrame
        PCA-reduced features.
    fitted_objects : dict
        Returned only in fit mode. Dict of {group_name: {"scaler": ..., "pca": ...}}.
    """
    reduced = pd.DataFrame(index=df.index)
    fit_mode = fitted_objects is None
    new_fitted = {}

    for group_name, cols in GROUPED_COLS.items():
        X = df[cols].copy()

        if fit_mode:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca_full = PCA()
            pca_full.fit(X_scaled)
            cum_var = pca_full.explained_variance_ratio_.cumsum()
            num_components = int(np.argmax(cum_var >= min_var_threshold) + 1)

            pca = PCA(n_components=num_components)
            pca_result = pca.fit_transform(X_scaled)

            new_fitted[group_name] = {"scaler": scaler, "pca": pca}
        else:
            scaler = fitted_objects[group_name]["scaler"]
            pca = fitted_objects[group_name]["pca"]
            X_scaled = scaler.transform(X)
            pca_result = pca.transform(X_scaled)

        for i in range(pca_result.shape[1]):
            reduced[f"{group_name}_pca{i+1}"] = pca_result[:, i]

    if fit_mode:
        return reduced, new_fitted
    return reduced


def add_team_clusters(df, fitted_pca=None, fitted_kmeans=None, n_clusters=5, random_state=42):
    """
    Cluster teams using PCA-reduced features.

    Fit mode (fitted_pca and fitted_kmeans are None):
        Fits PCA and KMeans on df. Returns (out, fitted_pca, fitted_kmeans).

    Transform mode (fitted objects provided):
        Applies pre-fitted objects to df. Returns (out, fitted_pca, fitted_kmeans).

    Parameters
    ----------
    df : pd.DataFrame
        Team season stats (indexed by Season, TeamID).
    fitted_pca : dict or None
        Pre-fitted PCA objects from a prior call. None = fit mode.
    fitted_kmeans : KMeans or None
        Pre-fitted KMeans model from a prior call. None = fit mode.
    n_clusters : int
        Number of clusters (fit mode only).
    random_state : int
        Random seed (fit mode only).

    Returns
    -------
    out : pd.DataFrame
        df with 'team_cluster' column added.
    fitted_pca : dict
        Fitted PCA objects.
    fitted_kmeans : KMeans
        Fitted KMeans model.
    """
    if fitted_pca is None:
        reduced, fitted_pca = group_pca(df)
        fitted_kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
        cluster_labels = fitted_kmeans.fit_predict(reduced.values)
    else:
        reduced = group_pca(df, fitted_objects=fitted_pca)
        cluster_labels = fitted_kmeans.predict(reduced.values)

    out = df.copy()
    out["team_cluster"] = cluster_labels
    return out, fitted_pca, fitted_kmeans

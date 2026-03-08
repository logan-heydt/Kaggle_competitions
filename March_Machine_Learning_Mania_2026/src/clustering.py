from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


def group_pca(df):

    offensive_cols = ['fg_pct', 'fg3_pct', 'ft_pct', 'off_rating', 'efg', 'tov_rate', 'ast_per_game', 'points_per_game']
    defensive_cols = ['opp_fg_pct', 'opp_fg3_pct', 'opp_ft_pct', 'def_rating', 'opp_tov_rate', 'blk_per_game', 'stl_per_game', 'pf_per_game', 'opp_points_per_game']
    rebounding_cols = ['orb_margin_pg', 'drb_margin_pg']
    overall_cols = ['win_pct', 'possessions_avg'] + offensive_cols + defensive_cols + rebounding_cols

    grouped_cols = {
        "Offense": offensive_cols,
        "Defense": defensive_cols,
        "Rebounds": rebounding_cols,
        "Overall": overall_cols
    }
    
    min_var_threshold=0.8
    
    reduced = pd.DataFrame(index=df.index)
    pca_group_dict = {}

    for group_name, cols in grouped_cols.items():
        X = df[cols].copy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # First PCA: determine number of components needed
        pca_full = PCA()
        pca_full.fit(X_scaled)

        cum_var = pca_full.explained_variance_ratio_.cumsum()
        num_components = np.argmax(cum_var >= min_var_threshold) + 1
        pca_group_dict[group_name] = int(num_components)

        # Second PCA: fit with chosen number of components
        pca = PCA(n_components=num_components)
        pca_result = pca.fit_transform(X_scaled)

        for i in range(num_components):
            reduced[f"{group_name}_pca{i+1}"] = pca_result[:, i]

    df_pca = df.join(reduced)

    return reduced, df_pca, pca_group_dict

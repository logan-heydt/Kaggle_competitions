import numpy as np
import pandas as pd

from features import get_summarized_season
from clustering import add_team_clusters


def get_pipeline(regular_season_df, seeds_df):
    season_stats = get_summarized_season(regular_season_df, seeds_df)
    season_with_cluster = add_team_clusters(season_stats)
    
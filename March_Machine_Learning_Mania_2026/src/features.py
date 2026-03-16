import numpy as np
import pandas as pd

# Ranking systems to pull individually — all have strong multi-season coverage
MASSEY_SYSTEMS = ["POM", "SAG", "MOR", "COL", "DOL", "AP"]


def build_massey_features(massey_df, systems=None, day_cutoff=133):
    """
    Build per-team-per-season ranking features from the Massey Ordinals file.

    Uses the last available snapshot at or before `day_cutoff` (default 133,
    which is just before the NCAA tournament). Computes individual system ranks
    plus consensus statistics across all available systems.

    Parameters
    ----------
    massey_df : pd.DataFrame
        Raw MMasseyOrdinals data (Season, RankingDayNum, SystemName, TeamID, OrdinalRank).
    systems : list[str] or None
        Specific systems to extract individually. Defaults to MASSEY_SYSTEMS.
    day_cutoff : int
        Only use rankings at or before this day number. For 2026, the latest
        available day is used automatically if it's below the cutoff.

    Returns
    -------
    pd.DataFrame
        Indexed by (Season, TeamID). Columns: massey_{system}_rank for each
        system, plus massey_mean_rank, massey_median_rank, massey_std_rank.
    """
    systems = systems or MASSEY_SYSTEMS

    df = massey_df.copy()

    # For each season, cap at min(day_cutoff, max available day) to handle
    # incomplete seasons like 2026
    season_max_day = df.groupby("Season")["RankingDayNum"].max()
    df["_cap"] = df["Season"].map(season_max_day).clip(upper=day_cutoff)
    df = df[df["RankingDayNum"] <= df["_cap"]].drop(columns="_cap")

    # Take the last snapshot per (Season, SystemName, TeamID)
    df = (
        df.sort_values("RankingDayNum")
        .groupby(["Season", "SystemName", "TeamID"], as_index=False)
        .last()
    )

    # Pivot: one column per system
    pivoted = df.pivot_table(
        index=["Season", "TeamID"],
        columns="SystemName",
        values="OrdinalRank",
        aggfunc="last",
    )
    pivoted.columns.name = None

    # Extract the target systems (may be missing for some seasons)
    result = pd.DataFrame(index=pivoted.index)
    for sys in systems:
        if sys in pivoted.columns:
            result[f"massey_{sys.lower()}_rank"] = pivoted[sys]

    # Consensus across ALL available systems
    all_rank_cols = pivoted.columns.tolist()
    result["massey_mean_rank"]   = pivoted[all_rank_cols].mean(axis=1)
    result["massey_median_rank"] = pivoted[all_rank_cols].median(axis=1)
    result["massey_std_rank"]    = pivoted[all_rank_cols].std(axis=1)

    return result.sort_index()


def get_summarized_season(df: pd.DataFrame, seeds: pd.DataFrame, last_n: int = 10) -> pd.DataFrame:

    seeds = seeds.copy()
    overlap_cols = ["Season", "DayNum", "NumOT"]

    base_stats = ["Score","FGM","FGA","FGM3","FGA3","FTM","FTA","OR","DR","Ast","TO","Stl","Blk","PF"]
    base_stats = [s for s in base_stats if f"W{s}" in df.columns and f"L{s}" in df.columns]

    # ---- Winner game lines (include OppTeamID for SOS computation)
    w_keep = overlap_cols + ["WTeamID", "LTeamID"] + [f"W{s}" for s in base_stats] + [f"L{s}" for s in base_stats]
    w = df[w_keep].copy()
    w = w.rename(columns={"WTeamID": "TeamID", "LTeamID": "OppTeamID"})
    w = w.rename(columns={f"W{s}": s for s in base_stats})
    w = w.rename(columns={f"L{s}": f"opp_{s}" for s in base_stats})
    w["is_win"] = 1

    # ---- Loser game lines
    l_keep = overlap_cols + ["LTeamID", "WTeamID"] + [f"L{s}" for s in base_stats] + [f"W{s}" for s in base_stats]
    l = df[l_keep].copy()
    l = l.rename(columns={"LTeamID": "TeamID", "WTeamID": "OppTeamID"})
    l = l.rename(columns={f"L{s}": s for s in base_stats})
    l = l.rename(columns={f"W{s}": f"opp_{s}" for s in base_stats})
    l["is_win"] = 0

    # ---- Stack to long format
    game_stats = pd.concat([w, l], ignore_index=True)

    # ---- Possessions
    required = {"FGA","OR","TO","FTA","opp_FGA","opp_OR","opp_TO","opp_FTA"}
    if required.issubset(game_stats.columns):
        game_stats["possessions"] = (
            game_stats["FGA"] - game_stats["OR"] + game_stats["TO"] + 0.475 * game_stats["FTA"]
        )
        game_stats["opp_possessions"] = (
            game_stats["opp_FGA"] - game_stats["opp_OR"] + game_stats["opp_TO"] + 0.475 * game_stats["opp_FTA"]
        )
        game_stats["possessions_avg"] = 0.5 * (game_stats["possessions"] + game_stats["opp_possessions"])

    game_stats["games"] = 1

    # ---- Season aggregation
    group_keys = ["Season", "TeamID"]
    sum_cols = (
        ["games", "is_win", "possessions_avg"]
        + base_stats
        + [f"opp_{s}" for s in base_stats]
    )

    season = (
        game_stats[group_keys + sum_cols]
        .groupby(group_keys, as_index=False)
        .sum(numeric_only=True)
    )

    # ---- Basic rates
    season["win_pct"] = season["is_win"] / season["games"]
    season["points_per_game"] = season['Score'] / season["games"]
    season["blk_per_game"] = season['Blk'] / season["games"]
    season["ast_per_game"] = season['Ast'] / season["games"]
    season["stl_per_game"] = season['Stl'] / season["games"]
    season["pf_per_game"] = season['PF'] / season["games"]

    season["opp_points_per_game"] = season['opp_Score'] / season["games"]
    season["opp_blk_per_game"] = season['opp_Blk'] / season["games"]
    season["opp_ast_per_game"] = season['opp_Ast'] / season["games"]
    season["opp_stl_per_game"] = season['opp_Stl'] / season["games"]
    season["opp_pf_per_game"] = season['opp_PF'] / season["games"]

    # ---- Shooting %
    season["fg_pct"]  = season["FGM"]  / season["FGA"]
    season["fg3_pct"] = season["FGM3"] / season["FGA3"]
    season["ft_pct"]  = season["FTM"]  / season["FTA"]

    season["opp_fg_pct"]  = season["opp_FGM"]  / season["opp_FGA"]
    season["opp_fg3_pct"] = season["opp_FGM3"] / season["opp_FGA3"]
    season["opp_ft_pct"]  = season["opp_FTM"]  / season["opp_FTA"]

    # ---- Efficiency metrics
    season["off_rating"] = season["Score"] / season["possessions_avg"]
    season["def_rating"] = season["opp_Score"] / season["possessions_avg"]
    season["net_rating"] = season["off_rating"] - season["def_rating"]

    # ---- Four-factors
    season["efg"] = (season["FGM"] + 0.5 * season["FGM3"]) / season["FGA"]
    season["opp_efg"] = (season["opp_FGM"] + 0.5 * season["opp_FGM3"]) / season["opp_FGA"]

    season["tov_rate"] = season["TO"] / season["possessions_avg"]
    season["opp_tov_rate"] = season["opp_TO"] / season["possessions_avg"]

    season["ft_rate"] = season["FTA"] / season["FGA"]
    season["opp_ft_rate"] = season["opp_FTA"] / season["opp_FGA"]

    season["orb_margin_pg"] = season["OR"]  / (season["OR"] + season["opp_OR"])
    season["drb_margin_pg"] = season["DR"]  / (season["DR"] + season["opp_DR"])

    season["pos_per_game"] = season["possessions_avg"] / season["games"]

    # ---- Strength of Schedule (mean opponent win_pct)
    sos_data = (
        game_stats[["Season", "TeamID", "OppTeamID"]]
        .merge(
            season[["Season", "TeamID", "win_pct"]].rename(
                columns={"TeamID": "OppTeamID", "win_pct": "opp_win_pct"}
            ),
            on=["Season", "OppTeamID"],
            how="left"
        )
    )
    sos = (
        sos_data.groupby(["Season", "TeamID"])["opp_win_pct"]
        .mean()
        .reset_index()
        .rename(columns={"opp_win_pct": "sos"})
    )
    season = season.merge(sos, on=["Season", "TeamID"], how="left")

    # Adjusted net rating: penalizes teams with weak schedules, rewards strong
    season["adj_net_rating"] = season["net_rating"] + (season["sos"] - 0.5) * 10

    # ---- Last-N-games form
    sorted_gs = game_stats.sort_values(["Season", "TeamID", "DayNum"])
    last_n_gs = sorted_gs.groupby(["Season", "TeamID"]).tail(last_n)

    form = last_n_gs.groupby(["Season", "TeamID"]).agg(
        _games=("games", "sum"),
        _wins=("is_win", "sum"),
        _score=("Score", "sum"),
        _opp_score=("opp_Score", "sum"),
        _poss=("possessions_avg", "sum"),
        _fgm=("FGM", "sum"),
        _fga=("FGA", "sum"),
        _fgm3=("FGM3", "sum"),
    ).reset_index()

    form["last_n_win_pct"] = form["_wins"] / form["_games"]
    form["last_n_net_rating"] = (form["_score"] - form["_opp_score"]) / form["_poss"]
    form["last_n_efg"] = (form["_fgm"] + 0.5 * form["_fgm3"]) / form["_fga"]

    form = form[["Season", "TeamID", "last_n_win_pct", "last_n_net_rating", "last_n_efg"]]
    season = season.merge(form, on=["Season", "TeamID"], how="left")

    # ---- Seeds
    seeds["seed"] = seeds["Seed"].apply(lambda x: int(x[1:3]))
    seeds["division"] = seeds["Seed"].apply(lambda x: x[0])
    seeds = seeds.drop(columns=['Seed', 'division'])

    season = season.merge(seeds, how='left', on=['Season', 'TeamID'])

    out = season.set_index(["Season", "TeamID"]).sort_index()
    return out

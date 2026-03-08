import pandas as pd
from sklearn.pipeline import Pipeline

def get_summarized_season(df: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:

    overlap_cols = ["Season", "DayNum", "NumOT"]

    base_stats = ["Score","FGM","FGA","FGM3","FGA3","FTM","FTA","OR","DR","Ast","TO","Stl","Blk","PF"]
    base_stats = [s for s in base_stats if f"W{s}" in df.columns and f"L{s}" in df.columns]

    # ---- 2) build team game lines for winners
    w_keep = overlap_cols + ["WTeamID"] + [f"W{s}" for s in base_stats] + [f"L{s}" for s in base_stats]
    w = df[w_keep].copy()

    # rename team stats
    w = w.rename(columns={"WTeamID": "TeamID"})
    w = w.rename(columns={f"W{s}": s for s in base_stats})
    # opponent stats come from the losing side
    w = w.rename(columns={f"L{s}": f"opp_{s}" for s in base_stats})

    w["is_win"] = 1

    # ---- 3) build team game lines for losers
    l_keep = overlap_cols + ["LTeamID"] + [f"L{s}" for s in base_stats] + [f"W{s}" for s in base_stats]
    l = df[l_keep].copy()

    l = l.rename(columns={"LTeamID": "TeamID"})
    l = l.rename(columns={f"L{s}": s for s in base_stats})
    # opponent stats come from the winning side
    l = l.rename(columns={f"W{s}": f"opp_{s}" for s in base_stats})

    l["is_win"] = 0

    # ---- 4) stack to long format
    game_stats = pd.concat([w, l], ignore_index=True)

    # ---- 5) possessions for team and opponent (game-level, then summed)
    # guard against missing columns
    required = {"FGA","OR","TO","FTA","opp_FGA","opp_OR","opp_TO","opp_FTA"}
    if required.issubset(game_stats.columns):
        game_stats["possessions"] = (
            game_stats["FGA"] - game_stats["OR"] + game_stats["TO"] + 0.475 * game_stats["FTA"]
        )
        game_stats["opp_possessions"] = (
            game_stats["opp_FGA"] - game_stats["opp_OR"] + game_stats["opp_TO"] + 0.475 * game_stats["opp_FTA"]
        )
        # common trick: average them to reduce noise
        game_stats["possessions_avg"] = 0.5 * (game_stats["possessions"] + game_stats["opp_possessions"])

    game_stats["games"] = 1

    # ---- 6) season aggregation (SUMS)
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

    # ---- 7) basic rates
    season["win_pct"] = season["is_win"] / season["games"]

    # shooting % from totals (correct weighting)
    season["fg_pct"]  = season["FGM"]  / season["FGA"]
    season["fg3_pct"] = season["FGM3"] / season["FGA3"]
    season["ft_pct"]  = season["FTM"]  / season["FTA"]

    season["opp_fg_pct"]  = season["opp_FGM"]  / season["opp_FGA"]
    season["opp_fg3_pct"] = season["opp_FGM3"] / season["opp_FGA3"]
    season["opp_ft_pct"]  = season["opp_FTM"]  / season["opp_FTA"]

    # ---- 8) efficiency metrics (points per possession)
    season["off_rating"] = season["Score"] / season["possessions_avg"]
    season["def_rating"] = season["opp_Score"] / season["possessions_avg"]
    season["net_rating"] = season["off_rating"] - season["def_rating"]

    # Four-factors style rates (very useful)
    # eFG%
    season["efg"] = (season["FGM"] + 0.5 * season["FGM3"]) / season["FGA"]
    season["opp_efg"] = (season["opp_FGM"] + 0.5 * season["opp_FGM3"]) / season["opp_FGA"]

    # TOV% ~ TO / poss
    season["tov_rate"] = season["TO"] / season["possessions_avg"]
    season["opp_tov_rate"] = season["opp_TO"] / season["possessions_avg"]

    # FT rate
    season["ft_rate"] = season["FTA"] / season["FGA"]
    season["opp_ft_rate"] = season["opp_FTA"] / season["opp_FGA"]

    # Rebounding proxies (need opp_DR / DR to do true rates; we can do margins now)
    season["orb_margin_pg"] = season["OR"]  / (season["OR"] + season["opp_OR"])
    season["drb_margin_pg"] = season["DR"]  / (season["DR"] + season["opp_DR"])

    season["pos_per_game"] = season["possessions_avg"] / season["games"]

    seeds["seed"] = seeds["Seed"].apply(lambda x: int(x[1:3]))
    seeds["division"] = seeds["Seed"].apply(lambda x: x[0])
    seeds = seeds.drop(columns=['Seed', 'division'])

    season = season.merge(seeds, how='left', on=['Season', 'TeamID'])

    # return indexed
    out = season.set_index(["Season", "TeamID"]).sort_index()
    return out
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_excel(file_path)
        # Create MatchID column
        df['MatchID'] = (
            df[['Date', 'Map', 'Offense', 'Defense']]
            .apply(lambda r: f"{r['Date']}_{r['Map']}_{'_'.join(sorted([r['Offense'], r['Defense']]))}", axis=1)
        )
        # Convert blank strings or whitespace-only to NaN
        df['PlantSite'] = df['PlantSite'].replace(r'^\s*$', np.nan, regex=True)
        logger.info(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def get_fb_leaderboard(df: pd.DataFrame, relevant_teams: list, group_by_map: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute first blood leaderboard."""

    # --- FB counts per player ---
    group_keys = ["FBPlayer", "FBTeam"] + (["Map"] if group_by_map else [])
    fb_counts = (
        df.groupby(group_keys)
          .size()
          .reset_index(name="TotalFBs")
    )

    # --- Rounds per team ---
    if group_by_map:
        rounds_per_team = (
            df.melt(id_vars=["Map"], value_vars=["Offense","Defense"], value_name="Team")
              .groupby(["Map","Team"])
              .size()
              .reset_index(name="RoundsPlayed")
              .rename(columns={"Team":"FBTeam"})
        )
    else:
        rounds_per_team = (
            df[["Offense","Defense"]]
              .stack()
              .value_counts()
              .rename_axis("FBTeam")
              .reset_index(name="RoundsPlayed")
        )

    # --- Merge + rate ---
    leaderboard = (
        fb_counts.merge(rounds_per_team, on=["FBTeam"] + (["Map"] if group_by_map else []), how="left")
                 .assign(FBRate=lambda x: x["TotalFBs"] * 100 / x["RoundsPlayed"])
                 .sort_values(["FBRate","TotalFBs"], ascending=[False, False], ignore_index=True)
    )

    # --- Filter + trim ---
    leaderboard = leaderboard[leaderboard["FBTeam"].isin(relevant_teams)].reset_index(drop=True)
    leaderboard = leaderboard[leaderboard["RoundsPlayed"] >= 9]  # Minimum rounds threshold

    # --- Player × Map matrix for heatmap ---
    if group_by_map:
        fb_matrix = leaderboard[leaderboard["FBTeam"] == 'SPG'].pivot_table(
            index=["FBPlayer"],
            columns="Map",
            values="FBRate",
            fill_value=0
        )
    else:
        fb_matrix = pd.DataFrame()

    return leaderboard, fb_matrix

def off_lb(df: pd.DataFrame, relevant_teams: list = None) -> pd.DataFrame:
    # Only offense rounds
    df_off = df.copy()
    df_off = df_off[df_off['FBTeam'] == df_off['Offense']]   # keep FBs where FB team = offense team

    # Count offense FBs per player
    fb_counts_off = (
        df_off.groupby(['FBPlayer', 'FBTeam', 'Map'])
            .size()
            .reset_index(name='TotalFBs')
    )

    # Count offense rounds per team × map
    off_rounds_per_team_map = (
        df.groupby(['Offense','Map']).size()
            .reset_index(name='RoundsPlayed')
            .rename(columns={'Offense':'FBTeam'})
    )

    # Merge and compute offense FB rate
    fb_leaderboard_off = (
        fb_counts_off.merge(off_rounds_per_team_map, on=['FBTeam','Map'], how='left')
                    .assign(FBRate=lambda x: x['TotalFBs'] * 100 / x['RoundsPlayed'])
                    .sort_values(['FBRate','TotalFBs'], ascending=[False, False], ignore_index=True)
    )

    # Filter for relevant teams (optional)
    fb_leaderboard_off = fb_leaderboard_off[
        fb_leaderboard_off['FBTeam'].isin(relevant_teams)
    ].reset_index(drop=True)

    # Pivot to create matrix
    matrix = (
        fb_leaderboard_off.pivot(index="FBPlayer", columns="Map", values="FBRate")
    )
    return fb_leaderboard_off, matrix

def per_team_map_record(df: pd.DataFrame, relevant_teams: list = None) -> pd.DataFrame:
    """Compute per-team W-L record and round diff per map."""

    # Summarize matches per MatchID + Map
    win_counts = df.groupby(['MatchID','Map','Winner']).size().reset_index(name='RoundsWon')

    # Collect all teams per match
    teams_per_match = (
        df.groupby(['MatchID','Map'])[['Offense','Defense']]
          .agg(lambda x: list(pd.unique(x)))
          .reset_index()
    )
    teams_per_match['Teams'] = teams_per_match['Offense'] + teams_per_match['Defense']
    teams_per_match['Teams'] = teams_per_match['Teams'].apply(lambda x: list(set(x)))

    # Build per-team per-match rows
    records = []
    for _, row in teams_per_match.iterrows():
        match_id, map_name, teams = row['MatchID'], row['Map'], row['Teams']
        t1, t2 = teams[0], teams[1]

        score_t1 = win_counts.loc[
            (win_counts['MatchID']==match_id) & (win_counts['Winner']==t1),
            'RoundsWon'
        ].sum()
        score_t2 = win_counts.loc[
            (win_counts['MatchID']==match_id) & (win_counts['Winner']==t2),
            'RoundsWon'
        ].sum()

        # Determine winner/loser
        if score_t1 > score_t2:
            w_team, l_team = t1, t2
        elif score_t2 > score_t1:
            w_team, l_team = t2, t1
        else:
            w_team, l_team = None, None  # tie

        records.append({
            'Team': t1,
            'Map': map_name,
            'Wins': 1 if t1 == w_team else 0,
            'Losses': 1 if t1 == l_team else 0,
            'RoundsWon': score_t1,
            'RoundsLost': score_t2
        })
        records.append({
            'Team': t2,
            'Map': map_name,
            'Wins': 1 if t2 == w_team else 0,
            'Losses': 1 if t2 == l_team else 0,
            'RoundsWon': score_t2,
            'RoundsLost': score_t1
        })

    team_map_df = pd.DataFrame(records)

    # Compute round difference per team
    team_map_df['RoundDiff'] = team_map_df['RoundsWon'] - team_map_df['RoundsLost']

    # Aggregate if multiple matches per map
    agg_cols = ['Wins','Losses','RoundsWon','RoundsLost','RoundDiff']
    team_map_df = team_map_df.groupby(['Team','Map'])[agg_cols].sum().reset_index()

    # Filter relevant teams
    if relevant_teams is not None:
        team_map_df = team_map_df[team_map_df['Team'].isin(relevant_teams)].reset_index(drop=True)

    return team_map_df

def compute_retake_rate(df: pd.DataFrame, teams: List[str]) -> pd.DataFrame:
    """Compute retake rate per defensive team, broken down by plant site."""
    # Filter only rounds where there was a plant against the defense
    retake_df = df[df['PlantSite'].notna()]

    # Aggregate counts
    agg = (
        retake_df.groupby(['Map', 'Defense', 'PlantSite'])
        .agg(
            RoundsWithPlantAgainst=('PlantSite', 'size'),
            SuccessfulRetakes=('WinType', lambda x: (x.str.lower() == 'defuse').sum())
        )
        .reset_index()
    )
    # Compute RetakeRate %
    agg['RetakeRate'] = agg['SuccessfulRetakes'] * 100 / agg['RoundsWithPlantAgainst']

    # Filter for relevant teams
    agg = agg[agg['Defense'].isin(teams)].reset_index(drop=True)

    # Pivot so you get columns per (Defense, PlantSite)
    agg = (
        agg.pivot(index=['Map', 'PlantSite'], columns='Defense', values=['SuccessfulRetakes','RetakeRate'])
           .fillna(0)
           .reset_index()
    )
    return agg


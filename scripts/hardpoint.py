import pandas as pd
import numpy as np
import logging
from typing import List

logging.basicConfig(level=logging.ERROR)

# Constants
HILL_LENGTH = 60  # seconds

# Loading function
def load_data(file_path: str) -> pd.DataFrame:
    """Load data from XLSX, add features, and return as DataFrame."""
    try:
        df = pd.read_excel(file_path)
        # Parse Score into numeric columns
        df[['Score1', 'Score2']] = df['Score'].str.split('-', expand=True).astype(int)
        # Compute points per hill for each team
        df['Points1'] = df.groupby(['Date','Map','Team1','Team2'])['Score1'].diff().fillna(df['Score1'])
        df['Points2'] = df.groupby(['Date','Map','Team1','Team2'])['Score2'].diff().fillna(df['Score2'])
        # Keep the original row order in a column
        df = df.reset_index().rename(columns={'index':'OrigRow'})
        return df
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return pd.DataFrame()
    except KeyError as e:
        logging.error(f"Missing expected column: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return pd.DataFrame()
    
def prepare_long_form(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DataFrame to long form."""
    try:
        df1 = df[['Map','Hill','Team1','Points1']].rename(columns={'Team1':'Team','Points1':'Points'})
        df2 = df[['Map','Hill','Team2','Points2']].rename(columns={'Team2':'Team','Points2':'Points'})
        long_df = pd.concat([df1, df2], ignore_index=True)
        return long_df
    except KeyError as e:
        logging.error(f"Missing expected column: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

def aggregate_team_stats(df: pd.DataFrame, long_pts: pd.DataFrame, teams: List[str], group_keys: List[str] = ['Team']) -> pd.DataFrame:
    """Aggregate Hardpoint stats per team, or per team × map × hill."""
    try:
        # Expand each row into team perspective
        teams_played = pd.melt(
            df,
            id_vars=df.columns.difference(['Team1','Team2']),
            value_vars=['Team1','Team2'],
            var_name="TeamSlot",
            value_name="Team"
        )
        # Role flags
        teams_played['is_rotator'] = teams_played['Team'] == teams_played['RotateFirst']
        teams_played['is_breaker'] = teams_played['Team'] == teams_played['BreakTeam']
        teams_played['is_scrapper'] = teams_played['Team'] == teams_played['ScrapTeam']
        # --- Rotation win % ---
        rotation = (
            teams_played[teams_played['is_rotator']]
            .groupby(group_keys)['RotWin']
            .mean() * 100
        )
        # --- Rotate first % ---
        rotate_first_counts = teams_played.groupby(group_keys)['is_rotator'].sum()
        total_hills = teams_played.groupby(group_keys).size()
        rotate_first_pct = (rotate_first_counts / total_hills * 100).round(2)

        # --- Break success % ---
        breaks = (
            teams_played[teams_played['is_breaker']]
            .groupby(group_keys)['BreakWin']
            .mean() * 100
        )
        # --- Scrap time per hill ---
        scrap = teams_played[teams_played['is_scrapper']].groupby(group_keys)['ScrapTime'].sum()
        hills_played = teams_played.groupby(group_keys).size()
        scrap_per_hill = scrap / hills_played
        # --- Avg points ---
        avg_points = long_pts.groupby(group_keys)['Points'].mean()
        # Combine everything
        out = pd.DataFrame({
            'RotateFirst%': rotate_first_pct.round(2),
            'RotationWin%': rotation.round(2),
            'BreakSuccess%': breaks.round(2),
            'ScrapTimePerHill': scrap_per_hill.round(2),
            'AvgPoints': avg_points.round(2),
        }).reset_index()
        # Filter for relevant teams
        out = out[out['Team'].isin(teams)]
        return out
    except KeyError as e:
        logging.error(f"Missing expected column: {e}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return []

def calc_control_share(df: pd.DataFrame, teams: List) -> pd.DataFrame:
    """Calculate control share statistics."""
    records = []
    # --- 1) Rotation-first holds ---
    rot_df = df[['RotateFirst', 'HoldDuration']].rename(
        columns={'RotateFirst': 'Team', 'HoldDuration': 'ControlSec'}
    )
    records.append(rot_df)
    # --- 2) Break holds---
    brk_df = df[['BreakTeam', 'BreakDuration']].rename(
        columns={'BreakTeam': 'Team', 'BreakDuration': 'ControlSec'}
    )
    records.append(brk_df)
    # --- 3) Scrap holds ---
    # Normal scrap
    scrap_df = df.loc[(df['ScrapTeam'].notna()) & (df['ScrapTeam'] != 'None') & (df['ScrapTeam'] != 'Split'),
                    ['ScrapTeam', 'ScrapTime']].rename(
        columns={'ScrapTeam': 'Team', 'ScrapTime': 'ControlSec'}
    )
    records.append(scrap_df)
    # Split scrap (duplicate ScrapTime for both Team1 and Team2)
    split_mask = df['ScrapTeam'] == 'Split'
    split_rows = df.loc[split_mask, ['Team1', 'Team2', 'ScrapTime']]
    if not split_rows.empty:
        split_long = pd.concat([
            split_rows[['Team1', 'ScrapTime']].rename(columns={'Team1': 'Team', 'ScrapTime': 'ControlSec'}),
            split_rows[['Team2', 'ScrapTime']].rename(columns={'Team2': 'Team', 'ScrapTime': 'ControlSec'})
        ])
        records.append(split_long)
    # --- Combine all control segments ---
    control_df = pd.concat(records, ignore_index=True)
    # --- Total control per team ---
    total_control = control_df.groupby('Team')['ControlSec'].sum()
    # --- Total hill time available per team ---
    hills_per_team = pd.concat([df['Team1'], df['Team2']]).value_counts()
    total_seconds_per_team = hills_per_team * HILL_LENGTH
    # --- Compute control share % ---
    control_share = (
        (total_control / total_seconds_per_team * 100)
        .reindex(hills_per_team.index)  # keep only teams that actually played
        .reset_index()
    )
    control_share.columns = ['Team', 'ControlSharePct']
    # --- Sort & filter ---
    control_share = control_share.sort_values('ControlSharePct', ascending=False)
    control_share = control_share[control_share['Team'].isin(teams)].round(2)
    return control_share

def prep_pred_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare pivoted dataset for building prediction model."""
    rotation_rows = []
    try:
        for (date, map_, t1, t2), group in df.groupby(['Date', 'Map', 'Team1', 'Team2']):
            # ensure sequential order is preserved
            group = group.reset_index(drop=True)
            # Get scores after first set of hills
            last_row = group[group['Hill'] == 'P4'].head(1)
            if last_row.empty:
                continue
            score1_last = last_row.iloc[0]['Score1']
            score2_last = last_row.iloc[0]['Score2']
            # Determine map winner
            final_row = group[(group['Score1'] == 250) | (group['Score2'] == 250)].head(1)
            if final_row.empty:
                continue
            if final_row.iloc[0]['Score1'] == 250:
                winner = t1
                loser = t2
                target = 1 # from team1's perspective
            else:
                winner = t2
                loser = t1
                target = 0
            # Final Score difference
            if winner == t1:
                score_diff = final_row.iloc[0]['Score1'] - final_row.iloc[0]['Score2']
            else:
                score_diff = final_row.iloc[0]['Score2'] - final_row.iloc[0]['Score1']
            rotation_rows.append({
                'Map': map_,
                'Team1': t1,
                'Team2': t2,
                'Score1_P4': score1_last,
                'Score2_P4': score2_last,
                'ScoreDiff_P4': score1_last - score2_last,
                'Winner': winner,
                'Loser': loser,
                'Target_T1': target,
                'FinalScoreDiff': score_diff,
            })
        hp_model_df = pd.DataFrame(rotation_rows)
        return hp_model_df
    except KeyError as e:
        logging.error(f"Missing expected column: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return pd.DataFrame()

def prep_games_df(df: pd.DataFrame) -> pd.DataFrame:
    """Helper to prepare winrate DataFrame."""
    try:
        # Keep only rows where either team hit 250
        finals = df[(df['Score1'] == 250) | (df['Score2'] == 250)]

        # Take the first such row per game
        finals = finals.groupby(['Date', 'Map', 'Team1', 'Team2'], as_index=False).first()

        # Determine winner/loser
        finals['Winner'] = np.where(finals['Score1'] == 250, finals['Team1'], finals['Team2'])
        finals['Loser']  = np.where(finals['Score1'] == 250, finals['Team2'], finals['Team1'])

        # Build output
        games_df = finals[['Date', 'Map', 'Team1', 'Team2', 'Winner', 'Loser', 'Score']].rename(
            columns={'Score': 'FinalScore'}
        )
        return games_df
    except KeyError as e:
        logging.error(f"Missing expected column: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return pd.DataFrame()
    
def calc_wr(df: pd.DataFrame, teams: List, group_keys: List[str] = ['Team']) -> pd.DataFrame:
    """Helper to calculate winrates."""
    try:
        # --- wins ---
        wins = (
            df.groupby(group_keys[:-1] + ['Winner']).size()
            .rename('Wins')
            .reset_index()
            .rename(columns={'Winner':'Team'})
        )
        # --- losses ---
        losses = (
            df.groupby(group_keys[:-1] + ['Loser']).size()
            .rename('Losses')
            .reset_index()
            .rename(columns={'Loser':'Team'})
        )
        # --- combine ---
        wr_df = pd.merge(wins, losses, on=group_keys, how='outer').fillna(0)
        wr_df = wr_df[wr_df['Team'].isin(teams)]
        wr_df[['Wins','Losses']] = wr_df[['Wins','Losses']].astype(int)
        wr_df['WinRate%'] = (wr_df['Wins'] / (wr_df['Wins'] + wr_df['Losses']) * 100).round(2)
        return wr_df
    except KeyError as e:
        logging.error(f"Missing expected column: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return pd.DataFrame()
    
# def get_winrates(df: pd.DataFrame, teams: List, return_games_df: bool = True) -> pd.DataFrame:
#     """Calculate winrates for each team."""
#     try:
#         # Prepare games DataFrame
#         games_df = _prep_wr_df(df)
#         # Assert games_df is not empty
#         assert not games_df.empty, "Games DataFrame is empty."
#         # Calculate winrates
#         wr_df = _calc_wr(games_df, teams)
#         # Assert wr_df is not empty
#         assert not wr_df.empty, "Winrate DataFrame is empty."
#         if return_games_df:
#             return wr_df, games_df
#         return wr_df
#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {e}")
#         return pd.DataFrame()
    


                

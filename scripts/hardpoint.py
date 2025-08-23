import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# Suppress pandas warnings
pd.options.mode.chained_assignment = None

# --- CONFIG ---
FILE_PATH = 'Week2/HP/hp.xlsx'
OUT_PATH = 'Week3/HP'
SEED = 42
CURVE_POINTS = 201
SEED = 42
N_BOOT = 1000
FEATURE = "ScoreDiff_P4"
TARGET  = "Target_T1"
PRED_OUT_PATH = f'{OUT_PATH}/win_predictor'

# --- HELPER FUNCTIONS ---
def create_df(df):
    rotation_rows = []
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
            target = 1 # from team1's perspective

        else:
            winner = t2
            target = 0

        rotation_rows.append({
            'Map': map_,
            'Team1': t1,
            'Team2': t2,
            'Score1_P4': score1_last,
            'Score2_P4': score2_last,
            'ScoreDiff_P4': score1_last - score2_last,
            'Winner': winner,
            'Target_T1': target
        })

    hp_model_df = pd.DataFrame(rotation_rows)
    return hp_model_df

def get_ci(df, grid):
    rng = np.random.default_rng(42)

    # Each row = one bootstrap model’s curve over the grid
    boot = np.full((N_BOOT, CURVE_POINTS), np.nan, dtype=float)

    for i in range(N_BOOT):
        idx = rng.integers(0, len(df), size=len(df))  # sample rows with replacement
        Xb_raw = df[[FEATURE]].values[idx]
        yb     = df[TARGET].values[idx]

        sc = StandardScaler()
        Xb = sc.fit_transform(Xb_raw)
        try:
            m = LogisticRegression(max_iter=1000, solver="lbfgs")
            m.fit(Xb, yb)
            preds = m.predict_proba(sc.transform(grid))[:, 1]  # length = CURVE_POINTS
            boot[i, :] = preds
        except Exception:
            # rare: perfect separation in tiny resamples; leave this row as NaNs
            pass

    # 95% CIs across bootstrap runs, per grid point
    ci_low  = np.nanpercentile(boot,  2.5, axis=0)   # length = CURVE_POINTS
    ci_high = np.nanpercentile(boot, 97.5, axis=0)   # length = CURVE_POINTS
    return ci_low, ci_high

def hardpoint_analysis(predict=False):
    # make sure output directory exists
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(PRED_OUT_PATH, exist_ok=True)

    # --- LOAD AND PREPARE DATA ---
    df = pd.read_excel(FILE_PATH) # select subset, as needed

    # Parse Score into numeric columns
    df[['Score1', 'Score2']] = df['Score'].str.split('-', expand=True).astype(int)

    # Compute points per hill for each team
    df['Points1'] = df.groupby(['Date','Map','Team1','Team2'])['Score1'].diff().fillna(df['Score1'])
    df['Points2'] = df.groupby(['Date','Map','Team1','Team2'])['Score2'].diff().fillna(df['Score2'])

    # Keep the original row order in a column
    df = df.reset_index().rename(columns={'index':'OrigRow'})

    # Prepare long format for points
    long_pts = pd.concat([
        df[['Map','Hill','Team1','Points1']].rename(columns={'Team1':'Team','Points1':'Points'}),
        df[['Map','Hill','Team2','Points2']].rename(columns={'Team2':'Team','Points2':'Points'})
    ])

    # --- TEAM-LEVEL STATS ---
    # List of all teams
    teams = sorted(set(long_pts['Team']))

    # Team-level stats
    team_stats = []

    for team in teams:
        played = df[(df['Team1']==team) | (df['Team2']==team)]
        # Rotation-win %
        rf = played[played['RotateFirst']==team]
        rot_rate = (rf['RotationWin']=='Yes').mean() * 100 if not rf.empty else np.nan
        # Break-success %
        opp = played[played['RotateFirst']!=team]
        breaks = played[(played['BreakTeam']==team) & (played['BreakSuccess']=='Yes')]
        break_rate = len(breaks) / len(opp) * 100 if len(opp)>0 else np.nan
        # Avg durations
        avg_hold = rf['HoldDuration'].mean()
        avg_break_hold = played[played['BreakTeam']==team]['BreakDuration'].mean()
        # Scrap points
        scrap_pts = df[df['ScrapTeam']==team]['ScrapTime'].sum()
        # Control-Share%
        
        team_stats.append({
            'Team': team,
            'RotationWin': rot_rate,
            'BreakSuccess': break_rate,
            'AvgHoldDuration (s)': avg_hold,
            'AvgBreakDuration (s)': avg_break_hold,
            'ScrapPoints': scrap_pts,
        })

    team_stats_df = pd.DataFrame(team_stats).set_index('Team')

    # Most dominant hills
    team_hill_means = long_pts.groupby(['Team', 'Map', 'Hill'])['Points'].mean().reset_index()
    top_hills = (
        team_hill_means
        .sort_values(['Team', 'Points'], ascending=[True, False])
        .groupby('Team', as_index=False)
        .first()
    )
    top_hills['TopHill'] = top_hills['Map'] + ' ' + top_hills['Hill']
    top_hills = top_hills[['Team', 'TopHill', 'Points']].rename(columns={'Points': 'TopHillAvgPts'})

    # merge team stats with top hills
    team_stats_df = team_stats_df.merge(top_hills, on='Team', how='left')

    # Calculate distance from begin perfect at rotation-win and break-success
    team_stats_df['RotBreakDist'] = np.sqrt((team_stats_df['RotationWin'] - 100)**2 + (team_stats_df['BreakSuccess'] - 100)**2)
    team_stats_df = team_stats_df.sort_values('RotBreakDist')
    # team_stats_df = team_stats_df.drop(columns='RotBreakDist')

    # Assume each hill is 60 seconds long
    HILL_LENGTH = 60  

    # Build a per-team tally of "seconds in control"
    records = []

    for _, row in df.iterrows():
        # 1) Rotation-first holds
        records.append({
            'Team': row['RotateFirst'],
            'ControlSec': row['HoldDuration']
        })
        # 2) Break holds (only if break succeeded)
        records.append({
            'Team': row['BreakTeam'],
            'ControlSec': row['BreakDuration']
        })
        # 3) Scrap holds
        if (row['ScrapTeam'] != 'None') and (row['ScrapTime'] > 0):
            if row['ScrapTeam'] == 'Split':
                # Split means both teams control the scrap
                records.append({
                    'Team': row['Team1'],
                    'ControlSec': row['ScrapTime'] / 2
                })
                records.append({
                    'Team': row['Team2'],
                    'ControlSec': row['ScrapTime'] / 2
                })
            else:
                # Single team controls the scrap
                records.append({
                    'Team': row['ScrapTeam'],
                    'ControlSec': row['ScrapTime']
            })

    control_df = pd.DataFrame(records)

    # Sum total seconds each team was in control
    total_control = control_df.groupby('Team')['ControlSec'].sum()

    hills_per_team = pd.concat([df['Team1'], df['Team2']]).value_counts()
    total_seconds_per_team = hills_per_team * HILL_LENGTH

    # Align and compute control-share % per team
    control_share = (total_control / total_seconds_per_team * 100).reset_index()
    control_share.columns = ['Team', 'ControlSharePct']

    # Sort by control share
    control_share = control_share.sort_values('ControlSharePct', ascending=False)

    team_stats_df = team_stats_df.merge(control_share, on='Team', how='left')
    team_stats_df.to_csv(f'{OUT_PATH}/team_stats.csv', index=False)
    print("Team-level stats saved to 'team_stats.csv'.")

    # --- MAP-LEVEL STATS ---
    mixiest = df.groupby(['Map','Hill'])['PossessionChanges'].mean().reset_index()
    mixiest = mixiest.sort_values(by='PossessionChanges', ascending=False, ignore_index=True).rename(columns={'PossessionChanges':'AvgPossessionChanges'})

    # --- RECAP STATS ---
    # Shutouts
    rows = []
    for _, r in df.iterrows():
        pts1 = r['Points1']
        pts2 = r['Points2']

        if pts1 == 0:
            rows.append({
                'DominantTeam':     r['Team2'],
                'ZeroScoreTeam':    r['Team1'],
                'Map':              r['Map'],
                'Hill':             r['Hill'],
                'ZeroScoreDuration': r['HoldDuration']
            })
        elif pts2 == 0:
            rows.append({
                'DominantTeam':  r['Team1'],
                'ZeroScoreTeam': r['Team2'],
                'Map':           r['Map'],
                'Hill':          r['Hill'],
                'ZeroScoreDuration': r['HoldDuration']
            })

    shutouts = pd.DataFrame(rows)
    shutouts = shutouts.sort_values('ZeroScoreDuration', ascending=False, ignore_index=True)
    shutouts.to_csv(f'{OUT_PATH}/shutouts.csv', index=False)
    print("Shutouts saved to 'shutouts.csv'.")

    # Chained holds
    left = df[['OrigRow','Map','Hill','Team1','Points1','Team2']].rename(
        columns={'Team1':'Team','Points1':'Points','Team2':'Opponent'}
    )
    right = df[['OrigRow','Map','Hill','Team2','Points2','Team1']].rename(
        columns={'Team2':'Team','Points2':'Points','Team1':'Opponent'}
    )
    long = pd.concat([left, right], ignore_index=True)

    # Sort by match and original play order
    long = long.sort_values(['OrigRow'])

    # Compute rolling sum of the last 3 hills *per team within each match*
    long['Chain4'] = (
        long
        .groupby(['Map','Opponent','Team'])['Points']
        .rolling(window=3, min_periods=3)
        .sum()
        .reset_index(level=[0,1,2], drop=True)
    )

    # For each team, find the row with its maximum Chain4
    best_idx = long.groupby('Team')['Chain4'].idxmax()

    # Construct final table including which hills were chained
    records = []
    for team, i in best_idx.items():
        r = long.loc[i]
        grp = long[
            (long['Team']==team) &
            (long['Map']==r['Map']) &
            (long['Opponent']==r['Opponent'])
        ].sort_values('OrigRow')
        pos = grp.index.get_loc(i)
        hills = grp.iloc[pos-3+1:pos+1]['Hill'].tolist()
        records.append({
            'Team': team,
            'Map': r['Map'],
            'Opponent': r['Opponent'],
            f'Best{3}HillSum': r['Chain4'],
            'Hills': hills
        })

    best_chains = pd.DataFrame(records)
    best_chains = best_chains.sort_values('Best3HillSum', ascending=False, ignore_index=True)
    best_chains.to_csv(f'{OUT_PATH}/best_chains.csv', index=False)
    print("Best chains saved to 'best_chains.csv'.")

    # --- PREDICTION MODEL ---
    if predict:
        hp_model_df = create_df(df)
        df = hp_model_df.dropna(subset=[FEATURE, TARGET]).copy()
        df[FEATURE] = df[FEATURE].astype(float)
        df[TARGET]  = df[TARGET].astype(int)

        scaler = StandardScaler()
        X = scaler.fit_transform(df[[FEATURE]].values)
        y = df[TARGET].values

        model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=SEED)
        model.fit(X, y)

        #  Build prediction grid 
        xmin, xmax = float(df[FEATURE].min()), float(df[FEATURE].max())
        pad = max(5.0, 0.1 * (xmax - xmin))
        grid = np.linspace(xmin - pad, xmax + pad, CURVE_POINTS).reshape(-1, 1)
        grid_scaled = scaler.transform(grid)
        p_base = model.predict_proba(grid_scaled)[:, 1]

        # Get 95% CIs via bootstrap
        ci_low, ci_high = get_ci(df, grid)

        # Sanity check (optional)
        assert len(grid.ravel()) == len(p_base) == len(ci_low) == len(ci_high)

        curve = pd.DataFrame({
            "ScoreDiff_P4": grid.ravel(),
            "WinProb_Team1": p_base,
            "CI_low": ci_low,
            "CI_high": ci_high
        })
        curve.to_csv(f'{PRED_OUT_PATH}/hp_curve.csv', index=False)
        print(f"Prediction curve saved to '{PRED_OUT_PATH}/hp_curve.csv'.")

if __name__ == "__main__":
    hardpoint_analysis(predict=True)
    print("Hardpoint analysis completed.")


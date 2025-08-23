import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.proportion import proportion_confint

# Suppress pandas warnings
pd.options.mode.chained_assignment = None

# --- CONFIG ---
FILE_PATH = 'Week2/Control/control.xlsx'
OUT_PATH = 'Week3/Control'
SEED = 42
N_BOOT = 1000
LIFE_COL = 'LifeDiff_2Seg'
TIME_COL = 'TimeTo2Ticks'

# Make sure output directory exists
os.makedirs(OUT_PATH, exist_ok=True)

# --- HELPER FUNCTIONS ---
def parse_diff(s):
    try:
        off, defe = s.split('/')
        return int(off) - int(defe)
    except:
        return pd.NA

def get_off_lives(s):
    try:
        return int(s.split('/')[0])
    except:
        return pd.NA

def get_def_lives(s):
    try:
        return int(s.split('/')[1])
    except:
        return pd.NA

def convert_to_seconds(t):
    try:
        minutes, seconds = map(int, t.split(':'))
        return 120 - (minutes * 60 + seconds)
    except:
        return pd.NA

def control_analysis(predict=False):
    # --- LOAD AND PREPARE DATA ---
    df_full = pd.read_excel(FILE_PATH) # select subset, as needed

    # Feature Engineering
    df_full['Off_Win'] = (df_full['Winner'] == df_full['Offense']).astype(int)
    df_full['LifeDiff_2Seg'] = df_full['Off/Def-2T'].apply(parse_diff)
    df_full['LifeDiff_End'] = df_full['Off/Def_RoundEnd'].apply(parse_diff)
    df_full['OffLivesEnd'] = df_full['Off/Def_RoundEnd'].apply(get_off_lives)
    df_full['DefLivesEnd'] = df_full['Off/Def_RoundEnd'].apply(get_def_lives)
    df_full['TimeTo2Ticks'] = df_full['2TickTime'].apply(convert_to_seconds)

    week3_mask = df_full['Date'] >= '2025-08-14' # Adjust date as needed
    df = df_full[week3_mask].copy()

    # --- MAP-LEVEL STATS ---
    # Side win splits
    win_split = df.groupby('Map')['Off_Win'].agg(
        OffenseWins='sum', TotalRounds='count'
    ).assign(DefenseWins=lambda x: x['TotalRounds'] - x['OffenseWins'],
            OffenseWinRate=lambda x: x['OffenseWins'] / x['TotalRounds'],
            DefenseWinRate=lambda x: x['DefenseWins'] / x['TotalRounds']
    ).reset_index()

    win_split.drop(columns=['OffenseWins', 'DefenseWins', 'TotalRounds'], inplace=True)

    # Zone capture frequencies per map
    zone_counts = []
    for _, row in df.iterrows():
        z = row['Zone(s) Captures']
        if pd.isna(z):
            continue
        zones = [z] if z in ['A', 'B'] else ['A', 'B']
        for zone in zones:
            zone_counts.append((row['Map'], zone))
    zone_df = pd.DataFrame(zone_counts, columns=['Map', 'Zone'])
    zone_freq = (zone_df
                .groupby(['Map', 'Zone'])
                .size()
                .reset_index(name='Count')
                .pivot(index='Map', columns='Zone', values='Count')
                .fillna(0)).reset_index()

    zone_freq.rename(columns={'A': 'A Captures', 'B': 'B Captures'}, inplace=True)

    # Total games played per map
    total_games = df['Map'].value_counts().reset_index()
    total_games.columns = ['Map', 'TotalRounds']

    # Merge zone frequencies with total games
    zone_freq = zone_freq.merge(total_games, on='Map')

    # Combine with map win splits
    map_summary = win_split.merge(zone_freq, on='Map')
    map_summary.to_csv(f'{OUT_PATH}/map_summary.csv', index=False)
    print("Map-level summary saved to 'map_summary.csv'.")

    # --- TEAM-LEVEL STATS ---
    teams = pd.unique(df[['Offense','Defense']].values.ravel())
    life_diff_records = []

    for team in teams:
        # Select only the rounds where this team played
        mask = (df['Offense'] == team) | (df['Defense'] == team)
        # Compute differential per round from that team's perspective
        diffs = df.loc[mask].apply(
            lambda r: (r['OffLivesEnd'] - r['DefLivesEnd'])
                    if r['Offense'] == team
                    else (r['DefLivesEnd'] - r['OffLivesEnd']),
            axis=1
        )
        life_diff_records.append({
            'Team': team,
            'AvgLifeDiff': diffs.mean()
        })

    # Create DataFrame of results
    life_diff = pd.DataFrame(life_diff_records)

    records = []
    for team in teams:
        played = df[(df['Offense']==team)|(df['Defense']==team)]
        wins = (played['Winner']==team).sum()
        losses = len(played) - wins
        records.append({'Team': team, 'RoundDiff': wins - losses})

    round_diff = pd.DataFrame(records)

    ticks_off_overall = (
        df.groupby('Offense', as_index=False)
        .agg(TicksCaptured=('OffTicks','sum'), OffenseRounds=('OffTicks','size'))
        .assign(AvgTicksPerOffRound=lambda d: d['TicksCaptured'] / d['OffenseRounds'].where(d['OffenseRounds']>0, pd.NA))
        .rename(columns={'Offense':'Team'})
        .sort_values('TicksCaptured', ascending=False)
    )
    ticks_def_overall = (
        df.groupby('Defense', as_index=False)
            .agg(TicksAllowed=('OffTicks','sum'), DefenseRounds=('OffTicks','size'))
            .assign(AvgTicksAllowedPerDefRound=lambda d: d['TicksAllowed'] / d['DefenseRounds'].where(d['DefenseRounds']>0, pd.NA))
            .rename(columns={'Defense':'Team'})
    )

    ticks_profile_overall = (
        pd.merge(ticks_off_overall, ticks_def_overall, on='Team', how='outer')
            .fillna({'TicksCaptured':0, 'OffenseRounds':0, 'AvgTicksPerOffRound':0,
                    'TicksAllowed':0, 'DefenseRounds':0, 'AvgTicksAllowedPerDefRound':0})
            .sort_values('TicksCaptured', ascending=False)
    )

    # Merge all team-level stats
    team_summary = (life_diff
                    .merge(round_diff, on='Team', how='outer')
                    .merge(ticks_profile_overall, on='Team', how='outer')
                )
    team_summary.to_csv(f'{OUT_PATH}/team_summary.csv', index=False)
    print("Team-level summary saved to 'team_summary.csv'.")

    # --- LIFE-DIFF CURVE ---
    if predict:
        # Prepare raw data
        df_clean = (
            df_full
            .dropna(subset=[LIFE_COL, TIME_COL, 'Off_Win'])
            .copy()
        )
        X_life = df_clean[LIFE_COL].to_numpy().reshape(-1, 1)
        X_time = df_clean[TIME_COL].to_numpy().reshape(-1, 1)
        y = df_clean['Off_Win'].astype(int).to_numpy()

        # Stack features: [LifeDiff, TwoTickTime]
        X = np.hstack([X_life, X_time])

        # Fit logistic regression on raw rounds (2D)
        log_reg = LogisticRegression(solver='lbfgs', max_iter=1000)
        log_reg.fit(X, y)

        # Build prediction grid along LifeDiff only, holding time at reference(s)
        x_min, x_max = X_life.min(), X_life.max()
        pad = max(1.0, 0.1 * (x_max - x_min))
        life_grid = np.linspace(x_min - pad, x_max + pad, 201)[:, None]

        # Time references (for plotting lines): median (main), and optional fast/slow (25th/75th)
        t_median = np.median(X_time)
        t_p25    = np.percentile(X_time, 25)
        t_p75    = np.percentile(X_time, 75)

        # Helper to predict over life grid for a fixed time value
        def predict_at_time(t_fixed):
            Xg = np.hstack([life_grid, np.full_like(life_grid, fill_value=t_fixed)])
            return log_reg.predict_proba(Xg)[:, 1]

        p_med  = predict_at_time(t_median)
        p_fast = predict_at_time(t_p25)   # "faster" reach to 2 ticks (lower time)
        p_slow = predict_at_time(t_p75)   # "slower" reach to 2 ticks (higher time)

        # Bootstrap CIs (for median-time curve)
        rng = np.random.default_rng(SEED)
        boot_preds = np.full((N_BOOT, life_grid.shape[0]), np.nan, dtype=float)

        for i in range(N_BOOT):
            idx = rng.integers(0, len(X), size=len(X))  # sample rows with replacement
            Xb, yb = X[idx], y[idx]
            try:
                m = LogisticRegression(solver='lbfgs', max_iter=1000)
                m.fit(Xb, yb)
                # predict on life grid at median time
                Xg_med = np.hstack([life_grid, np.full_like(life_grid, fill_value=t_median)])
                boot_preds[i, :] = m.predict_proba(Xg_med)[:, 1]
            except Exception:
                # leave this bootstrap row as NaNs on failure (rare with small samples / separation)
                pass

        ci_low  = np.nanpercentile(boot_preds,  2.5, axis=0)
        ci_high = np.nanpercentile(boot_preds, 97.5, axis=0)

        # Save CSV for Datawrapper ---
        out = pd.DataFrame({
            'LifeDiff': life_grid.ravel(),
            'WinProb_medTime': p_med,     # main curve (time fixed at median)
            'CI_low': ci_low,             # 95% CI around the median-time curve
            'CI_high': ci_high,
            'WinProb_fastTime': p_fast,   # optional comparison lines (no CI)
            'WinProb_slowTime': p_slow
        })

        # (Optional) include the numeric time references used (seconds) as columns for clarity
        out.attrs = {'t_median': float(t_median), 't_p25': float(t_p25), 't_p75': float(t_p75)}
        out.to_csv(f"{OUT_PATH}/life_diff_curve.csv", index=False)

        print(f"Saved curve (2-feature model; LifeDiff on x-axis) with CI at median {TIME_COL}: {OUT_PATH}/life_diff_curve.csv")
        print(f"Time refs used — median: {t_median:.2f}, p25: {t_p25:.2f}, p75: {t_p75:.2f} (same units as {TIME_COL})")

if __name__ == "__main__":
    control_analysis(predict=True)
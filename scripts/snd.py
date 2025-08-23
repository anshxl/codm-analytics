import os
import pandas as pd
import numpy as np

# CONFIG
RNG = np.random.default_rng(42)  # Random number generator for reproducibility
FILE_PATH = 'Week2/SnD/snd.xlsx'
OUT_PATH = 'Week3/SnD'

# Helper functions
def parse_clock_to_seconds(x):
    """Parse HH:MM:SS, M:SS, or SS into integer seconds (match time remaining)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan

    # Handle HH:MM:SS
    parts = s.split(":")
    if len(parts) == 3:  # HH:MM:SS → ignore hours
        mm, ss, _ = parts
        return int(mm) * 60 + int(ss)
    elif len(parts) == 2:  # M:SS
        mm, ss = parts
        return int(mm) * 60 + int(ss)
    elif s.isdigit():  # seconds only
        return int(s)

    return np.nan

def bootstrap_ci_mean(a, n_boot=1000, ci=95, rng=None):
    """Percentile bootstrap CI for the mean; ignores NaNs."""
    arr = pd.Series(a).dropna().to_numpy()
    if arr.size == 0:
        return (np.nan, np.nan)
    if rng is None:
        rng = np.random.default_rng(42)
    boot = np.empty(n_boot, dtype=float)
    n = arr.size
    for i in range(n_boot):
        boot[i] = np.mean(rng.choice(arr, size=n, replace=True))
    alpha = (100 - ci) / 2.0
    return (np.percentile(boot, alpha), np.percentile(boot, 100 - alpha))

def agg_with_ci(group, col):
    mean_val = group[col].mean()
    lo, hi = bootstrap_ci_mean(group[col], n_boot=1000, ci=95, rng=RNG)
    return pd.Series({f'{col}_mean': mean_val, f'{col}_CI_low': lo, f'{col}_CI_high': hi})

def main():
    # Ensure output directory exists
    os.makedirs(OUT_PATH, exist_ok=True)
    # Load the data
    df = pd.read_excel(FILE_PATH)
    # Change 'Xrock' to 'XROCK' in the 'Offense','Defense', 'FBTeam', 'Winner' columns
    df['Offense'] = df['Offense'].replace('Xrock', 'XROCK')
    df['Defense'] = df['Defense'].replace('Xrock', 'XROCK')
    df['FBTeam'] = df['FBTeam'].replace('Xrock', 'XROCK')
    df['Winner'] = df['Winner'].replace('Xrock', 'XROCK')
    # Convert blank strings or whitespace-only to NaN
    df['PlantSite'] = df['PlantSite'].replace(r'^\s*$', np.nan, regex=True)
    # week3_mask = df['Date'] >= '2025-08-14'
    # df = df[week3_mask]
    df_snd = df.copy()

    # LEADERBOARD
    # Count total FBs per player
    fb_counts = (
        df_snd.groupby(['FBPlayer', 'FBTeam'])
            .size()
            .reset_index(name='TotalFBs')
    )

    # Count total rounds played per team
    rounds_per_team = (
        pd.concat([
            df_snd.groupby('Offense').size(),
            df_snd.groupby('Defense').size()
        ], axis=1).fillna(0).sum(axis=1).astype(int).reset_index()
    )
    rounds_per_team.columns = ['FBTeam', 'RoundsPlayed']

    # Merge and compute FB rate
    fb_leaderboard = (
        fb_counts.merge(rounds_per_team, on='FBTeam', how='left')
                .assign(FBRate=lambda x: x['TotalFBs'] * 100 / x['RoundsPlayed'])
                .sort_values(['FBRate', 'TotalFBs'], ascending=[False, False], ignore_index=True)
                .head(10)
    )
    # Save leaderboard
    fb_leaderboard.to_csv(f'{OUT_PATH}/fb_leaderboard.csv', index=False)
    print("FB leaderboard saved to 'fb_leaderboard.csv'.")

    # FBs per team
    fb_per_team = (
        df_snd.groupby('FBTeam')
            .size()
            .reset_index(name='TotalFBs')
            .sort_values('TotalFBs', ascending=False)
    )
    fb_rate_per_team = (
        fb_per_team.merge(rounds_per_team, on='FBTeam', how='left')
        .assign(FBRate=lambda x: x['TotalFBs'] * 100 / x['RoundsPlayed'])
        .sort_values('FBRate', ascending=False, ignore_index=True)
    )

    # Count plants per map/site (ignore rounds with no plant)
    plants = (
        df_snd.dropna(subset=['PlantSite'])
            .groupby(['Map', 'PlantSite'])
            .size()
            .reset_index(name='Plants')
    )

    # Wide format: columns A and B
    site_counts = (
        plants.pivot(index='Map', columns='PlantSite', values='Plants')
            .fillna(0)
            .rename(columns={'A':'Plants_A', 'B':'Plants_B'})
            .reset_index()
    )

    # Totals and shares
    site_counts['TotalPlants'] = site_counts['Plants_A'] + site_counts['Plants_B']
    site_counts['Share_A'] = np.where(site_counts['TotalPlants']>0,
                                    site_counts['Plants_A']*100/site_counts['TotalPlants'], 0.0)
    site_counts['Share_B'] = np.where(site_counts['TotalPlants']>0,
                                    site_counts['Plants_B']*100/site_counts['TotalPlants'], 0.0)
    
    # Sort and save
    site_counts = site_counts.sort_values(['Share_A'], ascending=False, ignore_index=True)
    site_counts.to_csv(f'{OUT_PATH}/plant_sites.csv', index=False)
    print("Plant sites summary saved to 'plant_sites.csv'.")

    # Plant vs. no plant win rates
    df_snd['Planted'] = pd.notna(df_snd['PlantSite'])

    # Flag offense win
    df_snd['OffenseWin'] = df_snd['Winner'] == df_snd['Offense']

    # Aggregate
    off_win_plant_stats = (
        df_snd.groupby(['Offense','Planted'])['OffenseWin']
        .mean()
        .reset_index()
        .pivot(index='Offense', columns='Planted', values='OffenseWin')
        .rename(columns={False:'WinRate_NoPlant', True:'WinRate_Plant'})
        .reset_index()
        .fillna(0)
    )

    off_win_plant_stats['WinRate_NoPlant'] = off_win_plant_stats['WinRate_NoPlant'] * 100
    off_win_plant_stats['WinRate_Plant'] = off_win_plant_stats['WinRate_Plant'] * 100
    off_win_plant_stats = off_win_plant_stats.sort_values('WinRate_Plant', ascending=False, ignore_index=True)
    off_win_plant_stats.to_csv(f'{OUT_PATH}/offense_win_rates.csv', index=False)
    print("plant vs. no plant win rates saved to 'offense_win_rates.csv'.")

    # Aggregate plant rate
    plant_rate_per_team = (
        df_snd.groupby('Offense')['Planted']
        .mean()
        .reset_index(name='PlantRate')
    )
    plant_rate_per_team['PlantRate'] = plant_rate_per_team['PlantRate'] * 100

    # Retake stats
    # Filter rounds with a plant while team is on defense
    retake_df = df_snd[pd.notna(df['PlantSite'])]

    # For each defense team: total planted-against rounds
    retake_stats = (
        retake_df.groupby('Defense')
                .size()
                .reset_index(name='RoundsWithPlantAgainst')
    )

    # For each defense team: successful retakes (win by defuse)
    retake_success = (
        retake_df[retake_df['WinType'].str.lower() == 'defuse']
            .groupby('Defense')
            .size()
            .reset_index(name='SuccessfulRetakes')
    )
    # Defense win rate
    def_win_rate = (
        df_snd.groupby('Defense')['OffenseWin']
        .apply(lambda x: 100*(1 - x.mean()))
        .reset_index(name='DefenseWinRate')
    )
    # Merge and compute rate
    retake_stats = retake_stats.merge(retake_success, on='Defense', how='left').fillna(0)
    retake_stats['RetakeRate'] = retake_stats['SuccessfulRetakes'] *100 / retake_stats['RoundsWithPlantAgainst']

    retake_stats = retake_stats.merge(def_win_rate, on='Defense', how='left')
    retake_stats = retake_stats[['Defense', 'RetakeRate']]

    # Cumulative round differential across ALL rounds
    all_rounds = []

    # For Offense perspective
    offense_results = df_snd[['Offense', 'Winner']].copy()
    offense_results['Diff'] = np.where(offense_results['Offense'] == offense_results['Winner'], 1, -1)
    offense_results = offense_results.rename(columns={'Offense': 'Team'})[['Team', 'Diff']]
    all_rounds.append(offense_results)

    # For Defense perspective
    defense_results = df_snd[['Defense', 'Winner']].copy()
    defense_results['Diff'] = np.where(defense_results['Defense'] == defense_results['Winner'], 1, -1)
    defense_results = defense_results.rename(columns={'Defense': 'Team'})[['Team', 'Diff']]
    all_rounds.append(defense_results)

    # Combine offense + defense
    round_diff = (
        pd.concat(all_rounds)
        .groupby('Team')['Diff']
        .sum()
        .rename('RoundDiff')
    )

    retake_stats = retake_stats.merge(plant_rate_per_team[['Offense', 'PlantRate']], left_on='Defense', right_on='Offense', how='left')

    retake_stats = retake_stats.merge(round_diff, left_on='Defense', right_index=True, how='left')
    retake_stats = retake_stats[['Defense', 'PlantRate', 'RetakeRate', 'RoundDiff']]
    retake_stats.to_csv(f'{OUT_PATH}/retake_stats.csv', index=False)
    print("Plant-Retake stats saved to 'retake_stats.csv'.")

    # Harmonize FB clock column name
    if 'FBClock' not in df_snd.columns and 'FBTime' in df_snd.columns:
        df = df_snd.rename(columns={'FBTime': 'FBClock'})

    # Parse clocks -> seconds remaining at the event
    for col in ['PlantClock', 'EndClock', 'FBClock']:
        if col in df.columns:
            df[col + '_s'] = df[col].apply(parse_clock_to_seconds)
        else:
            df[col + '_s'] = np.nan  # if missing, fill with NaN

    # Calculate elapsed time for planting and end
    df['Planted'] = df['PlantSite'].notna()

    # Round elapsed
    df['RoundElapsed_s'] = np.where(
        df['Planted'],
        (120 - df['PlantClock_s']) + (45 - df['EndClock_s']),
        120 - df['EndClock_s']
    )

    df['FBElapsed_s'] = 120 - df['FBClock_s']

    # Plant elapsed (only if planted)
    df['PlantElapsed_s'] = np.where(df['Planted'], 120 - df['PlantClock_s'], np.nan)

    # Per attacking team
    tempo_round = df.groupby('Offense').apply(agg_with_ci, col='RoundElapsed_s').reset_index()
    tempo_fb    = df.groupby('Offense').apply(agg_with_ci, col='FBElapsed_s').reset_index()
    tempo_plant = (df[df['Planted']]
                .groupby('Offense')
                .apply(agg_with_ci, col='PlantElapsed_s')
                .reset_index())

    # Merge all
    tempo = (tempo_round
            .merge(tempo_fb, on='Offense', how='left')
            .merge(tempo_plant, on='Offense', how='left')
            .rename(columns={
                'RoundElapsed_s_mean': 'AvgRoundLen_s',
                'FBElapsed_s_mean':    'AvgFBElapsed_s',
                'PlantElapsed_s_mean': 'AvgPlantElapsed_s'
            })
            )
    
    # Optional: order columns nicely
    cols_order = [
        'Offense',
        'AvgRoundLen_s','RoundElapsed_s_CI_low','RoundElapsed_s_CI_high',
        'AvgFBElapsed_s','FBElapsed_s_CI_low','FBElapsed_s_CI_high',
        'AvgPlantElapsed_s','PlantElapsed_s_CI_low','PlantElapsed_s_CI_high'
    ]
    tempo = tempo.reindex(columns=[c for c in cols_order if c in tempo.columns])
    tempo.to_csv(f'{OUT_PATH}/tempo_stats.csv', index=False)
    print("Tempo stats saved to 'tempo_stats.csv'.")

if __name__ == "__main__":
    main()


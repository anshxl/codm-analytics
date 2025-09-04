import pandas as pd

# Helpers
def _parse_diff(s: str):
    try:
        off, defe = s.split('/')
        return int(off) - int(defe)
    except Exception as e:
        print(f"Error parsing life diff '{s}': {e}")
        return pd.NA

def _get_off_lives(s: str):
    try:
        return int(s.split('/')[0])
    except Exception as e:
        print(f"Error parsing off lives '{s}': {e}")
        return pd.NA

def _get_def_lives(s: str):
    try:
        return int(s.split('/')[1])
    except Exception as e:
        print(f"Error parsing def lives '{s}': {e}")
        return pd.NA
    
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    # Ensure Date is a date (stringify to avoid timezone issues in ID)
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime("%Y-%m-%d")
    # Build a stable team-pair token (order-free) and a MatchID: Date + Map + Teams
    df['TeamPair'] = df.apply(lambda r: " vs ".join(sorted([str(r['Offense']), str(r['Defense'])])), axis=1)
    df['MatchID'] = df['Date'] + " | " + df['Map'].astype(str) + " | " + df['TeamPair']
    # Create side win flag for Offense
    df['Off_Win'] = (df['Winner'] == df['Offense']).astype(int)
    df['LifeDiff_2Seg'] = df['Off/Def-2T'].apply(_parse_diff)
    df['LifeDiff_End'] = df['Off/Def_RoundEnd'].apply(_parse_diff)
    df['OffLivesEnd'] = df['Off/Def_RoundEnd'].apply(_get_off_lives)
    df['DefLivesEnd'] = df['Off/Def_RoundEnd'].apply(_get_def_lives)
    return df

def win_splits(df: pd.DataFrame, teams: list) -> pd.DataFrame:
    # Per-map, per-team win splits
    team_win_split = (
        df
        .melt(
            id_vars=['Map', 'Off_Win'],
            value_vars=['Offense', 'Defense'],
            var_name='Side',
            value_name='Team'
        )
        .assign(
            RoundWin=lambda x: (
                (x['Side'] == 'Offense') & (x['Off_Win'] == 1) |
                (x['Side'] == 'Defense') & (x['Off_Win'] == 0)
            ).astype(int)
        )
        .groupby(['Map', 'Team', 'Side'])
        .agg(
            Rounds=('RoundWin', 'count'),
            Wins=('RoundWin', 'sum')
        )
        .assign(WinRate=lambda x: x['Wins'] / x['Rounds'])
        .reset_index()
    )
    team_win_split = team_win_split[team_win_split['Team'].isin(teams)]
    # Pivot to get offense/defense side-by-side
    team_map_split = (
        team_win_split
        .pivot(index=['Map', 'Team'], columns='Side', values='WinRate')
        .reset_index()
        .rename_axis(None, axis=1)   # remove multi-index name
        .rename(columns={'Offense': 'OffenseWinRate', 'Defense': 'DefenseWinRate'})
    )
    return team_map_split

def avg_off_ticks(df: pd.DataFrame, teams: list) -> pd.DataFrame:
    ticks_off_overall = (
        df.groupby(['Map','Offense'], as_index=False)
        .agg(TicksCaptured=('OffTicks','sum'), OffenseRounds=('OffTicks','size'))
        .assign(AvgTicksPerOffRound=lambda d: d['TicksCaptured'] / d['OffenseRounds'].where(d['OffenseRounds']>0, pd.NA))
        .rename(columns={'Offense':'Team'})
        .sort_values('TicksCaptured', ascending=False)
    )
    ticks_off_overall = ticks_off_overall[ticks_off_overall['Team'].isin(teams)]
    # pivot table to have maps as columns
    ticks_off_overall = ticks_off_overall.pivot(index='Team', columns='Map', values='AvgTicksPerOffRound')
    ticks_off_overall = ticks_off_overall.fillna(0).reset_index()
    ticks_off_overall = ticks_off_overall.rename_axis(None, axis=1)   # remove multi-index name
    ticks_off_overall = ticks_off_overall.round(2)
    return ticks_off_overall

def avg_def_ticks(df: pd.DataFrame, teams:list) -> pd.DataFrame:
    ticks_def_overall = (
        df.groupby(['Map','Defense'], as_index=False)
            .agg(TicksAllowed=('OffTicks','sum'), DefenseRounds=('OffTicks','size'))
            .assign(AvgTicksAllowedPerDefRound=lambda d: d['TicksAllowed'] / d['DefenseRounds'].where(d['DefenseRounds']>0, pd.NA))
            .rename(columns={'Defense':'Team'})
            .sort_values('TicksAllowed', ascending=True)
    )
    ticks_def_overall = ticks_def_overall[ticks_def_overall['Team'].isin(teams)]
    # pivot table to have maps as columns
    ticks_def_overall = ticks_def_overall.pivot(index='Team', columns='Map', values='AvgTicksAllowedPerDefRound')
    ticks_def_overall = ticks_def_overall.fillna(0).reset_index()
    ticks_def_overall = ticks_def_overall.rename_axis(None, axis=1)   # remove multi-index name
    ticks_def_overall = ticks_def_overall.round(2)
    return ticks_def_overall

def tick_profile(df: pd.DataFrame, teams: list) -> pd.DataFrame:
    tick_profile_overall = (
        pd.merge(
            avg_off_ticks(df, teams),
            avg_def_ticks(df, teams),
            on='Team',
            how='outer',
            suffixes=('_Captured', '_Allowed')
        )
    )
    return tick_profile_overall

def calc_wr(df: pd.DataFrame, teams: list, by_map: bool = False) -> pd.DataFrame:
    # Decide grouping keys
    match_keys = ['MatchID'] + (['Map'] if by_map else [])

    # --- Round wins per match per team ---
    round_counts = (
        df.groupby(match_keys + ['Winner'])
          .size()
          .reset_index(name='RoundWins')
    )

    # --- Match winners (first to 4 rounds) ---
    match_winners = (
        round_counts.loc[round_counts.groupby(match_keys)['RoundWins'].idxmax()]
    )
    match_winners = match_winners[match_winners['RoundWins'] >= 4] \
                                 .rename(columns={'Winner': 'MatchWinner'}) \
                                 [match_keys + ['MatchWinner']]

    # --- Participants per match ---
    participants = (
        df.groupby(match_keys)
          .apply(lambda g: sorted(set(g['Offense']).union(set(g['Defense']))))
          .reset_index(name='Teams')
    )
    team_match_rows = participants.explode('Teams').rename(columns={'Teams': 'Team'})

    # --- Games played (only include decided matches) ---
    decided_matches = team_match_rows[
        team_match_rows['MatchID'].isin(match_winners['MatchID'])
    ]
    if by_map:
        decided_matches = decided_matches.merge(
            match_winners[match_keys], on=match_keys, how='inner'
        )
    games_played = (
        decided_matches.groupby(['Team'] + (['Map'] if by_map else []))
                       .size()
                       .reset_index(name='Games')
    )

    # --- Wins per team ---
    wins = (
        decided_matches.merge(match_winners, on=match_keys, how='inner')
                       .assign(Win=lambda x: (x['Team'] == x['MatchWinner']).astype(int))
                       .groupby(['Team'] + (['Map'] if by_map else []))['Win']
                       .sum()
                       .reset_index(name='Wins')
    )

    # --- Win rate ---
    win_rates = (
        games_played.merge(wins, on=['Team'] + (['Map'] if by_map else []), how='left')
                    .fillna({'Wins': 0})
                    .assign(WinRate=lambda x: x['Wins'] / x['Games'])
                    .sort_values(['WinRate', 'Wins', 'Games'], ascending=[False, False, False])
                    .reset_index(drop=True)
    )

    win_rates = win_rates[win_rates['Team'].isin(teams)].reset_index(drop=True)
    win_rates['WinRate'] = win_rates['WinRate'].round(2)

    return win_rates

def zone_caps(df: pd.DataFrame, teams: list) -> pd.DataFrame:
    # --- Step 1: Build long-form zone captures ---
    zone_long = (
        df
        .dropna(subset=['Zone(s) Captures'])
        .assign(
            Zones=lambda x: x['Zone(s) Captures'].map(
                lambda z: ['A', 'B'] if z not in ['A', 'B'] else [z]
            )
        )
        .explode('Zones')
        [['Map', 'Offense', 'Zones']]
        .rename(columns={'Offense': 'Team', 'Zones': 'Zone'})
    )

    # --- Step 2: Raw counts of A/B captures ---
    zone_counts = (
        zone_long
        .groupby(['Map', 'Team', 'Zone'])
        .size()
        .reset_index(name='Count')
        .pivot(index=['Map', 'Team'], columns='Zone', values='Count')
        .fillna(0)
        .reset_index()
        .rename(columns={'A': 'A Captures', 'B': 'B Captures'})
    )

    # --- Step 3: Total offensive rounds per team per map ---
    off_rounds = (
        df
        .groupby(['Map', 'Offense'])
        .size()
        .reset_index(name='TotalOffRounds')
        .rename(columns={'Offense': 'Team'})
    )

    # --- Step 4: Merge and compute capture rates ---
    zone_freq_team = (
        zone_counts.merge(off_rounds, on=['Map', 'Team'], how='left')
    )

    for zone in ['A', 'B']:
        zone_freq_team[f'{zone} CaptureRate'] = (
            zone_freq_team[f'{zone} Captures'] / zone_freq_team['TotalOffRounds'] * 100
        )

    # Final tidy DataFrame
    zone_freq_team = zone_freq_team[
        ['Map', 'Team',
        'A Captures', 'B Captures',
        'A CaptureRate', 'B CaptureRate']
    ]
    zone_freq_team = zone_freq_team[zone_freq_team['Team'].isin(teams)].reset_index(drop=True)
    zone_freq_team[['A CaptureRate', 'B CaptureRate']] = zone_freq_team[['A CaptureRate', 'B CaptureRate']].round(1)
    return zone_freq_team

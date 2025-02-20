import numpy as np
import pandas as pd


# =============================================================================
# Helper Functions
# =============================================================================

def preprocess_data(ufc_stats: pd.DataFrame, fighter_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the UFC and fighter stats dataframes:
      - Standardize string columns.
      - Convert dates.
      - Merge fighter DOB and compute age.
      - Clean and drop unwanted columns.
      - Convert time strings to seconds.
    """
    # Standardize fighter names and dates
    ufc_stats['fighter'] = ufc_stats['fighter'].astype(str).str.lower()
    ufc_stats['fight_date'] = pd.to_datetime(ufc_stats['fight_date'])
    fighter_stats['name'] = fighter_stats['FIGHTER'].astype(str).str.lower().str.strip()
    fighter_stats['dob'] = fighter_stats['DOB'].replace(['--', '', 'NA', 'N/A'], np.nan).apply(parse_date)

    # Merge fighter stats and calculate age
    ufc_stats = pd.merge(
        ufc_stats,
        fighter_stats[['name', 'dob']],
        left_on='fighter', right_on='name',
        how='left'
    )
    ufc_stats['age'] = (ufc_stats['fight_date'] - ufc_stats['dob']).dt.days / 365.25
    ufc_stats['age'] = ufc_stats['age'].fillna(np.nan).round().astype(float)
    ufc_stats.loc[ufc_stats['age'] < 0, 'age'] = np.nan

    # Clean data and drop unwanted columns
    ufc_stats = ufc_stats.drop(['round', 'location', 'name'], axis=1)
    ufc_stats = ufc_stats[~ufc_stats['weight_class'].str.contains("Women's")]

    # Convert time strings ('MM:SS') to seconds
    ufc_stats['time'] = (
            pd.to_datetime(ufc_stats['time'], format='%M:%S').dt.minute * 60 +
            pd.to_datetime(ufc_stats['time'], format='%M:%S').dt.second
    )

    return ufc_stats


def calculate_time_based_stats(final_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional time-based statistics such as takedowns and knockdowns per 15 minutes.
    """
    final_stats['time_career_minutes'] = final_stats['time_career'] / 60  # Convert seconds to minutes
    final_stats['takedowns_per_15min'] = (final_stats['takedown_successful_career'] / final_stats[
        'time_career_minutes']) * 15
    final_stats['knockdowns_per_15min'] = (final_stats['knockdowns_career'] / final_stats['time_career_minutes']) * 15
    # Handle division by zero and infinities
    final_stats['takedowns_per_15min'] = final_stats['takedowns_per_15min'].fillna(0).replace([np.inf, -np.inf], 0)
    final_stats['knockdowns_per_15min'] = final_stats['knockdowns_per_15min'].fillna(0).replace([np.inf, -np.inf], 0)
    return final_stats


def calculate_total_fight_stats(group: pd.DataFrame) -> pd.DataFrame:
    """
    Given a fighter’s group of fights (sorted chronologically),
    calculate cumulative fight stats including total fights, wins, losses,
    and breakdowns by outcome.
    """
    group = group.sort_values('fight_date').reset_index(drop=True)
    group['total_fights'] = range(1, len(group) + 1)
    group['total_wins'] = group['winner'].cumsum()
    group['total_losses'] = group['total_fights'] - group['total_wins']

    # Initialize outcome count columns
    for outcome in ['ko', 'submission', 'decision']:
        group[f'wins_by_{outcome}'] = 0
        group[f'losses_by_{outcome}'] = 0

    # Cumulatively calculate fight outcomes
    for i in range(len(group)):
        if i > 0:
            for col in ['wins_by_ko', 'wins_by_submission', 'wins_by_decision',
                        'losses_by_ko', 'losses_by_submission', 'losses_by_decision']:
                group.loc[i, col] = group.loc[i - 1, col]
        if group.loc[i, 'winner'] == 1:  # Win
            if group.loc[i, 'result'] in [0, 3]:
                group.loc[i, 'wins_by_ko'] += 1
            elif group.loc[i, 'result'] == 1:
                group.loc[i, 'wins_by_submission'] += 1
            elif group.loc[i, 'result'] in [2, 4]:
                group.loc[i, 'wins_by_decision'] += 1
        else:  # Loss
            if group.loc[i, 'result'] in [0, 3]:
                group.loc[i, 'losses_by_ko'] += 1
            elif group.loc[i, 'result'] == 1:
                group.loc[i, 'losses_by_submission'] += 1
            elif group.loc[i, 'result'] in [2, 4]:
                group.loc[i, 'losses_by_decision'] += 1

    # Calculate percentages
    for outcome in ['ko', 'submission', 'decision']:
        group[f'win_rate_by_{outcome}'] = (group[f'wins_by_{outcome}'] / group['total_wins']).fillna(0)
        group[f'loss_rate_by_{outcome}'] = (group[f'losses_by_{outcome}'] / group['total_losses']).fillna(0)

    return group


def rename_columns_general(col: str) -> str:
    """
    Generalized column renaming: if the column contains 'fighter' (but not starting with it)
    adjust the name for clarity.
    """
    if 'fighter' in col and not col.startswith('fighter'):
        if 'b_fighter_b' in col:
            return col.replace('b_fighter_b', 'fighter_b_opponent')
        elif 'b_fighter' in col:
            return col.replace('b_fighter', 'fighter_a_opponent')
        elif 'fighter' in col and 'fighter_b' not in col:
            return col.replace('fighter', 'fighter_a')
    return col


def get_opponent(fighter, fight_id, ufc_stats):
    fight_fighters = ufc_stats[ufc_stats['id'] == fight_id]['fighter'].unique()
    if len(fight_fighters) < 2:
        return None
    return fight_fighters[0] if fight_fighters[0] != fighter else fight_fighters[1]


def aggregate_fighter_stats(group, numeric_columns):
    group = group.sort_values('fight_date')
    cumulative_stats = group[numeric_columns].cumsum(skipna=True)
    fight_count = group.groupby('fighter').cumcount() + 1

    for col in numeric_columns:
        group[f"{col}_career"] = cumulative_stats[col]
        group[f"{col}_career_avg"] = (cumulative_stats[col] / fight_count).fillna(0)

    group['significant_strikes_rate_career'] = (
            cumulative_stats['significant_strikes_landed'] / cumulative_stats['significant_strikes_attempted']).fillna(
        0)
    group['takedown_rate_career'] = (
            cumulative_stats['takedown_successful'] / cumulative_stats['takedown_attempted']).fillna(0)
    group['total_strikes_rate_career'] = (
            cumulative_stats['total_strikes_landed'] / cumulative_stats['total_strikes_attempted']).fillna(0)
    group["combined_success_rate_career"] = (group["takedown_rate_career"] + group["total_strikes_rate_career"]) / 2

    return group


def calculate_experience_and_days(group):
    group = group.sort_values('fight_date')
    group['years_of_experience'] = (group['fight_date'] - group['fight_date'].iloc[0]).dt.days / 365.25
    group['days_since_last_fight'] = (group['fight_date'] - group['fight_date'].shift()).dt.days
    return group


def update_streaks(group):
    group = group.sort_values('fight_date')

    # Initialize the win and loss streak columns with 0 for the first fight
    group['win_streak'] = 0
    group['loss_streak'] = 0

    # Iterate over each row in the group, starting from the second row
    for i in range(1, len(group)):
        if group.iloc[i - 1]['winner'] == 1:
            # If the fighter won the previous fight, increment the win streak and reset the loss streak
            group.iloc[i, group.columns.get_loc('win_streak')] = group.iloc[i - 1]['win_streak'] + 1
            group.iloc[i, group.columns.get_loc('loss_streak')] = 0
        else:
            # If the fighter lost the previous fight, increment the loss streak and reset the win streak
            group.iloc[i, group.columns.get_loc('loss_streak')] = group.iloc[i - 1]['loss_streak'] + 1
            group.iloc[i, group.columns.get_loc('win_streak')] = 0

    return group


def parse_date(date_str):
    if pd.isna(date_str):
        return pd.NaT
    try:
        return pd.to_datetime(date_str, format='%d-%b-%y')
    except ValueError:
        try:
            return pd.to_datetime(date_str, format='%b %d, %Y')
        except ValueError:
            return pd.NaT


def process_odds_data(final_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Process and merge betting odds data with fight statistics using merge_asof to match
    on the nearest date within a 1 day tolerance.

    Args:
        final_stats (pd.DataFrame): DataFrame containing fight statistics

    Returns:
        pd.DataFrame: DataFrame with merged odds data
    """
    import pandas as pd

    # Copy and remove duplicate columns
    final_stats = final_stats.copy().loc[:, ~final_stats.columns.duplicated()]

    # Read odds data
    odds_df = pd.read_csv('../data/odds data/cleaned_fight_odds.csv')

    # Standardize fighter names
    final_stats['fighter'] = final_stats['fighter'].str.lower().str.strip()
    odds_df['Matchup'] = odds_df['Matchup'].str.lower().str.strip()

    # Rename odds_df column so both DataFrames share the same key
    odds_df.rename(columns={'Matchup': 'fighter'}, inplace=True)

    print("\nSample of standardized names:")
    print("final_stats.fighter:", final_stats['fighter'].head())
    print("odds_df.fighter:", odds_df['fighter'].head())

    # Convert dates to datetime
    final_stats['fight_date'] = pd.to_datetime(final_stats['fight_date'])
    odds_df['Date'] = pd.to_datetime(odds_df['Date'], format='%Y-%m-%d')

    # Sort both DataFrames by their respective date columns (required for merge_asof)
    final_stats.sort_values('fight_date', inplace=True)
    odds_df.sort_values('Date', inplace=True)

    # Merge using merge_asof with a grouping key and tolerance of 1 day
    merged_df = pd.merge_asof(
        final_stats,
        odds_df,
        left_on='fight_date',
        right_on='Date',
        by='fighter',
        tolerance=pd.Timedelta("1D"),
        direction='nearest'
    )

    # Drop the extra Date column from the odds DataFrame
    merged_df.drop(columns=['Date'], inplace=True)

    # Rename odds columns for clarity
    merged_df.rename(
        columns={
            'Open': 'open_odds',
            'Closing Range Start': 'closing_range_start',
            'Closing Range End': 'closing_range_end',
            'Movement': 'odds_movement'
        },
        inplace=True
    )

    return merged_df


def remove_correlated_features(matchup_df, correlation_threshold=0.95):
    # Select only the numeric columns
    numeric_columns = matchup_df.select_dtypes(include=[np.number]).columns
    numeric_df = matchup_df[numeric_columns]

    # Calculate the correlation matrix
    corr_matrix = numeric_df.corr().abs()

    # Select the upper triangle of the correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find the columns to drop, excluding 'current_fight_open_odds_diff'
    columns_to_drop = [column for column in upper_tri.columns
                       if any(upper_tri[column] > correlation_threshold)
                       and column != 'current_fight_open_odds_diff' and column != 'current_fight_closing_range_end_b'
                       and column != 'current_fight_closing_odds_diff'
                       ]

    # Drop the highly correlated columns
    matchup_df = matchup_df.drop(columns=columns_to_drop)

    return matchup_df, columns_to_drop


def round_to_nearest_1(x):
    return round(x)


def calculate_complementary_odd(odd):
    if odd > 0:
        prob = 100 / (odd + 100)
    else:
        prob = abs(odd) / (abs(odd) + 100)

    complementary_prob = 1.045 - prob  # Make probabilities sum to 104.5%

    if complementary_prob >= 0.5:
        complementary_odd = -100 * complementary_prob / (1 - complementary_prob)
    else:
        complementary_odd = 100 * (1 - complementary_prob) / complementary_prob

    return round_to_nearest_1(complementary_odd)

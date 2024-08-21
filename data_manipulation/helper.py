import numpy as np
import pandas as pd
from datetime import timedelta


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


def preprocess_odds_mappings(odds_mappings):
    processed_mappings = {}
    for odds_type, mappings in odds_mappings.items():
        processed_mappings[odds_type] = {}
        for (matchup, event_name), odds_data in mappings.items():
            fighters = [str(f).strip().lower() for f in matchup.split('vs')]
            event_key = str(event_name).lower()
            odds_date = pd.to_datetime(odds_data['Date'])
            odds = odds_data['odds']

            for fighter in fighters:
                if fighter not in processed_mappings[odds_type]:
                    processed_mappings[odds_type][fighter] = []
                processed_mappings[odds_type][fighter].append((event_key, odds_date, odds))

    return processed_mappings


def get_odds_efficient(df, processed_odds_mappings, odds_columns):
    odds_values = pd.DataFrame(index=df.index, columns=odds_columns)

    for odds_type in odds_columns:
        fighter_odds = processed_odds_mappings[odds_type]

        for idx, row in df.iterrows():
            fighter = row['fighter']
            event = row['event']
            fight_date = row['fight_date']

            if isinstance(fighter, pd.Series):
                fighter = fighter.iloc[0]
            if isinstance(event, pd.Series):
                event = event.iloc[0]
            if isinstance(fight_date, pd.Series):
                fight_date = fight_date.iloc[0]

            fighter = str(fighter).lower()
            event = str(event).lower()
            fight_date = pd.to_datetime(fight_date)

            if fighter in fighter_odds:
                for event_key, odds_date, odds in fighter_odds[fighter]:
                    if (event == event_key and odds_date.year == fight_date.year) or \
                            (abs((odds_date - fight_date).days) <= 1 and odds_date.year == fight_date.year):
                        odds_values.at[idx, odds_type] = odds
                        break

    return odds_values


def process_chunk(chunk, odds_mappings, odds_columns):
    result = chunk.apply(lambda row: get_odds_efficient(row, odds_mappings, odds_columns), axis=1)
    return result, len(chunk)


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
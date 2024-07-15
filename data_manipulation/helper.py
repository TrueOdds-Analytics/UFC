from datetime import datetime
import numpy as np
import pandas as pd


def calculate_damage_score(row):
    weights = {
        'knockdowns': 15,
        'significant_strikes_landed': 5,
        'total_strikes_landed': 1,
        'head_landed': 3,
        'body_landed': 2,
        'leg_landed': 1.5,
        'distance_landed': 2,
        'clinch_landed': 1.5,
        'ground_landed': 1
    }
    return sum(row[col] * weight for col, weight in weights.items())


def get_damage_taken(group):
    if len(group) == 2:
        return pd.Series({
            group.iloc[0]['fighter']: group.iloc[1]['damage_given'],
            group.iloc[1]['fighter']: group.iloc[0]['damage_given']
        })
    elif len(group) == 1:
        return pd.Series({group.iloc[0]['fighter']: 0})  # Assume no damage taken if only one fighter
    else:
        return pd.Series()  # Return empty series for any other case


def aggregate_fighter_stats(group, numeric_columns):
    group = group.sort_values('fight_date')
    cumulative_stats = group[numeric_columns].cumsum(skipna=True)
    fight_count = group.groupby('fighter').cumcount() + 1  # Add 1 to include current fight

    for col in numeric_columns:
        group[f"{col}_career"] = cumulative_stats[col]
        group[f"{col}_career_avg"] = (cumulative_stats[col] / fight_count).fillna(0)

    group['significant_strikes_rate_career'] = (
            cumulative_stats['significant_strikes_landed'] / cumulative_stats['significant_strikes_attempted']).fillna(
        0)
    group['takedown_rate_career'] = (
            cumulative_stats['takedown_successful'] / cumulative_stats['takedown_attempted']).fillna(0)

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


def get_odds(row, odds_mappings, odds_columns):
    fighter = row['fighter'].lower() if isinstance(row['fighter'], str) else row['fighter'].iloc[0].lower()
    event = row['event'].lower() if isinstance(row['event'], str) else row['event'].iloc[0].lower()
    fight_date = row['fight_date'] if isinstance(row['fight_date'], pd.Timestamp) else row['fight_date'].iloc[0]
    fight_date = parse_date(fight_date)

    odds_values = {}

    for odds_type in odds_columns:
        for key, odds_data in odds_mappings[odds_type].items():
            if isinstance(key, tuple) and len(key) == 2:
                matchup, event_name = key
                odds = odds_data['odds']
                odds_date = parse_date(odds_data['Date'])
            else:
                continue

            matchup_lower = matchup.lower()
            event_name_lower = event_name.lower()

            if fighter in matchup_lower:
                # Check if the event names match
                if event == event_name_lower:
                    # If event names match, also check if the years match
                    if odds_date and fight_date and odds_date.year == fight_date.year:
                        odds_values[odds_type] = odds
                        break

                # If event names don't match exactly, check for partial matches
                elif odds_date and fight_date and odds_date.year == fight_date.year:
                    # Check if the dates are exactly the same
                    if odds_date == fight_date:
                        odds_values[odds_type] = odds
                        break
                    # If not exact, allow for a 1-day difference
                    elif abs((odds_date - fight_date).days) <= 1:
                        odds_values[odds_type] = odds
                        break

        if odds_type not in odds_values:
            odds_values[odds_type] = None

    return pd.Series(odds_values)


def process_chunk(chunk, odds_mappings, odds_columns):
    result = chunk.apply(lambda row: get_odds(row, odds_mappings, odds_columns), axis=1)
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
                       and column != 'current_fight_open_odds_diff']

    # Drop the highly correlated columns
    matchup_df = matchup_df.drop(columns=columns_to_drop)

    return matchup_df, columns_to_drop


def round_to_nearest_5(x):
    return round(x / 5) * 5


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

    return round_to_nearest_5(complementary_odd)

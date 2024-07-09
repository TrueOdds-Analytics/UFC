from datetime import datetime
import numpy as np
import pandas as pd


def parse_date(date_input):
    if isinstance(date_input, pd.Timestamp):
        return date_input.to_pydatetime()
    elif isinstance(date_input, str):
        date_formats = [
            "%b %d %Y", "%b %dst %Y", "%b %dnd %Y", "%b %drd %Y", "%b %dth %Y",
            "%B %d %Y", "%B %dst %Y", "%B %dnd %Y", "%B %drd %Y", "%B %dth %Y",
            "%Y-%m-%d"
        ]
        for date_format in date_formats:
            try:
                return datetime.strptime(date_input, date_format)
            except ValueError:
                continue
        print(f"Unable to parse date: {date_input}")
        return None
    elif isinstance(date_input, datetime):
        return date_input
    else:
        print(f"Unexpected date type: {type(date_input)}")
        return None


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

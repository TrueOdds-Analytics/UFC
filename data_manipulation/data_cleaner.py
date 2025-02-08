import warnings

from helper import *
from data_manipulation.Elo import *

# Suppress Future and Deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def combine_rounds_stats(file_path: str) -> pd.DataFrame:
    """
    Load, preprocess, and combine round-level UFC fight data to create a single
    DataFrame of aggregated career stats per fighter. Also calculates per-minute,
    odds, and additional fight outcome statistics.
    """
    print("Loading and preprocessing data...")
    ufc_stats = pd.read_csv(file_path)
    fighter_stats = pd.read_csv('Fight Scraping/rough_data/ufc_fighter_tott.csv')
    ufc_stats = preprocess_data(ufc_stats, fighter_stats)

    # Identify numeric columns (excluding certain IDs and pre-calculated fields)
    numeric_columns = ufc_stats.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in ['id', 'last_round', 'age']]
    if 'time' not in numeric_columns:
        numeric_columns.append('time')

    print("Aggregating stats...")
    max_round_time = ufc_stats.groupby('id').agg({'last_round': 'max', 'time': 'max'}).reset_index()
    aggregated_stats = ufc_stats.groupby(['id', 'fighter'])[numeric_columns].sum().reset_index()

    # Calculate simple rate columns
    aggregated_stats['significant_strikes_rate'] = (
            aggregated_stats['significant_strikes_landed'] / aggregated_stats['significant_strikes_attempted']
    ).fillna(0)
    aggregated_stats['takedown_rate'] = (
            aggregated_stats['takedown_successful'] / aggregated_stats['takedown_attempted']
    ).fillna(0)

    # Get non-numeric columns (and ensure unique rows per id and fighter)
    non_numeric_columns = ufc_stats.select_dtypes(exclude=['int64', 'float64']).columns.difference(
        ['id', 'fighter'])
    non_numeric_data = ufc_stats.drop_duplicates(subset=['id', 'fighter'])[
        ['id', 'fighter', 'age'] + list(non_numeric_columns)
        ]

    print("Merging aggregated stats with non-numeric data...")
    merged_stats = pd.merge(aggregated_stats, non_numeric_data, on=['id', 'fighter'], how='left')
    merged_stats = pd.merge(merged_stats, max_round_time, on='id', how='left')

    print("Calculating career stats...")
    final_stats = merged_stats.groupby('fighter', group_keys=False).apply(
        lambda x: aggregate_fighter_stats(x, numeric_columns)
    )

    print("Calculating per-minute stats...")
    final_stats['fight_duration_minutes'] = final_stats['time'] / 60
    for col in ['significant_strikes_landed', 'significant_strikes_attempted',
                'total_strikes_landed', 'total_strikes_attempted']:
        final_stats[f'{col}_per_min'] = (final_stats[col] / final_stats['fight_duration_minutes']).fillna(0)

    final_stats["total_strikes_rate"] = (
        (final_stats["total_strikes_landed"] / final_stats["total_strikes_attempted"]).fillna(0)
    )
    final_stats["combined_success_rate"] = (final_stats["takedown_rate"] + final_stats["total_strikes_rate"]) / 2

    # Reorder and filter columns
    common_columns = ufc_stats.columns.intersection(final_stats.columns)
    career_columns = [col for col in final_stats.columns if col.endswith('_career') or col.endswith('_career_avg')]
    per_minute_columns = ['significant_strikes_landed_per_min', 'significant_strikes_attempted_per_min',
                          'total_strikes_landed_per_min', 'total_strikes_attempted_per_min']
    rate_columns = ['total_strikes_rate', 'combined_success_rate']
    final_columns = ['fighter', 'age'] + list(common_columns) + career_columns + per_minute_columns + rate_columns
    final_stats = final_stats[final_columns]

    # Filter out unwanted fight results
    final_stats = final_stats[~final_stats['winner'].isin(['NC/NC', 'D/D'])]
    final_stats = final_stats[
        ~final_stats['result'].isin(['DQ', 'DQ ', 'Could Not Continue ', 'Overturned ', 'Other '])]

    # Factorize categorical columns and print the mapping
    for column in ['result', 'winner', 'scheduled_rounds']:
        final_stats[column], unique = pd.factorize(final_stats[column])
        mapping = {index: label for index, label in enumerate(unique)}
        print(f"Mapping for {column}: {mapping}")

    # Process odds data and clean up columns
    final_stats = process_odds_data(final_stats)
    columns_to_drop = ['new_Open', 'new_Closing Range Start', 'new_Closing Range End', 'new_Movement', 'dob']
    final_stats = final_stats.drop(columns=columns_to_drop, errors='ignore')
    duplicate_columns = final_stats.columns[final_stats.columns.duplicated()]
    final_stats = final_stats.loc[:, ~final_stats.columns.duplicated()]
    print(f"Dropped duplicate columns: {list(duplicate_columns)}")

    print("Calculating additional stats...")
    final_stats = final_stats.sort_values(['fighter', 'fight_date'])
    final_stats = final_stats.groupby('fighter', group_keys=False).apply(calculate_experience_and_days)
    final_stats = final_stats.groupby('fighter', group_keys=False).apply(update_streaks)
    final_stats['days_since_last_fight'] = final_stats['days_since_last_fight'].fillna(0)

    print("Calculating takedowns and knockdowns per 15 minutes...")
    final_stats = calculate_time_based_stats(final_stats)

    print("Calculating total fights, wins, and losses...")
    final_stats = final_stats.groupby('fighter', group_keys=False).apply(calculate_total_fight_stats)

    print("Saving processed data...")
    final_stats.to_csv('../data/combined_rounds.csv', index=False)

    return final_stats


def combine_fighters_stats(file_path: str) -> pd.DataFrame:
    """
    Load fighter stats, create combined and mirrored fight rows, and calculate
    differential and ratio features between paired fighters.
    """
    df = pd.read_csv(file_path)

    # Remove columns with 'event' in the title and sort for consistency
    df = df.drop(columns=[col for col in df.columns if 'event' in col.lower()])
    df = df.sort_values(by=['id', 'fighter'])

    # Group rows by fight ID
    fights_dict = {}
    for _, row in df.iterrows():
        fight_id = row['id']
        fights_dict.setdefault(fight_id, []).append(row)

    # Combine rows: original and mirrored versions
    combined_fights = []
    for fight_id, fighters in fights_dict.items():
        if len(fighters) == 2:
            fighter_1, fighter_2 = fighters
            original = pd.concat([pd.Series(fighter_1), pd.Series(fighter_2).add_suffix('_b')])
            mirrored = pd.concat([pd.Series(fighter_2), pd.Series(fighter_1).add_suffix('_b')])
            combined_fights.extend([original, mirrored])

    final_combined_df = pd.DataFrame(combined_fights).reset_index(drop=True)

    # Define columns for processing
    base_columns = [
        'knockdowns', 'significant_strikes_landed', 'significant_strikes_attempted',
        'significant_strikes_rate', 'total_strikes_landed', 'total_strikes_attempted',
        'takedown_successful', 'takedown_attempted', 'takedown_rate', 'submission_attempt',
        'reversals', 'head_landed', 'head_attempted', 'body_landed', 'body_attempted',
        'leg_landed', 'leg_attempted', 'distance_landed', 'distance_attempted',
        'clinch_landed', 'clinch_attempted', 'ground_landed', 'ground_attempted'
    ]
    other_columns = [
        'open_odds', 'closing_range_start', 'closing_range_end', 'pre_fight_elo',
        'years_of_experience', 'win_streak', 'loss_streak', 'days_since_last_fight',
        'significant_strikes_landed_per_min', 'significant_strikes_attempted_per_min',
        'total_strikes_landed_per_min', 'total_strikes_attempted_per_min', 'takedowns_per_15min',
        'knockdowns_per_15min', 'total_fights', 'total_wins', 'total_losses',
        'wins_by_ko', 'losses_by_ko', 'wins_by_submission', 'losses_by_submission', 'wins_by_decision',
        'losses_by_decision', 'win_rate_by_ko', 'loss_rate_by_ko', 'win_rate_by_submission', 'loss_rate_by_submission',
        'win_rate_by_decision', 'loss_rate_by_decision'
    ]
    columns_to_process = (
            base_columns +
            [f"{col}_career" for col in base_columns] +
            [f"{col}_career_avg" for col in base_columns] +
            other_columns
    )

    # Calculate differential and ratio columns
    diff_df = pd.DataFrame({
        f"{col}_diff": final_combined_df[col] - final_combined_df[f"{col}_b"]
        for col in columns_to_process
    })
    ratio_df = pd.DataFrame({
        f"{col}_ratio": final_combined_df[col] / final_combined_df[f"{col}_b"]
        for col in columns_to_process
    }).replace([np.inf, -np.inf, np.nan], 0)

    final_combined_df = pd.concat([final_combined_df, diff_df, ratio_df], axis=1)

    # Filter unwanted fight outcomes and sort the data
    final_combined_df = final_combined_df[~final_combined_df['winner'].isin(['NC', 'D'])]
    final_combined_df['fight_date'] = pd.to_datetime(final_combined_df['fight_date'])
    final_combined_df = final_combined_df.sort_values(by=['fighter', 'fight_date'], ascending=[True, True])

    final_combined_df.to_csv('../data/combined_sorted_fighter_stats.csv', index=False)
    return final_combined_df


def split_train_val_test(matchup_data_file: str, start_date: str, end_date: str) -> None:
    """
    Load matchup data, remove correlated features, and split it into training,
    validation, and test datasets. Also removes duplicate fights from validation
    and test sets.
    """
    matchup_df = pd.read_csv(matchup_data_file)
    matchup_df, removed_features = remove_correlated_features(matchup_df)
    matchup_df['current_fight_date'] = pd.to_datetime(matchup_df['current_fight_date'])
    start_date = pd.to_datetime(start_date)
    years_before = start_date - pd.DateOffset(years=10)

    test_data = matchup_df[
        (matchup_df['current_fight_date'] >= start_date) &
        (matchup_df['current_fight_date'] <= end_date)
        ].copy()
    remaining_data = matchup_df[
        (matchup_df['current_fight_date'] >= years_before) &
        (matchup_df['current_fight_date'] < start_date)
        ].copy()

    remaining_data = remaining_data.sort_values(by='current_fight_date', ascending=True)
    split_index = int(len(remaining_data) * 0.8)
    train_data = remaining_data.iloc[:split_index].copy()
    val_data = remaining_data.iloc[split_index:].copy()

    def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['fight_pair'] = df.apply(
            lambda row: tuple(sorted([row['fighter_a'], row['fighter_b']])), axis=1
        )
        df = df.drop_duplicates(subset=['fight_pair'], keep='first')
        return df.drop(columns=['fight_pair']).reset_index(drop=True)

    val_data = remove_duplicates(val_data)
    test_data = remove_duplicates(test_data)

    # Sort each dataset
    sort_cols = ['current_fight_date', 'fighter_a', 'fighter_b']
    train_data = train_data.sort_values(by=sort_cols, ascending=True)
    val_data = val_data.sort_values(by=sort_cols, ascending=True)
    test_data = test_data.sort_values(by=sort_cols, ascending=True)

    # Save the datasets and removed features list
    train_data.to_csv('../data/train test data/train_data.csv', index=False)
    val_data.to_csv('../data/train test data/val_data.csv', index=False)
    test_data.to_csv('../data/train test data/test_data.csv', index=False)
    with open('../data/train test data/removed_features.txt', 'w') as file:
        file.write(','.join(removed_features))

    print(
        f"Train, validation, and test data saved successfully. {len(removed_features)} correlated features were removed."
    )
    print(f"Train set size: {len(train_data)}, Validation set size: {len(val_data)}, Test set size: {len(test_data)}")


def create_matchup_data(file_path: str, tester: int, name: bool) -> pd.DataFrame:
    """
    Create matchup data by computing averages of a fighter's past fights, processing
    odds and age differences, and combining a variety of features and labels.

    Parameters:
        file_path (str): CSV file with combined fighter stats.
        tester (int): Determines the number of most recent fights to use for outcome stats.
        name (bool): If True, include fighter names in the output.
    """
    df = pd.read_csv(file_path)
    columns_to_exclude = [
        'fighter', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
        'result', 'winner', 'weight_class', 'scheduled_rounds',
        'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b'
    ]
    features_to_include = [
        col for col in df.columns if col not in columns_to_exclude and
                                     col != 'age' and not col.endswith('_age')
    ]
    method_columns = ['winner']
    n_past_fights = 6 - tester
    matchup_data = []

    # Process each fight to build the matchup feature vector
    for _, current_fight in df.iterrows():
        fighter_a_name = current_fight['fighter']
        fighter_b_name = current_fight['fighter_b']

        fighter_a_df = df[
            (df['fighter'] == fighter_a_name) &
            (df['fight_date'] < current_fight['fight_date'])
            ].sort_values(by='fight_date', ascending=False).head(n_past_fights)
        fighter_b_df = df[
            (df['fighter'] == fighter_b_name) &
            (df['fight_date'] < current_fight['fight_date'])
            ].sort_values(by='fight_date', ascending=False).head(n_past_fights)

        if len(fighter_a_df) < n_past_fights or len(fighter_b_df) < n_past_fights:
            continue

        fighter_a_features = fighter_a_df[features_to_include].head(n_past_fights).mean().values
        fighter_b_features = fighter_b_df[features_to_include].head(n_past_fights).mean().values

        results_fighter_a = fighter_a_df[['result', 'winner', 'weight_class', 'scheduled_rounds']].head(
            tester).values.flatten()
        results_fighter_b = fighter_b_df[['result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']].head(
            tester).values.flatten()

        results_fighter_a = np.pad(results_fighter_a, (0, tester * 4 - len(results_fighter_a)), 'constant',
                                   constant_values=np.nan)
        results_fighter_b = np.pad(results_fighter_b, (0, tester * 4 - len(results_fighter_b)), 'constant',
                                   constant_values=np.nan)

        labels = current_fight[method_columns].values

        # Process open odds
        if pd.notna(current_fight['open_odds']) and pd.notna(current_fight['open_odds_b']):
            current_fight_odds = [current_fight['open_odds'], current_fight['open_odds_b']]
            current_fight_odds_diff = current_fight['open_odds'] - current_fight['open_odds_b']
            current_fight_odds_ratio = (
                current_fight['open_odds'] / current_fight['open_odds_b']
                if current_fight['open_odds_b'] != 0 else 0
            )
        elif pd.notna(current_fight['open_odds']):
            odds_a = round_to_nearest_1(current_fight['open_odds'])
            odds_b = calculate_complementary_odd(odds_a)
            current_fight_odds = [odds_a, odds_b]
            current_fight_odds_diff = odds_a - odds_b
            current_fight_odds_ratio = odds_a / odds_b if odds_b != 0 else 0
        elif pd.notna(current_fight['open_odds_b']):
            odds_b = round_to_nearest_1(current_fight['open_odds_b'])
            odds_a = calculate_complementary_odd(odds_b)
            current_fight_odds = [odds_a, odds_b]
            current_fight_odds_diff = odds_a - odds_b
            current_fight_odds_ratio = odds_a / odds_b if odds_b != 0 else 0
        else:
            current_fight_odds = [-111, -111]
            current_fight_odds_diff = 0
            current_fight_odds_ratio = 1

        # Process closing range end odds
        if pd.notna(current_fight['closing_range_end']) and pd.notna(current_fight['closing_range_end_b']):
            current_fight_closing_odds = [current_fight['closing_range_end'], current_fight['closing_range_end_b']]
            current_fight_closing_odds_diff = current_fight['closing_range_end'] - current_fight['closing_range_end_b']
            current_fight_closing_odds_ratio = (
                current_fight['closing_range_end'] / current_fight['closing_range_end_b']
                if current_fight['closing_range_end_b'] != 0 else 0
            )
        elif pd.notna(current_fight['closing_range_end']):
            odds_a = round_to_nearest_1(current_fight['closing_range_end'])
            odds_b = calculate_complementary_odd(odds_a)
            current_fight_closing_odds = [odds_a, odds_b]
            current_fight_closing_odds_diff = odds_a - odds_b
            current_fight_closing_odds_ratio = odds_a / odds_b if odds_b != 0 else 0
        elif pd.notna(current_fight['closing_range_end_b']):
            odds_b = round_to_nearest_1(current_fight['closing_range_end_b'])
            odds_a = calculate_complementary_odd(odds_b)
            current_fight_closing_odds = [odds_a, odds_b]
            current_fight_closing_odds_diff = odds_a - odds_b
            current_fight_closing_odds_ratio = odds_a / odds_b if odds_b != 0 else 0
        else:
            current_fight_closing_odds = [-111, -111]
            current_fight_closing_odds_diff = 0
            current_fight_closing_odds_ratio = 1

        current_fight_ages = [current_fight['age'], current_fight['age_b']]
        current_fight_age_diff = current_fight['age'] - current_fight['age_b']
        current_fight_age_ratio = current_fight['age'] / current_fight['age_b'] if current_fight['age_b'] != 0 else 0

        # ELO stats and additional stats
        elo_stats = [
            current_fight['pre_fight_elo'], current_fight['pre_fight_elo_b'],
            current_fight['pre_fight_elo_diff'],
            1 / (1 + 10 ** ((current_fight['pre_fight_elo_b'] - current_fight['pre_fight_elo']) / 400)),
            1 / (1 + 10 ** ((current_fight['pre_fight_elo'] - current_fight['pre_fight_elo_b']) / 400)),
        ]
        elo_ratio = (
            current_fight['pre_fight_elo'] / current_fight['pre_fight_elo_b']
            if current_fight['pre_fight_elo_b'] != 0 else 0
        )

        other_stats = [
            current_fight['win_streak'], current_fight['win_streak_b'],
            current_fight['win_streak'] - current_fight['win_streak_b'],
            current_fight['win_streak'] / (current_fight['win_streak_b'] if current_fight['win_streak_b'] != 0 else 1),
            current_fight['loss_streak'], current_fight['loss_streak_b'],
            current_fight['loss_streak'] - current_fight['loss_streak_b'],
            current_fight['loss_streak'] / (
                current_fight['loss_streak_b'] if current_fight['loss_streak_b'] != 0 else 1),
            current_fight['years_of_experience'], current_fight['years_of_experience_b'],
            current_fight['years_of_experience'] - current_fight['years_of_experience_b'],
            current_fight['years_of_experience'] / (
                current_fight['years_of_experience_b'] if current_fight['years_of_experience_b'] != 0 else 1),
            current_fight['days_since_last_fight'], current_fight['days_since_last_fight_b'],
            current_fight['days_since_last_fight'] - current_fight['days_since_last_fight_b'],
            current_fight['days_since_last_fight'] / (
                current_fight['days_since_last_fight_b'] if current_fight['days_since_last_fight_b'] != 0 else 1)
        ]

        combined_features = np.concatenate([
            fighter_a_features, fighter_b_features, results_fighter_a, results_fighter_b,
            current_fight_odds, [current_fight_odds_diff, current_fight_odds_ratio],
            current_fight_closing_odds, [current_fight_closing_odds_diff, current_fight_closing_odds_ratio],
            current_fight_ages, [current_fight_age_diff, current_fight_age_ratio],
            elo_stats, [elo_ratio], other_stats
        ])
        combined_row = np.concatenate([combined_features, labels])

        most_recent_date = max(fighter_a_df['fight_date'].max(), fighter_b_df['fight_date'].max())
        current_fight_date = current_fight['fight_date']

        if not name:
            matchup_data.append([most_recent_date] + combined_row.tolist() + [current_fight_date])
        else:
            matchup_data.append(
                [fighter_a_name, fighter_b_name, most_recent_date] + combined_row.tolist() + [current_fight_date])

    # Generate column names for the new dataframe
    results_columns = []
    for i in range(1, tester + 1):
        results_columns += [
            f"result_fight_{i}", f"winner_fight_{i}", f"weight_class_fight_{i}", f"scheduled_rounds_fight_{i}",
            f"result_b_fight_{i}", f"winner_b_fight_{i}", f"weight_class_b_fight_{i}", f"scheduled_rounds_b_fight_{i}"
        ]
    new_columns = [
        'current_fight_pre_fight_elo_a', 'current_fight_pre_fight_elo_b', 'current_fight_pre_fight_elo_diff',
        'current_fight_pre_fight_elo_a_win_chance', 'current_fight_pre_fight_elo_b_win_chance',
        'current_fight_pre_fight_elo_ratio', 'current_fight_win_streak_a', 'current_fight_win_streak_b',
        'current_fight_win_streak_diff', 'current_fight_win_streak_ratio', 'current_fight_loss_streak_a',
        'current_fight_loss_streak_b', 'current_fight_loss_streak_diff', 'current_fight_loss_streak_ratio',
        'current_fight_years_experience_a', 'current_fight_years_experience_b', 'current_fight_years_experience_diff',
        'current_fight_years_experience_ratio', 'current_fight_days_since_last_a',
        'current_fight_days_since_last_b', 'current_fight_days_since_last_diff', 'current_fight_days_since_last_ratio'
    ]
    base_columns = ['fight_date'] if not name else ['fighter_a', 'fighter_b', 'fight_date']
    feature_columns = (
            [f"{feature}_fighter_avg_last_{n_past_fights}" for feature in features_to_include] +
            [f"{feature}_fighter_b_avg_last_{n_past_fights}" for feature in features_to_include]
    )
    odds_age_columns = [
        'current_fight_open_odds', 'current_fight_open_odds_b', 'current_fight_open_odds_diff',
        'current_fight_open_odds_ratio',
        'current_fight_closing_odds', 'current_fight_closing_odds_b', 'current_fight_closing_odds_diff',
        'current_fight_closing_odds_ratio',
        'current_fight_age', 'current_fight_age_b', 'current_fight_age_diff', 'current_fight_age_ratio'
    ]
    column_names = (
            base_columns + feature_columns + results_columns + odds_age_columns + new_columns +
            [f"{method}" for method in method_columns] + ['current_fight_date']
    )
    matchup_df = pd.DataFrame(matchup_data, columns=column_names)
    matchup_df = matchup_df.drop(columns=['fight_date'], errors='ignore')
    matchup_df.columns = [rename_columns_general(col) for col in matchup_df.columns]

    # Calculate additional differential and ratio columns for fighter features
    diff_columns = {}
    ratio_columns = {}
    for feature in features_to_include:
        col_a = f"{feature}_fighter_a_avg_last_{n_past_fights}"
        col_b = f"{feature}_fighter_b_avg_last_{n_past_fights}"
        if col_a in matchup_df.columns and col_b in matchup_df.columns:
            diff_columns[f"matchup_{feature}_diff_avg_last_{n_past_fights}"] = matchup_df[col_a] - matchup_df[col_b]
            ratio_columns[f"matchup_{feature}_ratio_avg_last_{n_past_fights}"] = matchup_df[col_a] / matchup_df[
                col_b].replace(0, 1)
    matchup_df = pd.concat([matchup_df, pd.DataFrame(diff_columns), pd.DataFrame(ratio_columns)], axis=1)

    output_filename = f'../data/matchup data/matchup_data_{n_past_fights}_avg{"_name" if name else ""}.csv'
    matchup_df.to_csv(output_filename, index=False)
    return matchup_df


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    combined_rounds = combine_rounds_stats('../data/ufc_fight_processed.csv')
    calculate_elo_ratings('../data/combined_rounds.csv')
    combine_fighters_stats("../data/combined_rounds.csv")
    create_matchup_data("../data/combined_sorted_fighter_stats.csv", 3, True)
    split_train_val_test('../data/matchup data/matchup_data_3_avg_name.csv', '2024-01-01', '2024-12-31')

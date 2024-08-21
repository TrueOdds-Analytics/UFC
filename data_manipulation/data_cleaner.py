import warnings
from helper import *
from data_manipulation.Elo import *
import time

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def combine_rounds_stats(file_path):
    print("Loading and preprocessing data...")
    ufc_stats = pd.read_csv(file_path)
    fighter_stats = pd.read_csv('../data/general data/fighter_stats.csv')

    # Preprocessing
    ufc_stats['fighter'] = ufc_stats['fighter'].astype(str).str.lower()
    ufc_stats['fight_date'] = pd.to_datetime(ufc_stats['fight_date'])
    fighter_stats['name'] = fighter_stats['name'].astype(str).str.lower().str.strip()
    fighter_stats['dob'] = fighter_stats['dob'].replace(['--', '', 'NA', 'N/A'], np.nan).apply(parse_date)

    # Merge fighter stats
    ufc_stats = pd.merge(ufc_stats, fighter_stats[['name', 'dob']], left_on='fighter', right_on='name', how='left')
    ufc_stats['age'] = (ufc_stats['fight_date'] - ufc_stats['dob']).dt.days / 365.25
    ufc_stats['age'] = ufc_stats['age'].fillna(np.nan).round().astype(float)

    # Set negative age values to NaN
    ufc_stats.loc[ufc_stats['age'] < 0, 'age'] = np.nan

    # Data cleaning
    ufc_stats = ufc_stats.drop(['round', 'location', 'name'], axis=1)
    ufc_stats = ufc_stats[~ufc_stats['weight_class'].str.contains("Women's")]

    # Convert time to seconds
    ufc_stats['time'] = pd.to_datetime(ufc_stats['time'], format='%M:%S').dt.second + \
                        pd.to_datetime(ufc_stats['time'], format='%M:%S').dt.minute * 60

    fighter_identifier = 'fighter'

    # Identify numeric columns including 'time'
    numeric_columns = ufc_stats.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in ['id', 'last_round', 'age']]
    if 'time' not in numeric_columns:
        numeric_columns.append('time')

    print("Aggregating stats...")
    max_round_time = ufc_stats.groupby('id').agg({'last_round': 'max', 'time': 'max'}).reset_index()
    aggregated_stats = ufc_stats.groupby(['id', fighter_identifier])[numeric_columns].sum().reset_index()

    # Calculate rates
    aggregated_stats['significant_strikes_rate'] = (aggregated_stats['significant_strikes_landed'] /
                                                    aggregated_stats['significant_strikes_attempted']).fillna(0)
    aggregated_stats['takedown_rate'] = (aggregated_stats['takedown_successful'] /
                                         aggregated_stats['takedown_attempted']).fillna(0)

    non_numeric_columns = ufc_stats.select_dtypes(exclude=['int64', 'float64']).columns.difference(
        ['id', fighter_identifier])
    non_numeric_data = ufc_stats.drop_duplicates(subset=['id', fighter_identifier])[
        ['id', fighter_identifier, 'age'] + list(non_numeric_columns)]

    print("Merging data...")
    merged_stats = pd.merge(aggregated_stats, non_numeric_data, on=['id', fighter_identifier], how='left')
    merged_stats = pd.merge(merged_stats, max_round_time, on='id', how='left')

    print("Calculating career stats...")
    print("Numeric columns:", numeric_columns)
    print("Columns in merged_stats:", merged_stats.columns.tolist())

    final_stats = merged_stats.groupby(fighter_identifier, group_keys=False).apply(
        lambda x: aggregate_fighter_stats(x, numeric_columns))

    # Calculate per-minute stats
    print("Calculating per-minute stats...")
    final_stats['fight_duration_minutes'] = final_stats['time'] / 60
    final_stats['significant_strikes_landed_per_min'] = (
            final_stats['significant_strikes_landed'] / final_stats['fight_duration_minutes']).fillna(0)
    final_stats['significant_strikes_attempted_per_min'] = (
            final_stats['significant_strikes_attempted'] / final_stats['fight_duration_minutes']).fillna(0)
    final_stats['total_strikes_landed_per_min'] = (
            final_stats['total_strikes_landed'] / final_stats['fight_duration_minutes']).fillna(0)
    final_stats['total_strikes_attempted_per_min'] = (
            final_stats['total_strikes_attempted'] / final_stats['fight_duration_minutes']).fillna(0)

    # More rate calculations
    final_stats["total_strikes_rate"] = (
        (final_stats["total_strikes_landed"] / final_stats["total_strikes_attempted"]).fillna(0))

    final_stats["combined_success_rate"] = (final_stats["takedown_rate"] + final_stats["total_strikes_rate"]) / 2

    common_columns = ufc_stats.columns.intersection(final_stats.columns)
    career_columns = [col for col in final_stats.columns if col.endswith('_career') or col.endswith('_career_avg')]
    per_minute_columns = ['significant_strikes_landed_per_min', 'significant_strikes_attempted_per_min',
                          'total_strikes_landed_per_min', 'total_strikes_attempted_per_min']
    rate_columns = ['total_strikes_rate', 'combined_success_rate']
    final_columns = ['fighter', 'age'] + list(common_columns) + career_columns + per_minute_columns + rate_columns
    final_stats = final_stats[final_columns]

    final_stats = final_stats[~final_stats['winner'].isin(['NC/NC', 'D/D'])]
    final_stats = final_stats[
        ~final_stats['result'].isin(['DQ', 'Decision - Split ', 'DQ ', 'Could Not Continue ', 'Overturned ', 'Other '])]

    for column in ['result', 'winner', 'scheduled_rounds']:
        final_stats[column], unique = pd.factorize(final_stats[column])
        mapping = {index: label for index, label in enumerate(unique)}
        print(f"Mapping for {column}: {mapping}")

    print("Processing odds data...")
    cleaned_odds_df = pd.read_csv('../data/odds data/cleaned_fight_odds.csv')
    odds_columns = ['Open', 'Closing Range Start', 'Closing Range End', 'Movement']
    odds_mappings = {
        col: cleaned_odds_df.set_index(['Matchup', 'Event']).apply(lambda x: {'odds': x[col], 'Date': x['Date']},
                                                                   axis=1).to_dict() for col in odds_columns
    }

    processed_odds_mappings = preprocess_odds_mappings(odds_mappings)
    new_odds = get_odds_efficient(final_stats, processed_odds_mappings, odds_columns)

    for col in odds_columns:
        final_stats[f'new_{col}'] = new_odds[col]

    final_stats['open_odds'] = final_stats['new_Open']
    final_stats['closing_range_start'] = final_stats['new_Closing Range Start']
    final_stats['closing_range_end'] = final_stats['new_Closing Range End']
    final_stats['movement'] = final_stats['new_Movement']

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

    # Calculate takedowns and knockdowns per 15 minutes
    print("Calculating takedowns and knockdowns per 15 minutes...")
    final_stats['time_career_minutes'] = final_stats['time_career'] / 60  # Convert seconds to minutes
    final_stats['takedowns_per_15min'] = (final_stats['takedown_successful_career'] / final_stats['time_career_minutes']) * 15
    final_stats['knockdowns_per_15min'] = (final_stats['knockdowns_career'] / final_stats['time_career_minutes']) * 15

    # Handle potential division by zero
    final_stats['takedowns_per_15min'] = final_stats['takedowns_per_15min'].fillna(0).replace([np.inf, -np.inf], 0)
    final_stats['knockdowns_per_15min'] = final_stats['knockdowns_per_15min'].fillna(0).replace([np.inf, -np.inf], 0)

    # Calculate total fights, wins, and losses
    print("Calculating total fights, wins, and losses...")
    print("Calculating total fights, wins, losses, and fight outcomes...")

    def calculate_fight_stats(group):
        group = group.sort_values('fight_date').reset_index(drop=True)
        group['total_fights'] = range(1, len(group) + 1)
        group['total_wins'] = group['winner'].cumsum()
        group['total_losses'] = group['total_fights'] - group['total_wins']

        # Initialize new columns
        for outcome in ['ko', 'submission', 'decision']:
            group[f'wins_by_{outcome}'] = 0
            group[f'losses_by_{outcome}'] = 0

        # Cumulative calculation of fight outcomes
        for i in range(len(group)):
            if i > 0:  # Copy previous values for cumulative count
                for col in ['wins_by_ko', 'wins_by_submission', 'wins_by_decision',
                            'losses_by_ko', 'losses_by_submission', 'losses_by_decision']:
                    group.loc[i, col] = group.loc[i - 1, col]

            if group.loc[i, 'winner'] == 1:  # Win
                if group.loc[i, 'result'] in [0, 3]:  # KO/TKO or TKO - Doctor's Stoppage
                    group.loc[i, 'wins_by_ko'] += 1
                elif group.loc[i, 'result'] == 1:  # Submission
                    group.loc[i, 'wins_by_submission'] += 1
                elif group.loc[i, 'result'] in [2, 4]:  # Decision - Unanimous or Decision - Majority
                    group.loc[i, 'wins_by_decision'] += 1
            else:  # Loss
                if group.loc[i, 'result'] in [0, 3]:  # KO/TKO or TKO - Doctor's Stoppage
                    group.loc[i, 'losses_by_ko'] += 1
                elif group.loc[i, 'result'] == 1:  # Submission
                    group.loc[i, 'losses_by_submission'] += 1
                elif group.loc[i, 'result'] in [2, 4]:  # Decision - Unanimous or Decision - Majority
                    group.loc[i, 'losses_by_decision'] += 1

        # Calculate percentages
        for outcome in ['ko', 'submission', 'decision']:
            group[f'win_rate_by_{outcome}'] = (group[f'wins_by_{outcome}'] / group['total_wins']).fillna(0)
            group[f'loss_rate_by_{outcome}'] = (group[f'losses_by_{outcome}'] / group['total_losses']).fillna(0)

        return group

    final_stats = final_stats.groupby('fighter', group_keys=False).apply(calculate_fight_stats)

    print("Saving processed data...")
    final_stats.to_csv('../data/combined_rounds.csv', index=False)

    return final_stats


def combine_fighters_stats(file_path):
    # Load the data
    df = pd.read_csv(file_path)

    # Drop columns with 'event' in the title
    df = df.drop(columns=[col for col in df.columns if 'event' in col.lower()])

    # Sort by fight ID and fighter identifier to ensure consistent ordering
    df = df.sort_values(by=['id', 'fighter'])

    # Create a dictionary to store fights by ID
    fights_dict = {}

    # Group fights by ID
    for _, row in df.iterrows():
        fight_id = row['id']
        if fight_id not in fights_dict:
            fights_dict[fight_id] = []
        fights_dict[fight_id].append(row)

    # Combine fights and create mirrored versions
    combined_fights = []
    for fight_id, fighters in fights_dict.items():
        if len(fighters) == 2:
            fighter_1, fighter_2 = fighters

            # Original combination
            combined_fight = pd.concat([pd.Series(fighter_1), pd.Series(fighter_2).add_suffix('_b')])
            combined_fights.append(combined_fight)

            # Mirrored combination
            mirrored_fight = pd.concat([pd.Series(fighter_2), pd.Series(fighter_1).add_suffix('_b')])
            combined_fights.append(mirrored_fight)

    # Create the final combined DataFrame
    final_combined_df = pd.DataFrame(combined_fights)

    # Reset the index
    final_combined_df = final_combined_df.reset_index(drop=True)

    # Define the base columns to differentiate and calculate ratios
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

    # Generate the columns to differentiate and calculate ratios using list comprehension
    columns_to_process = base_columns + [f"{col}_career" for col in base_columns] + [f"{col}_career_avg" for col in base_columns] + other_columns

    # Calculate the differential and ratio for each column and store in new DataFrames
    diff_df = pd.DataFrame({f"{col}_diff": final_combined_df[col] - final_combined_df[f"{col}_b"] for col in columns_to_process})
    ratio_df = pd.DataFrame({f"{col}_ratio": final_combined_df[col] / final_combined_df[f"{col}_b"] for col in columns_to_process})

    # Replace infinity and NaN values in ratio_df with 0
    ratio_df = ratio_df.replace([np.inf, -np.inf, np.nan], 0)

    # Concatenate the differential and ratio DataFrames with the final_combined_df
    final_combined_df = pd.concat([final_combined_df, diff_df, ratio_df], axis=1)

    # Filter out rows where the winner column contains 'NC' or 'D'
    final_combined_df = final_combined_df[~final_combined_df['winner'].isin(['NC', 'D'])]

    # Convert the 'fight_date' column to datetime
    final_combined_df['fight_date'] = pd.to_datetime(final_combined_df['fight_date'])

    # Sort the DataFrame first by 'fighter_name' alphabetically, then by 'fight_date' descending (most recent first)
    final_combined_df = final_combined_df.sort_values(by=['fighter', 'fight_date'], ascending=[True, True])

    # Save the combined and sorted DataFrame to a CSV file
    final_combined_df.to_csv('../data/combined_sorted_fighter_stats.csv', index=False)

    return final_combined_df


def split_train_val_test(matchup_data_file, start_date, end_date):
    # Load the matchup data
    matchup_df = pd.read_csv(matchup_data_file)

    # Remove correlated features
    matchup_df, removed_features = remove_correlated_features(matchup_df)

    # Ensure 'current_fight_date' is in datetime format
    matchup_df['current_fight_date'] = pd.to_datetime(matchup_df['current_fight_date'])

    # Convert start_date to datetime
    start_date = pd.to_datetime(start_date)

    # Calculate the date 10 years before the start date
    years_before = start_date - pd.DateOffset(years=10)

    test_data = matchup_df[(matchup_df['current_fight_date'] >= start_date) &
                           (matchup_df['current_fight_date'] <= end_date)].copy()

    remaining_data = matchup_df[(matchup_df['current_fight_date'] >= years_before) &
                                (matchup_df['current_fight_date'] < start_date)].copy()

    # Sort remaining data by current_fight_date
    remaining_data = remaining_data.sort_values(by='current_fight_date', ascending=True)

    # Calculate the index to split the remaining data into train and validation sets
    split_index = int(len(remaining_data) * 0.8)

    # Split the remaining data into train and validation sets
    train_data = remaining_data.iloc[:split_index].copy()
    val_data = remaining_data.iloc[split_index:].copy()

    # Remove duplicate fights from validation and test data
    def remove_duplicates(df):
        df = df.copy()
        df['fight_pair'] = df.apply(lambda row: tuple(sorted([row['fighter_a'], row['fighter_b']])), axis=1)
        df = df.drop_duplicates(subset=['fight_pair'], keep='first')
        df = df.drop(columns=['fight_pair'])
        return df.reset_index(drop=True)

    val_data = remove_duplicates(val_data)
    test_data = remove_duplicates(test_data)

    # Sort train, validation, and test data by current_fight_date and fighters
    train_data = train_data.sort_values(by=['current_fight_date', 'fighter_a', 'fighter_b'], ascending=[True, True, True])
    val_data = val_data.sort_values(by=['current_fight_date', 'fighter_a', 'fighter_b'], ascending=[True, True, True])
    test_data = test_data.sort_values(by=['current_fight_date', 'fighter_a', 'fighter_b'], ascending=[True, True, True])

    # Save the train, validation, and test data to CSV files
    train_data.to_csv('../data/train test data/train_data.csv', index=False)
    val_data.to_csv('../data/train test data/val_data.csv', index=False)
    test_data.to_csv('../data/train test data/test_data.csv', index=False)

    # Save the removed features to a file
    with open('../data/train test data/removed_features.txt', 'w') as file:
        file.write(','.join(removed_features))

    print(
        f"Train, validation, and test data saved successfully. {len(removed_features)} correlated features were removed.")
    print(f"Train set size: {len(train_data)}, Validation set size: {len(val_data)}, Test set size: {len(test_data)}")


def create_matchup_data(file_path, tester, name):
    df = pd.read_csv(file_path)
    columns_to_exclude = ['fighter', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
                          'result', 'winner', 'weight_class', 'scheduled_rounds',
                          'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']

    features_to_include = [col for col in df.columns if
                           col not in columns_to_exclude and col != 'age' and not col.endswith('_age')]

    method_columns = ['winner']
    n_past_fights = 6 - tester
    matchup_data = []

    for index, current_fight in df.iterrows():
        fighter_a_name = current_fight['fighter']
        fighter_b_name = current_fight['fighter_b']

        fighter_a_df = df[(df['fighter'] == fighter_a_name) & (df['fight_date'] < current_fight['fight_date'])] \
            .sort_values(by='fight_date', ascending=False).head(n_past_fights)
        fighter_b_df = df[(df['fighter'] == fighter_b_name) & (df['fight_date'] < current_fight['fight_date'])] \
            .sort_values(by='fight_date', ascending=False).head(n_past_fights)

        if len(fighter_a_df) < n_past_fights or len(fighter_b_df) < n_past_fights:
            continue

        fighter_a_features = fighter_a_df.head(n_past_fights)[features_to_include].mean().values
        fighter_b_features = fighter_b_df.head(n_past_fights)[features_to_include].mean().values

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
            current_fight_odds_ratio = current_fight['open_odds'] / current_fight['open_odds_b'] if current_fight[
                                                                                                        'open_odds_b'] != 0 else 0
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
            current_fight_odds = [-110, -110]
            current_fight_odds_diff = 0
            current_fight_odds_ratio = 1

        # Process closing range end odds
        if pd.notna(current_fight['closing_range_end']) and pd.notna(current_fight['closing_range_end_b']):
            current_fight_closing_odds = [current_fight['closing_range_end'], current_fight['closing_range_end_b']]
            current_fight_closing_odds_diff = current_fight['closing_range_end'] - current_fight['closing_range_end_b']
            current_fight_closing_odds_ratio = current_fight['closing_range_end'] / current_fight['closing_range_end_b'] if current_fight['closing_range_end_b'] != 0 else 0
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
            current_fight_closing_odds = [-110, -110]
            current_fight_closing_odds_diff = 0
            current_fight_closing_odds_ratio = 1

        current_fight_ages = [current_fight['age'], current_fight['age_b']]
        current_fight_age_diff = current_fight['age'] - current_fight['age_b']
        current_fight_age_ratio = current_fight['age'] / current_fight['age_b'] if current_fight['age_b'] != 0 else 0

        # ELO and other stats
        elo_stats = [
            current_fight['pre_fight_elo'], current_fight['pre_fight_elo_b'],
            current_fight['pre_fight_elo_diff'],
            1 / (1 + 10 ** ((current_fight['pre_fight_elo_b'] - current_fight['pre_fight_elo']) / 400)),
            1 / (1 + 10 ** ((current_fight['pre_fight_elo'] - current_fight['pre_fight_elo_b']) / 400)),
        ]
        elo_ratio = current_fight['pre_fight_elo'] / current_fight['pre_fight_elo_b'] if current_fight[
                                                                                             'pre_fight_elo_b'] != 0 else 0

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

    # Generate column names
    results_columns = []
    for i in range(1, tester + 1):
        results_columns += [f"result_fight_{i}", f"winner_fight_{i}", f"weight_class_fight_{i}",
                            f"scheduled_rounds_fight_{i}",
                            f"result_b_fight_{i}", f"winner_b_fight_{i}", f"weight_class_b_fight_{i}",
                            f"scheduled_rounds_b_fight_{i}"]

    new_columns = [
        'current_fight_pre_fight_elo_a', 'current_fight_pre_fight_elo_b', 'current_fight_pre_fight_elo_diff',
        'current_fight_pre_fight_elo_a_win_chance', 'current_fight_pre_fight_elo_b_win_chance',
        'current_fight_pre_fight_elo_ratio',
        'current_fight_win_streak_a', 'current_fight_win_streak_b', 'current_fight_win_streak_diff',
        'current_fight_win_streak_ratio',
        'current_fight_loss_streak_a', 'current_fight_loss_streak_b', 'current_fight_loss_streak_diff',
        'current_fight_loss_streak_ratio',
        'current_fight_years_experience_a', 'current_fight_years_experience_b', 'current_fight_years_experience_diff',
        'current_fight_years_experience_ratio',
        'current_fight_days_since_last_a', 'current_fight_days_since_last_b', 'current_fight_days_since_last_diff',
        'current_fight_days_since_last_ratio'
    ]

    base_columns = ['fight_date'] if not name else ['fighter_a', 'fighter_b', 'fight_date']
    feature_columns = [f"{feature}_fighter_avg_last_{n_past_fights}" for feature in features_to_include] + \
                      [f"{feature}_fighter_b_avg_last_{n_past_fights}" for feature in features_to_include]
    odds_age_columns = ['current_fight_open_odds', 'current_fight_open_odds_b', 'current_fight_open_odds_diff',
                        'current_fight_open_odds_ratio',
                        'current_fight_closing_odds', 'current_fight_closing_odds_b', 'current_fight_closing_odds_diff',
                        'current_fight_closing_odds_ratio',
                        'current_fight_age', 'current_fight_age_b', 'current_fight_age_diff',
                        'current_fight_age_ratio']

    column_names = base_columns + feature_columns + results_columns + odds_age_columns + new_columns + \
                   [f"{method}" for method in method_columns] + ['current_fight_date']

    matchup_df = pd.DataFrame(matchup_data, columns=column_names)
    columns_to_drop = ['fight_date']
    matchup_df = matchup_df.drop(columns=columns_to_drop, errors='ignore')

    # Generalized column renaming
    def rename_column(col):
        if 'fighter' in col and not col.startswith('fighter'):
            if 'b_fighter_b' in col:
                return col.replace('b_fighter_b', 'fighter_b_opponent')
            elif 'b_fighter' in col:
                return col.replace('b_fighter', 'fighter_a_opponent')
            elif 'fighter' in col and 'fighter_b' not in col:
                return col.replace('fighter', 'fighter_a')
        return col

    matchup_df.columns = [rename_column(col) for col in matchup_df.columns]

    diff_columns = {}
    ratio_columns = {}

    for feature in features_to_include:
        col_a = f"{feature}_fighter_a_avg_last_{n_past_fights}"
        col_b = f"{feature}_fighter_b_avg_last_{n_past_fights}"

        if col_a in matchup_df.columns and col_b in matchup_df.columns:
            diff_columns[f"matchup_{feature}_diff_avg_last_{n_past_fights}"] = matchup_df[col_a] - matchup_df[col_b]
            ratio_columns[f"matchup_{feature}_ratio_avg_last_{n_past_fights}"] = matchup_df[col_a] / matchup_df[
                col_b].replace(0, 1)

    # Add all new columns at once
    matchup_df = pd.concat([matchup_df, pd.DataFrame(diff_columns), pd.DataFrame(ratio_columns)], axis=1)

    output_filename = f'../data/matchup data/matchup_data_{n_past_fights}_avg{"_name" if name else ""}.csv'
    matchup_df.to_csv(output_filename, index=False)

    return matchup_df


if __name__ == "__main__":
    combine_rounds_stats('../data/ufc_fight_processed.csv')
    calculate_elo_ratings('../data/combined_rounds.csv')
    combine_fighters_stats("../data/combined_rounds.csv")
    create_matchup_data("../data/combined_sorted_fighter_stats.csv", 3, True)
    split_train_val_test('../data/matchup data/matchup_data_3_avg_name.csv', '2024-01-01', '2024-08-30')

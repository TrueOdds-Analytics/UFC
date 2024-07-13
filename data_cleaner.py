import multiprocessing
from functools import partial
import time
import warnings
from tqdm import tqdm
from helper import *
from Elo import *

warnings.filterwarnings("ignore", category=FutureWarning)


def combine_rounds_stats(file_path):
    # Load UFC stats and fighter stats
    ufc_stats = pd.read_csv(file_path)
    fighter_stats = pd.read_csv('data/general data/fighter_stats.csv')

    # Preprocess UFC stats
    ufc_stats['fighter'] = ufc_stats['fighter'].astype(str).str.lower()
    ufc_stats['fight_date'] = pd.to_datetime(ufc_stats['fight_date'], format='%Y-%m-%d')

    # Preprocess fighter stats
    fighter_stats['name'] = fighter_stats['name'].astype(str).str.lower().str.strip()
    fighter_stats['dob'] = fighter_stats['dob'].replace(['--', '', 'NA', 'N/A'], np.nan)
    fighter_stats['dob'] = pd.to_datetime(fighter_stats['dob'], format='%b %d, %Y', errors='coerce')

    # Merge other fighter stats with UFC stats
    ufc_stats = pd.merge(ufc_stats, fighter_stats[['name', 'dob']], left_on='fighter', right_on='name', how='left')

    # Calculate age
    ufc_stats['dob'] = pd.to_datetime(ufc_stats['dob'], errors='coerce')
    ufc_stats['age'] = (ufc_stats['fight_date'] - ufc_stats['dob']).dt.days / 365.25
    ufc_stats['age'] = ufc_stats['age'].fillna(np.nan).round().astype(float)

    # Drop unnecessary columns and rows
    ufc_stats = ufc_stats.drop(['round', 'location', 'name'], axis=1)
    ufc_stats = ufc_stats[~ufc_stats['weight_class'].str.contains("Women's")]

    # Identify numeric columns for aggregation
    numeric_columns = ufc_stats.select_dtypes(include='number').columns
    numeric_columns = numeric_columns.drop(['id', 'last_round', 'age'])

    fighter_identifier = 'fighter'

    # Convert 'time' column to seconds
    ufc_stats['time'] = pd.to_datetime(ufc_stats['time'], format='%M:%S').dt.second + \
                        pd.to_datetime(ufc_stats['time'], format='%M:%S').dt.minute * 60

    # Find the highest round and time for each fight ID
    max_round_time = ufc_stats.groupby('id').agg({'last_round': 'max', 'time': 'max'}).reset_index()

    # Aggregate numeric columns by fight ID and fighter identifier
    aggregated_stats = ufc_stats.groupby(['id', fighter_identifier], as_index=False)[numeric_columns].sum()

    # Recalculate percentage columns
    aggregated_stats['significant_strikes_rate'] = (aggregated_stats['significant_strikes_landed'] / aggregated_stats[
        'significant_strikes_attempted']).fillna(0)
    aggregated_stats['takedown_rate'] = (
                aggregated_stats['takedown_successful'] / aggregated_stats['takedown_attempted']).fillna(0)

    # Extract non-numeric columns and find unique rows
    non_numeric_columns = ufc_stats.select_dtypes(exclude='number').columns.difference(['id', fighter_identifier])
    non_numeric_data = ufc_stats.drop_duplicates(subset=['id', fighter_identifier])[
        ['id', fighter_identifier, 'age'] + list(non_numeric_columns)]

    # Merge the aggregated numeric and non-numeric data
    merged_stats = pd.merge(aggregated_stats, non_numeric_data, on=['id', fighter_identifier], how='left')
    merged_stats = pd.merge(merged_stats, max_round_time, on='id', how='left')

    # Function to cumulatively sum numeric stats and create career stats columns
    def aggregate_up_to_date(group):
        group = group.sort_values('fight_date')
        cumulative_stats = group[numeric_columns].cumsum(skipna=True)
        fight_count = group.groupby('fighter').cumcount() + 1

        for col in numeric_columns:
            group[f"{col}_career"] = cumulative_stats[col]
            group[f"{col}_career_avg"] = group[f"{col}_career"] / fight_count

        group['significant_strikes_rate_career'] = (
                    group['significant_strikes_landed_career'] / group['significant_strikes_attempted_career']).fillna(
            0)
        group['takedown_rate_career'] = (
                    group['takedown_successful_career'] / group['takedown_attempted_career']).fillna(0)

        return group

    # Apply the aggregation function for each fighter up to each fight date
    final_stats = merged_stats.groupby(fighter_identifier, group_keys=False).apply(aggregate_up_to_date)

    # Reorder columns
    common_columns = ufc_stats.columns.intersection(final_stats.columns)
    career_columns = [col for col in final_stats.columns if col.endswith('_career') or col.endswith('_career_avg')]
    final_columns = ['fighter', 'age'] + list(common_columns) + career_columns
    final_stats = final_stats[final_columns]

    final_stats = final_stats[~final_stats['winner'].isin(['NC/NC', 'D/D'])]
    final_stats = final_stats[
        ~final_stats['result'].isin(['DQ', 'Decision - Split ', 'DQ ', 'Could Not Continue ', 'Overturned ', 'Other '])]

    # Consolidate weight classes
    weight_class_mapping = {
        'Flyweight': 'Flyweight', 'Bantamweight': 'Bantamweight', 'Featherweight': 'Featherweight',
        'Lightweight': 'Lightweight', 'Welterweight': 'Welterweight', 'Middleweight': 'Middleweight',
        'Light Heavyweight': 'Light Heavyweight', 'Heavyweight': 'Heavyweight', 'Tournament': 'Tournament'
    }

    final_stats['weight_class'] = final_stats['weight_class'].apply(
        lambda x: next((v for k, v in weight_class_mapping.items() if k in str(x)), x))

    # Convert unique strings to integers and create dictionary mappings
    for column in ['result', 'winner', 'weight_class', 'scheduled_rounds']:
        final_stats[column], unique = pd.factorize(final_stats[column])
        mapping = {index: label for index, label in enumerate(unique)}
        print(f"Mapping for {column}: {mapping}")

    # Load the cleaned fight odds data
    cleaned_odds_df = pd.read_csv('data/odds data/cleaned_fight_odds.csv')

    # Create mapping dictionaries for each odds column
    odds_columns = ['Open', 'Closing Range Start', 'Closing Range End', 'Movement']
    odds_mappings = {
        col: cleaned_odds_df.set_index(['Matchup', 'Event']).apply(lambda x: {'odds': x[col], 'Date': x['Date']},
                                                                   axis=1).to_dict() for col in odds_columns}

    # Split the dataframe into chunks
    num_cores = multiprocessing.cpu_count()
    chunks = np.array_split(final_stats, num_cores)

    # Create a partial function with fixed arguments
    partial_process = partial(process_chunk, odds_mappings=odds_mappings, odds_columns=odds_columns)

    # Use multiprocessing to process chunks in parallel
    total_rows = len(final_stats)
    print(f"Processing {total_rows} rows...")

    start_time = time.time()
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = []
        with tqdm(total=total_rows, desc="Processing", unit="row") as pbar:
            for chunk_result, chunk_size in pool.imap(partial_process, chunks):
                results.append(chunk_result)
                pbar.update(chunk_size)

    new_odds = pd.concat(results)

    end_time = time.time()
    print(f"\nOdds matching completed in {end_time - start_time:.2f} seconds.")

    # Add the new columns to final_stats
    for col in odds_columns:
        final_stats[f'new_{col}'] = new_odds[col]

    # Update the 'open_odds' column with the new odds and add new columns
    final_stats['open_odds'] = final_stats['new_Open']
    final_stats['closing_range_start'] = final_stats['new_Closing Range Start']
    final_stats['closing_range_end'] = final_stats['new_Closing Range End']
    final_stats['movement'] = final_stats['new_Movement']

    # Drop temporary columns
    columns_to_drop = ['new_Open', 'new_Closing Range Start', 'new_Closing Range End', 'new_Movement', 'dob']
    final_stats = final_stats.drop(columns=columns_to_drop, errors='ignore')

    # Identify and drop duplicate columns
    duplicate_columns = final_stats.columns[final_stats.columns.duplicated()]
    final_stats = final_stats.loc[:, ~final_stats.columns.duplicated()]

    print(f"Dropped duplicate columns: {list(duplicate_columns)}")

    # Sort the dataframe by fighter and fight date
    final_stats = final_stats.sort_values(['fighter', 'fight_date'])

    # Calculate years of experience and days since last fight
    def calculate_experience_and_days(group):
        group = group.sort_values('fight_date')
        group['years_of_experience'] = (group['fight_date'] - group['fight_date'].iloc[0]).dt.days / 365.25
        group['days_since_last_fight'] = (group['fight_date'] - group['fight_date'].shift()).dt.days
        return group

    final_stats = final_stats.groupby('fighter', group_keys=False).apply(calculate_experience_and_days)

    # Calculate win and loss streaks for the next fight
    def update_streaks(group):
        group = group.sort_values('fight_date')
        group['win_streak'] = group['winner'].shift().cumsum().fillna(0).astype(int)
        group['loss_streak'] = (1 - group['winner']).shift().cumsum().fillna(0).astype(int)

        # Reset streaks after a loss or win respectively
        group.loc[group['winner'].shift() == 0, 'win_streak'] = 0
        group.loc[group['winner'].shift() == 1, 'loss_streak'] = 0

        return group

    final_stats = final_stats.groupby('fighter', group_keys=False).apply(update_streaks)

    # Fill NaN values in the new columns
    final_stats['days_since_last_fight'] = final_stats['days_since_last_fight'].fillna(0)

    # Save to new CSV
    final_stats.to_csv('data/combined_rounds.csv', index=False)

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

    # Define the base columns to differentiate
    base_columns = [
        'knockdowns', 'significant_strikes_landed', 'significant_strikes_attempted',
        'significant_strikes_rate', 'total_strikes_landed', 'total_strikes_attempted',
        'takedown_successful', 'takedown_attempted', 'takedown_rate', 'submission_attempt',
        'reversals', 'head_landed', 'head_attempted', 'body_landed', 'body_attempted',
        'leg_landed', 'leg_attempted', 'distance_landed', 'distance_attempted',
        'clinch_landed', 'clinch_attempted', 'ground_landed', 'ground_attempted'
    ]

    other_columns = ['open_odds', 'closing_range_start', 'closing_range_end', 'elo', 'elo_win_probability',
                     'years_of_experience', 'win_streak', 'loss_streak', 'days_since_last_fight']

    # Generate the columns to differentiate using list comprehension
    columns_to_diff = base_columns + [f"{col}_career" for col in base_columns] + [f"{col}_career_avg" for col in
                                                                                  base_columns] + other_columns

    # Calculate the differential for each column and store in a new DataFrame
    diff_df = pd.DataFrame(
        {f"{col}_diff": final_combined_df[col] - final_combined_df[f"{col}_b"] for col in columns_to_diff})

    # Concatenate the differential DataFrame with the final_combined_df
    final_combined_df = pd.concat([final_combined_df, diff_df], axis=1)

    # Filter out rows where the winner column contains 'NC' or 'D'
    final_combined_df = final_combined_df[~final_combined_df['winner'].isin(['NC', 'D'])]

    # Convert the 'fight_date' column to datetime
    final_combined_df['fight_date'] = pd.to_datetime(final_combined_df['fight_date'])

    # Sort the DataFrame first by 'fighter_name' alphabetically, then by 'fight_date' descending (most recent first)
    final_combined_df = final_combined_df.sort_values(by=['fighter', 'fight_date'], ascending=[True, True])

    # Save the combined and sorted DataFrame to a CSV file
    final_combined_df.to_csv('data/combined_sorted_fighter_stats.csv', index=False)

    return final_combined_df


def split_train_val_test(matchup_data_file):
    # Load the matchup data
    matchup_df = pd.read_csv(matchup_data_file)

    # Remove correlated features
    matchup_df, removed_features = remove_correlated_features(matchup_df)

    # Ensure 'current_fight_date' is in datetime format
    matchup_df['current_fight_date'] = pd.to_datetime(matchup_df['current_fight_date'])

    start_date = '2022-06-01'
    end_date = '2024-6-30'  # Change this to your desired end date

    # Convert start_date to datetime
    start_date = pd.to_datetime(start_date)

    # Calculate the date 10 years before the start date
    ten_years_before = start_date - pd.DateOffset(years=10)

    test_data = matchup_df[(matchup_df['current_fight_date'] >= start_date) &
                           (matchup_df['current_fight_date'] <= end_date)].copy()

    remaining_data = matchup_df[(matchup_df['current_fight_date'] >= ten_years_before) &
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
        df['fight_pair'] = df.apply(lambda row: tuple(sorted([row['fighter'], row['fighter_b']])), axis=1)
        df = df.drop_duplicates(subset=['fight_pair'], keep='first')
        df = df.drop(columns=['fight_pair'])
        return df.reset_index(drop=True)

    val_data = remove_duplicates(val_data)
    test_data = remove_duplicates(test_data)

    # Sort train, validation, and test data by current_fight_date
    train_data = train_data.sort_values(by='current_fight_date', ascending=True)
    val_data = val_data.sort_values(by='current_fight_date', ascending=True)
    test_data = test_data.sort_values(by='current_fight_date', ascending=True)

    # Save the train, validation, and test data to CSV files
    train_data.to_csv('data/train test data/train_data.csv', index=False)
    val_data.to_csv('data/train test data/val_data.csv', index=False)
    test_data.to_csv('data/train test data/test_data.csv', index=False)

    # Save the removed features to a file
    with open('data/train test data/removed_features.txt', 'w') as file:
        file.write(','.join(removed_features))

    print(
        f"Train, validation, and test data saved successfully. {len(removed_features)} correlated features were removed.")
    print(f"Train set size: {len(train_data)}, Validation set size: {len(val_data)}, Test set size: {len(test_data)}")

def create_matchup_data(file_path, tester, name):
    df = pd.read_csv(file_path)

    columns_to_exclude = ['fighter', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
                          'result', 'winner', 'weight_class', 'scheduled_rounds',
                          'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']

    features_to_include = [col for col in df.columns if col not in columns_to_exclude and 'age' not in col.lower()]

    method_columns = ['winner']

    matchup_data = []
    n_past_fights = 6 - tester

    for index, current_fight in df.iterrows():
        fighter_name = current_fight['fighter']
        opponent_name = current_fight['fighter_b']

        fighter_df = df[(df['fighter'] == fighter_name) & (df['fight_date'] < current_fight['fight_date'])] \
            .sort_values(by='fight_date', ascending=False).head(n_past_fights)
        opponent_df = df[(df['fighter'] == opponent_name) & (df['fight_date'] < current_fight['fight_date'])] \
            .sort_values(by='fight_date', ascending=False).head(n_past_fights)

        if (len(fighter_df)) < n_past_fights or (len(opponent_df)) < n_past_fights:
            continue

        fighter_features = fighter_df[features_to_include].mean().values
        opponent_features = opponent_df[features_to_include].mean().values

        results_fighter = fighter_df[['result', 'winner', 'weight_class', 'scheduled_rounds']].head(tester).values.flatten()
        results_opponent = opponent_df[['result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']].head(
            tester).values.flatten()

        labels = current_fight[method_columns].values

        if pd.notna(current_fight['open_odds']) and pd.notna(current_fight['open_odds_b']):
            current_fight_odds = [current_fight['open_odds'], current_fight['open_odds_b']]
            current_fight_odds_diff = current_fight['open_odds'] - current_fight['open_odds_b']
        elif pd.notna(current_fight['open_odds']):
            odds_a = round_to_nearest_5(current_fight['open_odds'])
            odds_b = calculate_complementary_odd(odds_a)
            current_fight_odds = [odds_a, odds_b]
            current_fight_odds_diff = odds_a - odds_b
        elif pd.notna(current_fight['open_odds_b']):
            odds_b = round_to_nearest_5(current_fight['open_odds_b'])
            odds_a = calculate_complementary_odd(odds_b)
            current_fight_odds = [odds_a, odds_b]
            current_fight_odds_diff = odds_a - odds_b
        else:
            current_fight_odds = [-110, -110]
            current_fight_odds_diff = 0

        current_fight_ages = [current_fight['age'], current_fight['age_b']]
        current_fight_age_diff = current_fight['age'] - current_fight['age_b']

        # Retrieve the current fight Elo ratings for fighter A and fighter B
        current_fight_elo_a = current_fight['elo']
        current_fight_elo_b = current_fight['elo_b']
        current_fight_elo_diff = current_fight_elo_a - current_fight_elo_b

        current_fight_elo_a_win_chance = current_fight['elo_win_probability']
        current_fight_elo_b_win_chance = current_fight['elo_win_probability_b']
        current_fight_elo_chance_diff = current_fight_elo_a_win_chance - current_fight_elo_b_win_chance

        # Add new features
        current_fight_win_streak_a = current_fight['win_streak']
        current_fight_win_streak_b = current_fight['win_streak_b']
        current_fight_win_streak_diff = current_fight_win_streak_a - current_fight_win_streak_b

        current_fight_loss_streak_a = current_fight['loss_streak']
        current_fight_loss_streak_b = current_fight['loss_streak_b']
        current_fight_loss_streak_diff = current_fight_loss_streak_a - current_fight_loss_streak_b

        current_fight_years_experience_a = current_fight['years_of_experience']
        current_fight_years_experience_b = current_fight['years_of_experience_b']
        current_fight_years_experience_diff = current_fight_years_experience_a - current_fight_years_experience_b

        current_fight_days_since_last_a = current_fight['days_since_last_fight']
        current_fight_days_since_last_b = current_fight['days_since_last_fight_b']
        current_fight_days_since_last_diff = current_fight_days_since_last_a - current_fight_days_since_last_b

        combined_features = np.concatenate(
            [fighter_features, opponent_features, results_fighter, results_opponent, current_fight_odds,
             [current_fight_odds_diff], current_fight_ages, [current_fight_age_diff],
             [current_fight_elo_a, current_fight_elo_b, current_fight_elo_diff,
              current_fight_elo_a_win_chance, current_fight_elo_b_win_chance, current_fight_elo_chance_diff,
              current_fight_win_streak_a, current_fight_win_streak_b, current_fight_win_streak_diff,
              current_fight_loss_streak_a, current_fight_loss_streak_b, current_fight_loss_streak_diff,
              current_fight_years_experience_a, current_fight_years_experience_b, current_fight_years_experience_diff,
              current_fight_days_since_last_a, current_fight_days_since_last_b, current_fight_days_since_last_diff]])
        combined_row = np.concatenate([combined_features, labels])

        most_recent_date = max(fighter_df['fight_date'].max(), opponent_df['fight_date'].max())
        current_fight_date = current_fight['fight_date']  # Get the current fight date

        if not name:
            matchup_data.append([most_recent_date] + combined_row.tolist() + [current_fight_date])
        else:
            matchup_data.append([fighter_name, opponent_name, most_recent_date] + combined_row.tolist() + [current_fight_date])

    results_columns = []
    for i in range(1, tester + 1):
        results_columns += [f"result_fight_{i}", f"winner_fight_{i}", f"weight_class_fight_{i}",
                            f"scheduled_rounds_fight_{i}"]
        results_columns += [f"result_b_fight_{i}", f"winner_b_fight_{i}", f"weight_class_b_fight_{i}",
                            f"scheduled_rounds_b_fight_{i}"]

    new_columns = ['current_fight_elo_a', 'current_fight_elo_b', 'current_fight_elo_diff',
                   'current_fight_elo_a_win_chance', 'current_fight_elo_b_win_chance', 'current_fight_elo_chance_diff',
                   'current_fight_win_streak_a', 'current_fight_win_streak_b', 'current_fight_win_streak_diff',
                   'current_fight_loss_streak_a', 'current_fight_loss_streak_b', 'current_fight_loss_streak_diff',
                   'current_fight_years_experience_a', 'current_fight_years_experience_b',
                   'current_fight_years_experience_diff',
                   'current_fight_days_since_last_a', 'current_fight_days_since_last_b',
                   'current_fight_days_since_last_diff']

    if not name:
        column_names = ['fight_date'] + [f"{feature}_fighter_avg_last_{n_past_fights}" for feature in
                                         features_to_include] + \
                       [f"{feature}_fighter_b_avg_last_{n_past_fights}" for feature in features_to_include] + \
                       results_columns + ['current_fight_open_odds', 'current_fight_open_odds_b',
                                          'current_fight_open_odds_diff',
                                          'current_fight_age', 'current_fight_age_b',
                                          'current_fight_age_diff'] + new_columns + \
                       [f"{method}" for method in method_columns] + ['current_fight_date']
    else:
        column_names = ['fighter', 'fighter_b', 'fight_date'] + [f"{feature}_fighter_avg_last_{n_past_fights}" for
                                                                 feature in features_to_include] + \
                       [f"{feature}_fighter_b_avg_last_{n_past_fights}" for feature in features_to_include] + \
                       results_columns + ['current_fight_open_odds', 'current_fight_open_odds_b',
                                          'current_fight_open_odds_diff',
                                          'current_fight_age', 'current_fight_age_b',
                                          'current_fight_age_diff'] + new_columns + \
                       [f"{method}" for method in method_columns] + ['current_fight_date']

    matchup_df = pd.DataFrame(matchup_data, columns=column_names)

    if not name:
        matchup_df.to_csv(f'data/matchup data/matchup_data_{n_past_fights}_avg.csv', index=False)
    else:
        matchup_df.to_csv(f'data/matchup data/matchup_data_{n_past_fights}_avg_name.csv', index=False)

    return matchup_df


def create_specific_matchup_data(file_path, fighter_name, opponent_name, n_past_fights, name=False):
    df = pd.read_csv(file_path)

    # Convert fighter names to lowercase
    fighter_name = fighter_name.lower()
    opponent_name = opponent_name.lower()
    df['fighter'] = df['fighter'].str.lower()
    df['fighter_b'] = df['fighter_b'].str.lower()

    # Define the features to include for averaging, excluding identifiers and non-numeric features
    columns_to_exclude = ['fighter', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
                          'result', 'winner', 'weight_class', 'scheduled_rounds',
                          'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']
    features_to_include = [col for col in df.columns if col not in columns_to_exclude and 'age' not in col.lower()]

    matchup_data = []

    # Get the last 'n' fights for Fighter A and Fighter B before the specified matchup
    fighter_df = df[(df['fighter'] == fighter_name)].sort_values(by='fight_date', ascending=False).head(n_past_fights)
    opponent_df = df[(df['fighter'] == opponent_name)].sort_values(by='fight_date', ascending=False).head(n_past_fights)

    # Check if either fighter doesn't have enough fights
    if len(fighter_df) < n_past_fights or len(opponent_df) < n_past_fights:
        print("Specific matchup failure: One of the fighters doesn't have enough fights.")
        return None

    # Calculate the average of the relevant columns over the past 'n' fights
    fighter_features = fighter_df[features_to_include].mean().values
    opponent_features = opponent_df[features_to_include].mean().values

    # Create new columns for the specified features for each of the last three fights
    results_fighter = fighter_df[['result', 'winner', 'weight_class', 'scheduled_rounds']].head(3).values.flatten()
    results_opponent = opponent_df[['result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']].head(3).values.flatten()

    # Get user input for current fight odds, ages, and date
    current_fight_open_odds = float(input(f"Enter current open odds for {fighter_name}: "))
    current_fight_open_odds_b = float(input(f"Enter current open odds for {opponent_name}: "))
    current_fight_age = float(input(f"Enter current age for {fighter_name}: "))
    current_fight_age_b = float(input(f"Enter current age for {opponent_name}: "))
    current_fight_date = input("Enter the date of the fight (YYYY-MM-DD): ")

    # Calculate differentials
    current_fight_open_odds_diff = current_fight_open_odds - current_fight_open_odds_b
    current_fight_age_diff = current_fight_age - current_fight_age_b

    # Add current fight information to the features
    current_fight_info = [current_fight_open_odds, current_fight_open_odds_b, current_fight_open_odds_diff,
                          current_fight_age, current_fight_age_b, current_fight_age_diff]

    combined_features = np.concatenate(
        [fighter_features, opponent_features, results_fighter, results_opponent, current_fight_info])

    # Get the most recent fight date among the averaged fights
    most_recent_date = max(fighter_df['fight_date'].max(), opponent_df['fight_date'].max())

    # Add the combined features, most recent fight date, current fight date, and fighter names to the dataset
    matchup_data.append([fighter_name, opponent_name, most_recent_date] + combined_features.tolist() + [current_fight_date])

    # Define column names for the new DataFrame
    results_columns = []
    for i in range(1, 4):
        results_columns += [f"result_fight_{i}", f"winner_fight_{i}", f"weight_class_fight_{i}",
                            f"scheduled_rounds_fight_{i}"]
        results_columns += [f"result_b_fight_{i}", f"winner_b_fight_{i}", f"weight_class_b_fight_{i}",
                            f"scheduled_rounds_b_fight_{i}"]

    column_names = ['fighter', 'fighter_b', 'fight_date'] + \
                   [f"{feature}_fighter_avg_last_{n_past_fights}" for feature in features_to_include] + \
                   [f"{feature}_fighter_b_avg_last_{n_past_fights}" for feature in features_to_include] + \
                   results_columns + \
                   ['current_fight_open_odds', 'current_fight_open_odds_b', 'current_fight_open_odds_diff',
                    'current_fight_age', 'current_fight_age_b', 'current_fight_age_diff', 'current_fight_date']

    # Convert the matchup data into a DataFrame
    matchup_df = pd.DataFrame(matchup_data, columns=column_names)

    # Load the removed features from the file
    with open('data/train test data/removed_features.txt', 'r') as file:
        removed_features = file.read().split(',')

    # Drop the specified columns from the removed features
    matchup_df = matchup_df.drop(columns=[col for col in removed_features if col in matchup_df.columns], axis=1)

    # Remove 'fighter' and 'fighter_b' columns if name is False
    if not name:
        matchup_df = matchup_df.drop(['fighter', 'fighter_b'], axis=1)

    # Save the specific matchup data to a CSV file
    csv_name = f'specific_matchup_data.csv'
    matchup_df.to_csv(f'data/matchup data/{csv_name}', index=False)

    print("Specific matchup success. Data saved to CSV.")
    return matchup_df


if __name__ == "__main__":
    # combine_rounds_stats('data/ufc_fight_processed.csv')
    # calculate_elo_ratings('data/combined_rounds.csv')
    # combine_fighters_stats("data/combined_rounds.csv")
    # create_matchup_data("data/combined_sorted_fighter_stats.csv", 3, True)
    split_train_val_test('data/matchup data/matchup_data_3_avg_name.csv')
    # create_specific_matchup_data("data/combined_sorted_fighter_stats.csv", "leon edwards", "Belal Muhammad", 3, True)

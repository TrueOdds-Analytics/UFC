import pandas as pd
import numpy as np
from datetime import datetime
import multiprocessing
from functools import partial
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


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
    numeric_columns = numeric_columns.drop(['id', 'last_round', 'attendance', 'age'])

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

    final_stats = final_stats[~final_stats['winner'].isin(['NC', 'D'])]
    final_stats = final_stats[~final_stats['result'].isin(['DQ', 'Decision - Split'])]

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

    # Drop fights before 2014
    final_stats = final_stats[final_stats['fight_date'] >= '2014-01-01']

    # Load the cleaned fight odds data
    cleaned_odds_df = pd.read_csv('data/odds data/cleaned_fight_odds.csv')

    # Create mapping dictionaries for each odds column
    odds_columns = ['Open', 'Closing Range Start', 'Closing Range End', 'Movement']
    odds_mappings = {
        col: cleaned_odds_df.set_index(['Matchup', 'Event']).apply(lambda x: {'odds': x[col], 'Date': x['Date']},
                                                                   axis=1).to_dict() for col in odds_columns
    }

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
        processed_rows = 0
        for chunk_result, chunk_size in pool.imap(partial_process, chunks):
            results.append(chunk_result)
            processed_rows += chunk_size
            print(f"\rProgress: {processed_rows}/{total_rows} ({processed_rows / total_rows * 100:.2f}%)", end="",
                  flush=True)

    new_odds = pd.concat(results)

    end_time = time.time()
    print(f"\nOdds matching completed in {end_time - start_time:.2f} seconds.")

    # Add the new columns to final_stats
    for col in odds_columns:
        final_stats[f'new_{col}'] = new_odds[col]

    # Compare new odds with original odds if they existed
    if 'open_odds' in final_stats.columns:
        final_stats['original_open_odds'] = final_stats['open_odds']
        odds_diff = final_stats[final_stats['original_open_odds'] != final_stats['new_Open']]
        print(f"Number of rows with different odds: {len(odds_diff)}")
        print(odds_diff[['fighter', 'event', 'original_open_odds', 'new_Open']])
    else:
        print("No original odds to compare with. New odds have been added.")

    # Update the 'open_odds' column with the new odds and add new columns
    final_stats['open_odds'] = final_stats['new_Open']
    final_stats['closing_range_start'] = final_stats['new_Closing Range Start']
    final_stats['closing_range_end'] = final_stats['new_Closing Range End']
    final_stats['movement'] = final_stats['new_Movement']

    # Drop temporary columns
    columns_to_drop = ['new_Open', 'new_Closing Range Start', 'new_Closing Range End', 'new_Movement']
    if 'original_open_odds' in final_stats.columns:
        columns_to_drop.append('original_open_odds')
    final_stats = final_stats.drop(columns=columns_to_drop)

    # Drop the 'dob' column
    final_stats = final_stats.drop(columns=['dob'], errors='ignore')

    # Identify and drop duplicate columns
    duplicate_columns = final_stats.columns[final_stats.columns.duplicated()]
    final_stats = final_stats.loc[:, ~final_stats.columns.duplicated()]

    print(f"Dropped duplicate columns: {list(duplicate_columns)}")

    # Save to new CSV
    final_stats.to_csv('data/combined_rounds.csv', index=False)

    return final_stats


def combine_fighters_stats(file_path):
    # Load the data
    df = pd.read_csv(file_path)

    # Drop columns with 'event' in the title
    df = df.drop(columns=[col for col in df.columns if 'event' in col.lower()])

    # Sort by fight ID and fighter identifier to ensure consistent ordering
    fighter_identifier = 'fighter'
    df = df.sort_values(by=['id', fighter_identifier])

    # Generate column names for the second fighter's stats
    cols_fighter_1 = [col for col in df.columns if col != 'id']
    cols_fighter_2 = [f"{col}_b" for col in cols_fighter_1]

    # Split the DataFrame into two, one for each fighter in a fight
    df_fighter_1 = df.iloc[::2].reset_index(drop=True)
    df_fighter_2 = df.iloc[1::2].reset_index(drop=True).rename(columns=dict(zip(cols_fighter_1, cols_fighter_2)))

    # Merge the two DataFrames side-by-side, aligning by the fight ID
    combined_df = pd.concat([df_fighter_1, df_fighter_2], axis=1)

    # Create a mirrored DataFrame where the roles of Fighter 1 and Fighter 2 are reversed
    mirrored_cols_fighter_1 = [f"{col}_b" for col in cols_fighter_1]
    mirrored_cols_fighter_2 = cols_fighter_1

    df_fighter_1_mirror = df_fighter_1.rename(columns=dict(zip(cols_fighter_1, mirrored_cols_fighter_1)))
    df_fighter_2_mirror = df_fighter_2.rename(columns=dict(zip(mirrored_cols_fighter_1, mirrored_cols_fighter_2)))
    mirrored_combined_df = pd.concat([df_fighter_2_mirror, df_fighter_1_mirror], axis=1)

    # Concatenate both the original and mirrored DataFrames
    final_combined_df = pd.concat([combined_df, mirrored_combined_df], ignore_index=True)

    # Define the base columns to differentiate
    base_columns = [
        'knockdowns', 'significant_strikes_landed', 'significant_strikes_attempted',
        'significant_strikes_rate', 'total_strikes_landed', 'total_strikes_attempted',
        'takedown_successful', 'takedown_attempted', 'takedown_rate', 'submission_attempt',
        'reversals', 'head_landed', 'head_attempted', 'body_landed', 'body_attempted',
        'leg_landed', 'leg_attempted', 'distance_landed', 'distance_attempted',
        'clinch_landed', 'clinch_attempted', 'ground_landed', 'ground_attempted'
    ]

    other_columns = ['open_odds', 'closing_range_start', 'closing_range_end']

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


def split_train_val(matchup_data_file):
    # Load the matchup data
    matchup_df = pd.read_csv(matchup_data_file)

    # Remove correlated features
    matchup_df, removed_features = remove_correlated_features(matchup_df)

    # Create a separate DataFrame with fight dates
    fight_dates_df = matchup_df[['fight_date']]
    fight_dates_df = fight_dates_df.sort_values(by='fight_date', ascending=True)

    # Calculate the index to split the data into train and validation sets
    split_index = int(len(fight_dates_df) * 0.90)

    # Get the date threshold for splitting the data
    split_date = fight_dates_df.iloc[split_index]['fight_date']

    # Split the data into train and validation sets based on the fight date
    train_data = matchup_df[matchup_df['fight_date'] < split_date]
    val_data = matchup_df[matchup_df['fight_date'] >= split_date]

    # Drop the fight_date column after using it for splitting
    train_data = train_data.drop(columns=['fight_date'])
    val_data = val_data.drop(columns=['fight_date'])

    # Save the train and validation data to CSV files
    train_data.to_csv('data/train test data/train_data.csv', index=False)
    val_data.to_csv('data/train test data/val_data.csv', index=False)

    # Save the removed features to a file
    with open('data/train test data/removed_features.txt', 'w') as file:
        file.write(','.join(removed_features))

    print(f"Train and validation data saved successfully. {len(removed_features)} correlated features were removed.")
    print(f"Train set size: {len(train_data)}, Validation set size: {len(val_data)}")


def calculate_odds(odds):
    if pd.isna(odds):
        return 0, 0

    if odds > 0:
        win_percentage = 100 / (odds + 100)
    else:
        win_percentage = abs(odds) / (abs(odds) + 100)

    opponent_percentage = 1 - win_percentage

    if opponent_percentage > 0.5:
        opponent_odds = -100 * opponent_percentage / (1 - opponent_percentage)
    else:
        opponent_odds = 100 * (1 - opponent_percentage) / opponent_percentage

    return odds, opponent_odds


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

        if len(fighter_df) < n_past_fights or len(opponent_df) < n_past_fights:
            continue

        fighter_features = fighter_df[features_to_include].mean().values
        opponent_features = opponent_df[features_to_include].mean().values

        results_fighter = fighter_df[['result', 'winner', 'weight_class', 'scheduled_rounds']].head(3).values.flatten()
        results_opponent = opponent_df[['result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']].head(
            3).values.flatten()

        labels = current_fight[method_columns].values

        # Handle odds calculation
        if pd.notna(current_fight['open_odds']) and pd.notna(current_fight['open_odds_b']):
            current_fight_odds = [current_fight['open_odds'], current_fight['open_odds_b']]
            current_fight_odds_diff = current_fight['open_odds'] - current_fight['open_odds_b']
        elif pd.notna(current_fight['open_odds']):
            current_fight_odds = calculate_odds(current_fight['open_odds'])
            current_fight_odds_diff = current_fight_odds[0] - current_fight_odds[1]
        elif pd.notna(current_fight['open_odds_b']):
            reversed_odds = calculate_odds(current_fight['open_odds_b'])
            current_fight_odds = [reversed_odds[1], reversed_odds[0]]  # Swap the order
            current_fight_odds_diff = current_fight_odds[0] - current_fight_odds[1]
        else:
            current_fight_odds = [0, 0]
            current_fight_odds_diff = 0

        current_fight_ages = [current_fight['age'], current_fight['age_b']]
        current_fight_age_diff = current_fight['age'] - current_fight['age_b']

        combined_features = np.concatenate(
            [fighter_features, opponent_features, results_fighter, results_opponent, current_fight_odds,
             [current_fight_odds_diff], current_fight_ages, [current_fight_age_diff]])
        combined_row = np.concatenate([combined_features, labels])

        most_recent_date = max(fighter_df['fight_date'].max(), opponent_df['fight_date'].max())

        if not name:
            matchup_data.append([most_recent_date] + combined_row.tolist())
        else:
            matchup_data.append([fighter_name, opponent_name, most_recent_date] + combined_row.tolist())

    results_columns = []
    for i in range(1, 4):
        results_columns += [f"result_fight_{i}", f"winner_fight_{i}", f"weight_class_fight_{i}",
                            f"scheduled_rounds_fight_{i}"]
        results_columns += [f"result_b_fight_{i}", f"winner_b_fight_{i}", f"weight_class_b_fight_{i}",
                            f"scheduled_rounds_b_fight_{i}"]

    if not name:
        column_names = ['fight_date'] + [f"{feature}_fighter_avg_last_{n_past_fights - 1}" for feature in
                                         features_to_include] + \
                       [f"{feature}_fighter_b_avg_last_{n_past_fights - 1}" for feature in features_to_include] + \
                       results_columns + ['current_fight_open_odds', 'current_fight_open_odds_b',
                                          'current_fight_open_odds_diff',
                                          'current_fight_age', 'current_fight_age_b',
                                          'current_fight_age_diff'] + \
                       [f"{method}" for method in method_columns]
    else:
        column_names = ['fighter', 'fighter_b', 'fight_date'] + [f"{feature}_fighter_avg_last_{n_past_fights - 1}" for
                                                                 feature in features_to_include] + \
                       [f"{feature}_fighter_b_avg_last_{n_past_fights - 1}" for feature in features_to_include] + \
                       results_columns + ['current_fight_open_odds', 'current_fight_open_odds_b',
                                          'current_fight_open_odds_diff',
                                          'current_fight_age', 'current_fight_age_b',
                                          'current_fight_age_diff'] + \
                       [f"{method}" for method in method_columns]

    matchup_df = pd.DataFrame(matchup_data, columns=column_names)

    if not name:
        matchup_df.to_csv(f'data/matchup data/matchup_data_{n_past_fights - 1}_avg.csv', index=False)
    else:
        matchup_df.to_csv(f'data/matchup data/matchup_data_{n_past_fights - 1}_avg_name.csv', index=False)

    return matchup_df


def create_specific_matchup_data(file_path, fighter_name, opponent_name, n_past_fights, name=False):
    df = pd.read_csv(file_path)

    # Convert fighter names to lowercase
    fighter_name = fighter_name.lower()
    opponent_name = opponent_name.lower()
    df['fighter'] = df['fighter'].str.lower()
    df['fighter_b'] = df['fighter_b'].str.lower()

    # Load the removed features from the file
    with open('data/train test data/removed_features.txt', 'r') as file:
        removed_features = file.read().split(',')

    # Define the features to include for averaging, excluding identifiers, non-numeric features, and removed features
    columns_to_exclude = ['fighter', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
                          'result', 'winner', 'weight_class', 'scheduled_rounds',
                          'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']
    features_to_include = [col for col in df.columns if
                           col not in columns_to_exclude and col not in removed_features and 'age' not in col.lower()]

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
    results_opponent = opponent_df[['result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']].head(
        3).values.flatten()

    # Get user input for current fight odds and ages
    current_fight_open_odds = float(input(f"Enter current open odds for {fighter_name}: "))
    current_fight_open_odds_b = float(input(f"Enter current open odds for {opponent_name}: "))
    current_fight_age = float(input(f"Enter current age for {fighter_name}: "))
    current_fight_age_b = float(input(f"Enter current age for {opponent_name}: "))

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

    # Add the combined features, most recent fight date, and fighter names to the dataset
    matchup_data.append([fighter_name, opponent_name, most_recent_date] + combined_features.tolist())

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
                    'current_fight_age', 'current_fight_age_b', 'current_fight_age_diff']

    # Convert the matchup data into a DataFrame
    matchup_df = pd.DataFrame(matchup_data, columns=column_names)

    # Drop the specified columns from the removed features
    matchup_df = matchup_df.drop(columns=[col for col in removed_features if col in matchup_df.columns], axis=1)

    # Drop the 'fight_date' column
    matchup_df = matchup_df.drop(['fight_date'], axis=1)

    # Remove 'fighter' and 'fighter_b' columns if name is False
    if not name:
        matchup_df = matchup_df.drop(['fighter', 'fighter_b'], axis=1)

    # Save the specific matchup data to a CSV file
    csv_name = f'specific_matchup.csv'
    matchup_df.to_csv(f'data/{csv_name}', index=False)

    print("Specific matchup success. Data saved to CSV.")
    return matchup_df


if __name__ == "__main__":
    combine_rounds_stats('data/UFC_STATS_ORIGINAL.csv')
    combine_fighters_stats("data/combined_rounds.csv")
    create_matchup_data("data/combined_sorted_fighter_stats.csv", 2, False)
    split_train_val('data/matchup data/matchup_data_3_avg.csv')

import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings


def combine_rounds_stats(file_path):
    ufc_stats = pd.read_csv(file_path)
    ufc_stats['fighter'] = ufc_stats['fighter'].str.lower()
    ufc_stats['fight_date'] = pd.to_datetime(ufc_stats['fight_date'])

    # Drop unnecessary columns and rows
    ufc_stats = ufc_stats.drop(['round', 'location'], axis=1)
    ufc_stats = ufc_stats[~ufc_stats['weight_class'].str.contains("Women's")]

    # Identify numeric columns for aggregation
    numeric_columns = ufc_stats.select_dtypes(include='number').columns
    numeric_columns = numeric_columns.drop(['id', 'last_round', 'attendance'])

    fighter_identifier = 'fighter'

    # Convert 'time' column from minutes:seconds format to seconds
    ufc_stats['time'] = pd.to_datetime(ufc_stats['time'], format='%M:%S').dt.second + \
                        pd.to_datetime(ufc_stats['time'], format='%M:%S').dt.minute * 60

    # Find the highest round and time for each fight ID
    max_round_time = ufc_stats.groupby('id').agg({'last_round': 'max', 'time': 'max'}).reset_index()

    # Aggregate numeric columns by fight ID and fighter identifier
    aggregated_stats = ufc_stats.groupby(['id', fighter_identifier], as_index=False)[numeric_columns].sum()

    # Recalculate percentage columns
    aggregated_stats['significant_strikes_rate'] = (
            aggregated_stats['significant_strikes_landed'] / aggregated_stats['significant_strikes_attempted']).fillna(
        0)
    aggregated_stats['takedown_rate'] = (
            aggregated_stats['takedown_successful'] / aggregated_stats['takedown_attempted']).fillna(0)

    # Extract non-numeric columns and find unique rows
    non_numeric_columns = ufc_stats.select_dtypes(exclude='number').columns.difference(['id', fighter_identifier])
    non_numeric_data = ufc_stats.drop_duplicates(subset=['id', fighter_identifier])[
        ['id', fighter_identifier] + list(non_numeric_columns)]

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
                group['significant_strikes_landed_career'] / group['significant_strikes_attempted_career']).fillna(0)
        group['takedown_rate_career'] = (
                group['takedown_successful_career'] / group['takedown_attempted_career']).fillna(0)

        return group

    # Apply the aggregation function for each fighter up to each fight date
    final_stats = merged_stats.groupby(fighter_identifier, group_keys=False).apply(aggregate_up_to_date)

    # Reorder columns to ensure the final DataFrame has the same column order as the original
    common_columns = ufc_stats.columns.intersection(final_stats.columns)
    career_columns = [col for col in final_stats.columns if col.endswith('_career') or col.endswith('_career_avg')]

    if fighter_identifier not in common_columns:
        final_columns = [fighter_identifier] + list(common_columns) + career_columns
    else:
        final_columns = list(common_columns) + career_columns

    final_stats = final_stats[final_columns]

    final_stats = final_stats[~final_stats['winner'].isin(['NC', 'D'])]

    # Drop rows with 'DQ' and 'Decision - Split' in the 'result' column
    final_stats = final_stats[~final_stats['result'].isin(['DQ', 'Decision - Split'])]

    # Consolidate weight classes
    weight_class_mapping = {
        'Flyweight': 'Flyweight',
        'Bantamweight': 'Bantamweight',
        'Featherweight': 'Featherweight',
        'Lightweight': 'Lightweight',
        'Welterweight': 'Welterweight',
        'Middleweight': 'Middleweight',
        'Light Heavyweight': 'Light Heavyweight',
        'Heavyweight': 'Heavyweight',
        'Tournament': 'Tournament'
    }

    final_stats['weight_class'] = final_stats['weight_class'].apply(
        lambda x: next((v for k, v in weight_class_mapping.items() if k in x), x))

    # Convert unique strings to integers and create dictionary mappings
    for column in ['result', 'winner', 'weight_class', 'scheduled_rounds']:
        final_stats[column], unique = pd.factorize(final_stats[column])
        mapping = {index: label for index, label in enumerate(unique)}
        print(f"Mapping for {column}: {mapping}")

    # Drop fights before 2014
    final_stats = final_stats[final_stats['fight_date'] >= '2014-01-01']

    # Load the cleaned fight odds data
    cleaned_odds_df = pd.read_csv('data/odds data/cleaned_fight_odds.csv')

    # Create a mapping dictionary from Matchup and Event to Open odds
    odds_mapping = cleaned_odds_df.set_index(['Matchup', 'Event'])['Open'].to_dict()

    def get_odds(row):
        fighter = row['fighter'].lower()
        event = row['event'].lower()

        for (matchup, event_name), odds in odds_mapping.items():
            matchup_lower = matchup.lower()
            event_name_lower = event_name.lower()

            if fighter in matchup_lower:
                # Step 1: Direct comparison
                if event == event_name_lower:
                    return odds

                # Step 2: Compare first and last words
                event_words = event.split()
                event_name_words = event_name_lower.split()
                if len(event_words) > 1 and len(event_name_words) > 1:
                    if event_words[0] == event_name_words[0] and event_words[-1] == event_name_words[-1]:
                        return odds

                # Step 3: Levenshtein distance
                similarity_score = fuzz.ratio(event, event_name_lower)
                if similarity_score >= 80:  # 80% similarity threshold
                    return odds

        return None

    # Apply the function to get odds for each fighter
    final_stats['new_open_odds'] = final_stats.apply(get_odds, axis=1)

    # Compare new odds with original odds if they existed
    if 'open_odds' in final_stats.columns:
        final_stats['original_open_odds'] = final_stats['open_odds']
        odds_diff = final_stats[final_stats['original_open_odds'] != final_stats['new_open_odds']]
        print(f"Number of rows with different odds: {len(odds_diff)}")
        print(odds_diff[['fighter', 'event', 'original_open_odds', 'new_open_odds']])
    else:
        print("No original odds to compare with. New odds have been added.")

    # Update the 'open_odds' column with the new odds
    final_stats['open_odds'] = final_stats['new_open_odds']
    final_stats = final_stats.drop(columns=['new_open_odds'])

    if 'original_open_odds' in final_stats.columns:
        final_stats = final_stats.drop(columns=['original_open_odds'])

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

    # Generate the columns to differentiate using list comprehension
    columns_to_diff = base_columns + [f"{col}_career" for col in base_columns] + [f"{col}_career_avg" for col in base_columns]

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


def remove_multicollinear_features(matchup_df, vif_threshold=10):
    numeric_columns = matchup_df.select_dtypes(include=[np.number]).columns
    numeric_df = matchup_df[numeric_columns]

    # Create a temporary DataFrame for VIF calculation
    temp_df = numeric_df.copy()

    # Handle missing values and infinity values in the temporary DataFrame
    temp_df = temp_df.replace([np.inf, -np.inf], np.nan)
    temp_df = temp_df.dropna()

    vif_data = pd.DataFrame()
    vif_data["feature"] = temp_df.columns

    # Catch the runtime warning and set a default value for VIF
    vif_values = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(len(temp_df.columns)):
            try:
                vif = variance_inflation_factor(temp_df.values, i)
            except RuntimeWarning:
                vif = np.inf  # Set a default value of infinity for VIF
            vif_values.append(vif)

    vif_data["VIF"] = vif_values

    columns_to_drop = vif_data[vif_data["VIF"] > vif_threshold]["feature"].tolist()
    matchup_df = matchup_df.drop(columns=columns_to_drop)

    return matchup_df, columns_to_drop


def split_train_val(matchup_data_file):
    # Load the matchup data
    matchup_df = pd.read_csv(matchup_data_file)

    # Remove correlated features
    matchup_df, removed_features = remove_multicollinear_features(matchup_df)

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


def create_matchup_data(file_path, tester, name):
    df = pd.read_csv(file_path)

    columns_to_exclude = ['fighter', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
                          'result', 'winner', 'weight_class', 'scheduled_rounds',
                          'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b',
                          'open_odds', 'open_odds_b']  # Exclude odds from features to average
    features_to_include = [col for col in df.columns if col not in columns_to_exclude]

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
        results_opponent = opponent_df[['result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']].head(3).values.flatten()

        labels = current_fight[method_columns].values

        # Get current fight odds
        current_fight_odds = [current_fight['open_odds'], current_fight['open_odds_b']]

        combined_features = np.concatenate([fighter_features, opponent_features, results_fighter, results_opponent, current_fight_odds])
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
        column_names = ['fight_date'] + [f"{feature}_fighter_avg_last_{n_past_fights - 1}" for feature in features_to_include] + \
                       [f"{feature}_fighter_b_avg_last_{n_past_fights - 1}" for feature in features_to_include] + \
                       results_columns + ['current_fight_open_odds', 'current_fight_open_odds_b'] + \
                       [f"{method}" for method in method_columns]
    else:
        column_names = ['fighter', 'fighter_b', 'fight_date'] + [f"{feature}_fighter_avg_last_{n_past_fights - 1}" for feature in features_to_include] + \
                       [f"{feature}_fighter_b_avg_last_{n_past_fights - 1}" for feature in features_to_include] + \
                       results_columns + ['current_fight_open_odds', 'current_fight_open_odds_b'] + \
                       [f"{method}" for method in method_columns]

    matchup_df = pd.DataFrame(matchup_data, columns=column_names)

    if not name:
        matchup_df.to_csv(f'data/matchup data/matchup_data_{n_past_fights - 1}_avg.csv', index=False)
    else:
        matchup_df.to_csv(f'data/matchup data/matchup_data_{n_past_fights - 1}_avg_name.csv', index=False)

    return matchup_df


def create_specific_matchup_data(file_path, fighter_name, opponent_name, n_past_fights, csv_name=None):
    df = pd.read_csv(file_path)

    # Convert fighter names to lowercase
    fighter_name = fighter_name.lower()
    opponent_name = opponent_name.lower()
    df['fighter'] = df['fighter'].str.lower()
    df['fighter_b'] = df['fighter_b'].str.lower()

    # Load the removed features from the file
    with open('data/removed_features.txt', 'r') as file:
        removed_features = file.read().split(',')

    # Define the features to include for averaging, excluding identifiers, non-numeric features, and removed features
    columns_to_exclude = ['fighter', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
                          'result', 'winner', 'weight_class', 'scheduled_rounds',
                          'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']
    features_to_include = [col for col in df.columns if col not in columns_to_exclude and col not in removed_features]

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
    results_fighter = fighter_df[['result', 'winner', 'weight_class', 'scheduled_rounds', 'open_odds']].head(3).values.flatten()
    results_opponent = opponent_df[['result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b', 'open_odds_b']].head(
        3).values.flatten()

    combined_features = np.concatenate([fighter_features, opponent_features, results_fighter, results_opponent])

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

    column_names = ['fighter', 'fighter_b', 'fight_date'] + [f"{feature}_fighter_avg_last_{n_past_fights}" for feature
                                                             in features_to_include] + \
                   [f"{feature}_fighter_b_avg_last_{n_past_fights}" for feature in features_to_include] + \
                   results_columns

    # Convert the matchup data into a DataFrame
    matchup_df = pd.DataFrame(matchup_data, columns=column_names)

    # Drop the specified columns from the removed features
    matchup_df = matchup_df.drop(columns=[col for col in removed_features if col in matchup_df.columns], axis=1)

    # Drop the 'fight_date' column
    matchup_df = matchup_df.drop(['fight_date'], axis=1)

    # Save the specific matchup data to a CSV file
    if csv_name is None:
        csv_name = f'specific_matchup.csv'
    matchup_df.to_csv(f'data/{csv_name}', index=False)

    print("Specific matchup success. Data saved to CSV.")
    return matchup_df


if __name__ == "__main__":
    combine_rounds_stats('data/UFC_STATS_ORIGINAL.csv')
    combine_fighters_stats("data/combined_rounds.csv")
    create_matchup_data("data/combined_sorted_fighter_stats.csv", 2, False)
    split_train_val('data/matchup data/matchup_data_3_avg.csv')

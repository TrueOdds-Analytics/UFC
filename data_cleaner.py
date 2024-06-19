import numpy as np
import pandas as pd


def combine_rounds_stats(file_path):
    ufc_stats = pd.read_csv(file_path)
    ufc_stats['fighter'] = ufc_stats['fighter'].str.lower()

    # Convert fight_date to datetime for filtering
    ufc_stats['fight_date'] = pd.to_datetime(ufc_stats['fight_date'])

    # Drop 'round', 'location', and 'event' columns
    ufc_stats = ufc_stats.drop(['round', 'location', 'event'], axis=1)
    ufc_stats = ufc_stats[~ufc_stats['winner'].isin(['NC', 'D'])]

    # Drop fights with "Women's" in the weight_class column
    ufc_stats = ufc_stats[~ufc_stats['weight_class'].str.contains("Women's")]

    # Identify numeric columns
    numeric_columns = ufc_stats.select_dtypes(include='number').columns

    # Exclude 'id', 'last_round', and 'attendance' from the numeric columns for aggregation purposes
    numeric_columns = numeric_columns.drop(['id', 'last_round', 'attendance'])

    # Assume there's a 'fighter' column that uniquely identifies each fighter in a fight
    fighter_identifier = 'fighter'

    # Convert 'time' column from minutes:seconds format to seconds
    ufc_stats['time'] = pd.to_datetime(ufc_stats['time'], format='%M:%S').dt.second + pd.to_datetime(ufc_stats['time'], format='%M:%S').dt.minute * 60

    # Find the highest round and time for each fight ID
    max_round_time = ufc_stats.groupby('id').agg({'last_round': 'max', 'time': 'max'}).reset_index()

    # Aggregate the numeric columns by both fight ID and fighter identifier, summing them up
    aggregated_stats = ufc_stats.groupby(['id', fighter_identifier], as_index=False)[numeric_columns].sum()

    # Recalculate percentage columns based on the provided formulas
    aggregated_stats['significant_strikes_rate'] = (
        aggregated_stats['significant_strikes_landed'] / aggregated_stats['significant_strikes_attempted'])
    aggregated_stats['takedown_rate'] = (
        aggregated_stats['takedown_successful'] / aggregated_stats['takedown_attempted'])

    # Handle potential division by zero issues by replacing NaN values with zeros
    aggregated_stats['significant_strikes_rate'] = aggregated_stats['significant_strikes_rate'].fillna(0)
    aggregated_stats['takedown_rate'] = aggregated_stats['takedown_rate'].fillna(0)

    # Extract non-numeric columns (excluding 'id' and fighter_identifier) and find unique rows
    non_numeric_columns = ufc_stats.select_dtypes(exclude='number').columns.difference(['id', fighter_identifier])
    non_numeric_data = ufc_stats.drop_duplicates(subset=['id', fighter_identifier])[
        ['id', fighter_identifier] + list(non_numeric_columns)]

    # Merge the aggregated numeric and non-numeric data
    merged_stats = pd.merge(aggregated_stats, non_numeric_data, on=['id', fighter_identifier], how='left')

    # Merge the highest round and time data with merged_stats
    merged_stats = pd.merge(merged_stats, max_round_time, on='id', how='left')

    # Function to cumulatively sum numeric stats and create career stats columns
    def aggregate_up_to_date(group):
        group = group.sort_values('fight_date')
        cumulative_stats = group[numeric_columns].cumsum(skipna=True)
        fight_count = group.groupby('fighter').cumcount() + 1

        # Add new columns for cumulative career stats
        for col in numeric_columns:
            group[f"{col}_career"] = cumulative_stats[col]

        # Calculate career averages for each numeric column
        for col in numeric_columns:
            group[f"{col}_career_avg"] = group[f"{col}_career"] / fight_count

        # Calculate rates and add as new columns
        group['significant_strikes_rate_career'] = (
            group['significant_strikes_landed_career'] / group['significant_strikes_attempted_career']).fillna(0)
        group['takedown_rate_career'] = (
            group['takedown_successful_career'] / group['takedown_attempted_career']).fillna(0)

        return group

    # Apply the aggregation function for each fighter up to each fight date
    final_stats = merged_stats.groupby(fighter_identifier, group_keys=False).apply(aggregate_up_to_date)

    # Convert unique strings to integers and create dictionary mappings
    for column in ['result', 'winner', 'weight_class', 'scheduled_rounds']:
        final_stats[column], unique = pd.factorize(final_stats[column])
        mapping = {index: label for index, label in enumerate(unique)}
        print(f"Mapping for {column}: {mapping}")

    # Get the common columns between ufc_stats and final_stats
    common_columns = ufc_stats.columns.intersection(final_stats.columns)

    # Reorder columns to ensure the final DataFrame has the same column order as the original
    # Add the fighter_identifier to the ordering if it's not already included in common_columns
    if fighter_identifier not in common_columns:
        final_columns = [fighter_identifier] + list(common_columns) + [col for col in final_stats.columns if
                                                                       col.endswith('_career') or col.endswith('_career_avg')]
    else:
        final_columns = list(common_columns) + [col for col in final_stats.columns if
                                                col.endswith('_career') or col.endswith('_career_avg')]

    final_stats = final_stats[final_columns]

    # Drop fights before 2014
    final_stats = final_stats[final_stats['fight_date'] >= '2014-01-01']

    # Save to new CSV
    final_stats.to_csv('data/combined_rounds.csv', index=False)


def combine_fighters_stats(file_path):
    # Load the data
    df = pd.read_csv(file_path)

    # Assuming there's an identifier to sort by (e.g., fighter name)
    # This ensures that for each fight, Fighter 1 and Fighter 2 are consistently ordered
    fighter_identifier = 'fighter'  # Adjust based on your column name
    df = df.sort_values(by=['id', fighter_identifier])

    # Generate new column names for the second fighter's stats
    cols_fighter_1 = [col for col in df.columns if col not in ['id']]
    cols_fighter_2 = [f"{col}_b" for col in cols_fighter_1]

    # Split the DataFrame into two, one for each fighter in a fight
    df_fighter_1 = df.iloc[::2].reset_index(drop=True)
    df_fighter_2 = df.iloc[1::2].reset_index(drop=True).rename(columns=dict(zip(cols_fighter_1, cols_fighter_2)))

    # Merge the two DataFrames side-by-side, aligning by the fight 'id'
    combined_df = pd.concat([df_fighter_1, df_fighter_2], axis=1)

    # Create a mirrored DataFrame where the roles of Fighter 1 and Fighter 2 are reversed
    mirrored_cols_fighter_1 = [f"{col}_b" for col in cols_fighter_1]
    mirrored_cols_fighter_2 = cols_fighter_1

    # Original fighter columns now for opponent
    df_fighter_1_mirror = df_fighter_1.rename(columns=dict(zip(cols_fighter_1, mirrored_cols_fighter_1)))
    df_fighter_2_mirror = df_fighter_2.rename(columns=dict(zip(mirrored_cols_fighter_1, mirrored_cols_fighter_2)))
    mirrored_combined_df = pd.concat([df_fighter_2_mirror, df_fighter_1_mirror], axis=1)

    # Concatenate both the original and mirrored DataFrames
    final_combined_df = pd.concat([combined_df, mirrored_combined_df], ignore_index=True)

    # Define the columns to differentiate
    columns_to_diff = ['knockdowns_career_avg', 'significant_strikes_landed_career_avg',
                       'significant_strikes_attempted_career_avg',
                       'significant_strikes_rate_career_avg', 'total_strikes_landed_career_avg',
                       'total_strikes_attempted_career_avg',
                       'takedown_successful_career_avg', 'takedown_attempted_career_avg', 'takedown_rate_career_avg',
                       'submission_attempt_career_avg', 'reversals_career_avg',
                       'head_landed_career_avg', 'head_attempted_career_avg', 'body_landed_career_avg',
                       'body_attempted_career_avg',
                       'leg_landed_career_avg', 'leg_attempted_career_avg', 'distance_landed_career_avg',
                       'distance_attempted_career_avg',
                       'clinch_landed_career_avg', 'clinch_attempted_career_avg', 'ground_landed_career_avg',
                       'ground_attempted_career_avg',
                       'knockdowns', 'significant_strikes_landed', 'significant_strikes_attempted',
                       'significant_strikes_rate',
                       'total_strikes_landed', 'total_strikes_attempted', 'takedown_successful', 'takedown_attempted',
                       'takedown_rate',
                       'submission_attempt', 'reversals', 'head_landed', 'head_attempted', 'body_landed',
                       'body_attempted',
                       'leg_landed',
                       'leg_attempted', 'distance_landed', 'distance_attempted', 'clinch_landed', 'clinch_attempted',
                       'ground_landed',
                       'ground_attempted', 'body_attempted_career', 'body_landed_career', 'clinch_attempted_career',
                       'clinch_landed_career',
                       'distance_attempted_career', 'distance_landed_career', 'ground_attempted_career',
                       'ground_landed_career',
                       'head_attempted_career', 'head_landed_career', 'knockdowns_career', 'leg_attempted_career',
                       'leg_landed_career', 'reversals_career', 'significant_strikes_attempted_career',
                       'significant_strikes_landed_career',
                       'significant_strikes_rate_career', 'submission_attempt_career', 'takedown_attempted_career',
                       'takedown_rate_career',
                       'takedown_successful_career', 'total_strikes_attempted_career', 'total_strikes_landed_career'
                       ]

    # Calculate the differential for each column and add it as a new column
    for col in columns_to_diff:
        final_combined_df[f"{col}_diff"] = final_combined_df[col] - final_combined_df[f"{col}_b"]

    # Filter out rows where the winner column contains 'NC' or 'D'
    final_combined_df = final_combined_df[~final_combined_df['winner'].isin(['NC', 'D'])]

    # Convert the 'fight_date' column to datetime
    final_combined_df['fight_date'] = pd.to_datetime(final_combined_df['fight_date'])

    # Sort the DataFrame first by 'fighter_name' alphabetically, then by 'fight_date' descending (most recent first)
    final_combined_df = final_combined_df.sort_values(by=['fighter', 'fight_date'], ascending=[True, True])

    # Save the combined and sorted DataFrame to a CSV file
    final_combined_df.to_csv('data/combined_sorted_fighter_stats.csv', index=False)


def create_matchup_data(file_path, tester, name):
    df = pd.read_csv(file_path)

    # Define the features to include, excluding identifiers and non-numeric features
    if not name:
        features_to_include = [col for col in df.columns if
                               col not in ['fighter', 'id', 'fighter_b']]
    else:
        features_to_include = [col for col in df.columns if
                               col not in ['id']]

    method_columns = ['winner']

    matchup_data = []

    # Iterate through the DataFrame
    for index, current_fight in df.iterrows():
        fighter_name = current_fight['fighter']
        opponent_name = current_fight['fighter_b']

        # Get the last 3 fights for Fighter A and Fighter B before the current fight
        fighter_df = df[(df['fighter'] == fighter_name) & (df['fight_date'] < current_fight['fight_date'])].sort_values(
            by='fight_date', ascending=False).head(6 - tester)
        opponent_df = df[
            (df['fighter'] == opponent_name) & (df['fight_date'] < current_fight['fight_date'])].sort_values(
            by='fight_date', ascending=False).head(6 - tester)

        if len(fighter_df) < (6 - tester) or len(opponent_df) < (6 - tester):
            continue  # Skip if either fighter does not have at least 4 past fights

        # Ensure only the last 3 fights are used for features and the 4th (most recent) for labeling
        feature_fights_fighter = fighter_df
        feature_fights_opponent = opponent_df

        labels = current_fight[method_columns].values

        combined_features = np.array([])

        # Loop through each of the last 3 fights to alternate between Fighter A and Fighter B
        for i in range(5 - tester):
            fighter_features = feature_fights_fighter.iloc[i][features_to_include].values
            opponent_features = feature_fights_opponent.iloc[i][features_to_include].values
            combined_features = np.concatenate([combined_features, fighter_features, opponent_features])

        combined_row = np.concatenate([combined_features, labels])

        # Add the combined row to the dataset
        matchup_data.append(combined_row)

    # Define column names for the new DataFrame
    column_names = []
    for i in range(1, 6 - tester):  # Matches the range 1 to 3 for 3 past fights
        column_names += [f"{feature}_fighter_fight{i}" for feature in features_to_include]
        column_names += [f"{feature}_opponent_fight{i}" for feature in features_to_include]
    column_names.extend([f"{method}" for method in method_columns])

    # Convert the matchup data into a DataFrame
    matchup_df = pd.DataFrame(matchup_data, columns=column_names)
    if not name:
        matchup_df.to_csv(f'data/matchup_data_{5 - tester}_sequence.csv', index=False)
    else:
        matchup_df.to_csv(f'data/matchup_data_{5 - tester}_sequence_name.csv', index=False)

    return matchup_df


def remove_correlated_features(matchup_df, correlation_threshold=0.95):
    # Select only the numeric columns
    numeric_columns = matchup_df.select_dtypes(include=[np.number]).columns
    numeric_df = matchup_df[numeric_columns]

    # Calculate the correlation matrix
    corr_matrix = numeric_df.corr().abs()

    # Select the upper triangle of the correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find the columns to drop
    columns_to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > correlation_threshold)]

    # Drop the highly correlated columns
    matchup_df = matchup_df.drop(columns=columns_to_drop)

    # Get the number of removed features
    num_removed_features = len(columns_to_drop)

    return matchup_df, num_removed_features


def split_train_test(matchup_data_file, test):
    # Load the matchup data
    matchup_df = pd.read_csv(matchup_data_file)

    # Remove correlated features
    matchup_df, num_removed_features = remove_correlated_features(matchup_df)

    # Create a separate DataFrame with fight dates
    fight_dates_df = matchup_df[['fight_date_fighter_fight1']]
    fight_dates_df = fight_dates_df.sort_values(by='fight_date_fighter_fight1', ascending=True)

    # Calculate the index to split the data into train and test sets
    split_index = int(len(fight_dates_df) * 0.9)

    # Get the date threshold for splitting the data
    split_date = fight_dates_df.iloc[split_index]['fight_date_fighter_fight1']

    # Split the data into train and test sets based on the fight date
    train_data = matchup_df[matchup_df['fight_date_fighter_fight1'] < split_date]
    test_data = matchup_df[matchup_df['fight_date_fighter_fight1'] >= split_date]

    if not test:
        # Remove columns with "fight_date" in the column name
        fight_date_columns = [col for col in train_data.columns if 'fight_date' in col]
        train_data = train_data.drop(columns=fight_date_columns)
        test_data = test_data.drop(columns=fight_date_columns)

        # Apply abs() to the entire DataFrame
        train_data = train_data.abs()
        test_data = test_data.abs()


    if not test:
        # Save the train and test data to CSV files
        train_data.to_csv('data/train_data.csv', index=False)
        test_data.to_csv('data/test_data.csv', index=False)
    else:
        train_data.to_csv('data/train_data_test.csv', index=False)
        test_data.to_csv('data/test_data_test.csv', index=False)

    print(f"Train and test data saved successfully. {num_removed_features} correlated features were removed.")


def create_matchup_data_history(file_path, name):
    df = pd.read_csv(file_path)

    # Define the features to include, excluding identifiers and non-numeric features
    # Define the features to include, excluding identifiers and non-numeric features
    if not name:
        features_to_include = [col for col in df.columns if
                               col not in ['round', 'time', 'attendance', 'id', 'round_opponent',
                                           'time_opponent', 'event_opponent', 'event', 'fight_date',
                                           'location_opponent', 'attendance_opponent',
                                           'id.1', 'location', 'last_round', 'scheduled_rounds', 'weight_class',
                                           'last_round_opponent', 'scheduled_rounds_opponent',
                                           'weight_class_opponent', 'fight_date_opponent', 'fighter', 'fighter.1']]
    else:
        features_to_include = [col for col in df.columns if
                               col not in ['round', 'time', 'attendance', 'id', 'round_opponent',
                                           'time_opponent', 'event_opponent', 'event', 'fight_date',
                                           'location_opponent', 'attendance_opponent',
                                           'id.1', 'location', 'last_round', 'scheduled_rounds', 'weight_class',
                                           'last_round_opponent', 'scheduled_rounds_opponent',
                                           'weight_class_opponent', 'fight_date_opponent']]

    method_columns = ['winner']

    matchup_data = []
    labels_data = []

    # Iterate through the DataFrame
    for index, current_fight in df.iterrows():
        fighter_name = current_fight['fighter']
        opponent_name = current_fight['fighter.1']

        # Get the last fights for Fighter A and Fighter B before the current fight
        fighter_df = df[(df['fighter'] == fighter_name) & (df['fight_date'] < current_fight['fight_date'])].sort_values(
            by='fight_date', ascending=False)
        opponent_df = df[
            (df['fighter'] == opponent_name) & (df['fight_date'] < current_fight['fight_date'])].sort_values(
            by='fight_date', ascending=False)

        if len(fighter_df) < 2 or len(opponent_df) < 1:
            continue

        # Ensure only the last 3 fights are used for features and the 4th (most recent) for labeling
        feature_fights_fighter = fighter_df
        feature_fights_opponent = opponent_df

        combined_features = np.array([])

        # Loop through each of the last fights to alternate between Fighter A and Fighter B
        max_fights = max(len(feature_fights_fighter), len(feature_fights_opponent))
        for i in range(max_fights):
            try:
                fighter_features = feature_fights_fighter.iloc[i][features_to_include].values
            except IndexError:
                fighter_features = np.zeros(len(features_to_include))

            try:
                opponent_features = feature_fights_opponent.iloc[i][features_to_include].values
            except IndexError:
                opponent_features = np.zeros(len(features_to_include))

            combined_features = np.concatenate([combined_features, fighter_features, opponent_features])

        # Extract the labels for the current fight
        labels = current_fight[method_columns].values

        # Add the combined features and labels to the respective lists
        matchup_data.append(combined_features)
        labels_data.append(labels)

    # Define column names for the new DataFrame
    column_names = []
    for i in range(1, 42):
        column_names += [f"{feature}_fighter_fight{i}" for feature in features_to_include]
        column_names += [f"{feature}_opponent_fight{i}" for feature in features_to_include]

    # Convert the matchup data and labels data into DataFrames
    matchup_df = pd.DataFrame(matchup_data, columns=column_names)
    labels_df = pd.DataFrame(labels_data, columns=method_columns)

    # Concatenate the matchup DataFrame and labels DataFrame horizontally
    final_matchup_df = pd.concat([matchup_df, labels_df], axis=1)

    keywords = [
        'knockdowns', 'significant_strikes_landed', 'significant_strikes_attempted',
        'significant_strikes_rate', 'total_strikes_landed', 'total_strikes_attempted',
        'takedown_successful', 'takedown_attempted', 'takedown_rate', 'submission_attempt',
        'reversals', 'head_landed', 'head_attempted', 'body_landed', 'body_attempted',
        'leg_landed', 'leg_attempted', 'distance_landed', 'distance_attempted',
        'clinch_landed', 'clinch_attempted', 'ground_landed', 'ground_attempted'
    ]

    feature_means = {}
    feature_stds = {}

    for col in final_matchup_df.columns:
        for keyword in keywords:
            if keyword in col:
                final_matchup_df[col] = final_matchup_df[col].astype(float)
                break

    for col in final_matchup_df.columns:
        if any(keyword in col for keyword in keywords):
            feature_data = final_matchup_df[col].replace(0, np.nan)
            feature_mean = feature_data.mean()
            feature_std = feature_data.std()

            feature_means[col] = feature_mean
            feature_stds[col] = feature_std

            final_matchup_df.loc[final_matchup_df[col] != 0, col] = (feature_data - feature_mean) / feature_std

    # Fill NaN cells with 0
    final_matchup_df.fillna(0, inplace=True)

    # Flip the DataFrame while keeping each sequence of 128 rows intact and preserving the order within the sequence

    flipped_df = pd.DataFrame()
    if name:
        num_sequences = len(final_matchup_df) // 128
        for i in range(num_sequences):
            start_index = i * 128
            end_index = (i + 1) * 128
            sequence_df = final_matchup_df.iloc[start_index:end_index]
            sequence_columns = sequence_df.columns[:-7]  # Exclude the last 7 label columns
            flipped_sequence_df = sequence_df[sequence_columns[::-1]]
            flipped_sequence_df = pd.concat([flipped_sequence_df, sequence_df.iloc[:, -7:]], axis=1)
            flipped_df = pd.concat([flipped_df, flipped_sequence_df])
    else:
        num_sequences = len(final_matchup_df) // 124
        for i in range(num_sequences):
            start_index = i * 124
            end_index = (i + 1) * 124
            sequence_df = final_matchup_df.iloc[start_index:end_index]
            sequence_columns = sequence_df.columns[:-7]  # Exclude the last 7 label columns
            flipped_sequence_df = sequence_df[sequence_columns[::-1]]
            flipped_sequence_df = pd.concat([flipped_sequence_df, sequence_df.iloc[:, -7:]], axis=1)
            flipped_df = pd.concat([flipped_df, flipped_sequence_df])

    final_matchup_df = flipped_df

    if not name:
        final_matchup_df.to_csv(f'data/matchup data/matchup_data_history_sequence_ndc.csv', index=False)
    else:
        final_matchup_df.to_csv(f'data/matchup data/matchup_data_name_history_sequence_ndc.csv',
                                index=False)

    # Save the mean and standard deviation dictionaries to files
    pd.to_pickle(feature_means, 'data/feature_means.pkl')
    pd.to_pickle(feature_stds, 'data/feature_stds.pkl')

    return final_matchup_df


def create_specific_matchup_data(file_path, fighter_a, fighter_b, tester):
    CRED = '\33[31m'
    CGREEN = '\33[32m'
    CEND = '\33[0m'
    df = pd.read_csv(file_path)

    # Define the features to include, excluding identifiers and non-numeric features
    features_to_include = [col for col in df.columns if
                           col not in ['round', 'time', 'attendance', 'id', 'round_opponent',
                                       'time_opponent', 'event_opponent', 'event',
                                       'location_opponent', 'attendance_opponent',
                                       'id.1', 'location', 'last_round', 'scheduled_rounds', 'weight_class',
                                       'fight_date', 'last_round_opponent', 'scheduled_rounds_opponent',
                                       'weight_class_opponent', 'fight_date_opponent', 'fighter', 'fighter.1']]

    # Retrieve the last 3 fights for both fighters
    fighter_df = df[df['fighter'] == fighter_a].sort_values(by='fight_date', ascending=False).head(3)
    opponent_df = df[df['fighter'] == fighter_b].sort_values(by='fight_date', ascending=False).head(3)

    if len(fighter_df) < (3 - tester) or len(opponent_df) < (3 - tester):
        print(CRED + "Not enough historical data for one of the fighters" + CEND)
        return None

    combined_features = np.array([])

    # Loop through each of the last 3 fights to alternate between Fighter A and Fighter B
    for i in range(3 - tester):
        fighter_features = fighter_df.iloc[i][features_to_include].values
        opponent_features = opponent_df.iloc[i][features_to_include].values
        combined_features = np.concatenate([combined_features, fighter_features, opponent_features])

    # Prepare for DataFrame creation
    matchup_data = [combined_features]  # Wrap in a list to create a single-row DataFrame

    # Define column names for the new DataFrame
    column_names = []
    for i in range(1, 4 - tester):  # Matches the range 1 to 3 for 3 past fights
        column_names += [f"{feature}_fighter_fight{i}" for feature in features_to_include]
        column_names += [f"{feature}_opponent_fight{i}" for feature in features_to_include]

    # Create DataFrame
    matchup_df = pd.DataFrame(matchup_data, columns=column_names)

    # Define keywords to apply Z-scoring
    keywords = [
        'knockdowns', 'significant_strikes_landed', 'significant_strikes_attempted',
        'significant_strikes_rate', 'total_strikes_landed', 'total_strikes_attempted',
        'takedown_successful', 'takedown_attempted', 'takedown_rate', 'submission_attempt',
        'reversals', 'head_landed', 'head_attempted', 'body_landed', 'body_attempted',
        'leg_landed', 'leg_attempted', 'distance_landed', 'distance_attempted',
        'clinch_landed', 'clinch_attempted', 'ground_landed', 'ground_attempted'
    ]

    # Load the mean and standard deviation dictionaries from pickle files
    feature_means = pd.read_pickle('data/feature_means.pkl')
    feature_stds = pd.read_pickle('data/feature_stds.pkl')

    # Apply z-scoring to columns containing the specified keywords
    for col in matchup_df.columns:
        if any(keyword in col for keyword in keywords):
            if col in feature_means and col in feature_stds:
                matchup_df[col] = (matchup_df[col].fillna(0) - feature_means[col]) / feature_stds[col]
            else:
                print(f"Warning: Mean and standard deviation not found for column '{col}'. Skipping z-scoring.")

    # Save to CSV
    matchup_df.to_csv('data/matchup data/specific_matchup_data.csv', index=False)
    print(CGREEN + "Matchup created successfully." + CEND)
    return matchup_df


if __name__ == "__main__":
    combine_rounds_stats('data/UFC_STATS_ORIGINAL.csv')
    combine_fighters_stats("data/combined_rounds.csv")
    create_matchup_data("data/combined_sorted_fighter_stats.csv", 2, False)
    split_train_test('data/matchup_data_3_sequence.csv', False)
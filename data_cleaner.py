import numpy as np
import pandas as pd


def combine_rounds_stats(file_path):
    ufc_stats = pd.read_csv(file_path)
    ufc_stats['fighter'] = ufc_stats['fighter'].str.lower()

    # Convert fight_date to datetime for filtering
    ufc_stats['fight_date'] = pd.to_datetime(ufc_stats['fight_date'])

    # Drop 'round', 'location', and 'event' columns
    ufc_stats = ufc_stats.drop(['round', 'location', 'event'], axis=1)

    # Drop fights with "Women's" in the weight_class column and fights by split decision and DQs
    ufc_stats = ufc_stats[~ufc_stats['weight_class'].str.contains("Women's")]

    # Identify numeric columns
    numeric_columns = ufc_stats.select_dtypes(include='number').columns

    # Exclude 'id', 'last_round', and 'attendance' from the numeric columns for aggregation purposes
    numeric_columns = numeric_columns.drop(['id', 'last_round', 'attendance'])

    # Assume there's a 'fighter' column that uniquely identifies each fighter in a fight
    fighter_identifier = 'fighter'

    # Convert 'time' column from minutes:seconds format to seconds
    ufc_stats['time'] = pd.to_datetime(ufc_stats['time'], format='%M:%S').dt.second + pd.to_datetime(ufc_stats['time'],
                                                                                                     format='%M:%S').dt.minute * 60

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

    # Convert unique strings to integers and create dictionary mappings
    for column in ['result', 'winner', 'weight_class', 'scheduled_rounds']:
        merged_stats[column], unique = pd.factorize(merged_stats[column])
        mapping = {index: label for index, label in enumerate(unique)}
        print(f"Mapping for {column}: {mapping}")

    # Read the fighter_stats file
    fighter_stats = pd.read_csv('data/fighter_stats.csv')

    # Rename columns to match the naming convention
    fighter_stats.columns = fighter_stats.columns.str.lower()

    # Preprocess fighter names in merged_stats
    merged_stats['fighter'] = merged_stats['fighter'].str.lower().str.strip()

    # Preprocess fighter names in fighter_stats
    fighter_stats['name'] = fighter_stats['name'].str.lower().str.strip()

    # Replace '--' with NaN in the 'height' column
    fighter_stats['height'] = fighter_stats['height'].replace('--', np.nan)

    # Convert height to inches
    fighter_stats['height'] = fighter_stats['height'].apply(
        lambda x: int(x.split("'")[0]) * 12 + int(x.split("'")[1].replace('"', '')) if pd.notna(x) else x)

    # Fill NaN values in the 'height' column with the median
    fighter_stats['height'] = fighter_stats['height'].fillna(fighter_stats['height'].median())

    # Replace '--' with NaN in the 'reach' column
    fighter_stats['reach'] = fighter_stats['reach'].replace('--', np.nan)

    # Convert reach to inches
    fighter_stats['reach'] = fighter_stats['reach'].str.replace('"', '').astype(float)

    # Fill NaN values in the 'reach' column with the median
    fighter_stats['reach'] = fighter_stats['reach'].fillna(fighter_stats['reach'].median())

    # Encode stance column
    stance_mapping = {'Orthodox': 1, 'Southpaw': 2, 'Switch': 3}
    fighter_stats['stance'] = fighter_stats['stance'].map(stance_mapping)

    # Replace NaN values in the 'stance' column with 1
    fighter_stats['stance'] = fighter_stats['stance'].fillna(1)

    # Merge fighter_stats with merged_stats based on the fighter name
    merged_stats = pd.merge(merged_stats, fighter_stats, left_on='fighter', right_on='name', how='left')

    # Drop the 'name' column since it's redundant with 'fighter'
    merged_stats = merged_stats.drop('name', axis=1)

    # Replace '--' with NaN in the 'dob' column
    merged_stats['dob'] = merged_stats['dob'].replace('--', np.nan)

    # Convert 'dob' to datetime format
    merged_stats['dob'] = pd.to_datetime(merged_stats['dob'])

    # Fill NaN values in the 'dob' column with the median
    merged_stats['dob'] = merged_stats['dob'].fillna(merged_stats['dob'].median())

    # Calculate the age of the fighter at the time of each fight
    merged_stats['age'] = (merged_stats['fight_date'] - merged_stats['dob']).dt.days / 365.25

    # Replace NaN values in the 'age' column with a default value
    merged_stats['age'] = merged_stats['age'].fillna(30)

    # Round the age to the nearest integer
    merged_stats['age'] = merged_stats['age'].round().astype(int)

    # Function to cumulatively sum numeric stats, create career stats columns, and calculate average age
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

        # Calculate cumulative sum of ages
        cumulative_age = group['age'].cumsum()

        # Calculate average age by dividing the cumulative sum of ages by the fight count
        group['age_avg'] = cumulative_age / fight_count

        return group

    # Apply the aggregation function for each fighter up to each fight date
    final_stats = merged_stats.groupby(fighter_identifier, group_keys=False).apply(aggregate_up_to_date)

    # Reorder columns to ensure the desired column order
    column_order = [
                       'fighter', 'height', 'reach', 'stance', 'age', 'age_avg'
                   ] + [col for col in final_stats.columns if
                        col not in ['fighter', 'height', 'reach', 'stance', 'dob', 'age', 'age_avg']]
    final_stats = final_stats[column_order]

    # Drop specified columns
    columns_to_drop = ['height', 'age', 'reach', 'age_avg', 'stance', 'dob', 'url', 'slpm', 'str_acc', 'sapm', 'str_def', 'td_avg', 'td_acc', 'td_def', 'sub_avg',
                       'num_fights']
    final_stats = final_stats.drop(columns=columns_to_drop, errors='ignore')

    # Convert 'result' and 'winner' columns to string before filtering
    final_stats['result'] = final_stats['result'].astype(str)
    final_stats['winner'] = final_stats['winner'].astype(str)

    # Drop fights before 2014 and fights with specific numeric conditions in the 'result' and 'winner' columns
    final_stats = final_stats[(final_stats['fight_date'] >= '2014-01-01') &
                              (~final_stats['result'].str.contains('5', case=False)) &
                              (~final_stats['result'].str.contains('6', case=False)) &
                              (~final_stats['winner'].str.contains('2', case=False)) &
                              (~final_stats['winner'].str.contains('3', case=False))]

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
    # 'height', 'age', 'reach', 'age_avg',
    columns_to_diff = [
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

    # columns_to_diff = ['height', 'age', 'reach', 'knockdowns_career_avg', 'significant_strikes_landed_career_avg',
    #                    'significant_strikes_attempted_career_avg', 'age_avg',
    #                    'significant_strikes_rate_career_avg', 'total_strikes_landed_career_avg',
    #                    'total_strikes_attempted_career_avg',
    #                    'takedown_successful_career_avg', 'takedown_attempted_career_avg', 'takedown_rate_career_avg',
    #                    'submission_attempt_career_avg', 'reversals_career_avg',
    #                    'head_landed_career_avg', 'head_attempted_career_avg', 'body_landed_career_avg',
    #                    'body_attempted_career_avg',
    #                    'leg_landed_career_avg', 'leg_attempted_career_avg', 'distance_landed_career_avg',
    #                    'distance_attempted_career_avg',
    #                    'clinch_landed_career_avg', 'clinch_attempted_career_avg', 'ground_landed_career_avg',
    #                    'ground_attempted_career_avg',
    #                    'knockdowns', 'significant_strikes_landed', 'significant_strikes_attempted',
    #                    'significant_strikes_rate',
    #                    'total_strikes_landed', 'total_strikes_attempted', 'takedown_successful', 'takedown_attempted',
    #                    'takedown_rate',
    #                    'submission_attempt', 'reversals', 'head_landed', 'head_attempted', 'body_landed',
    #                    'body_attempted',
    #                    'leg_landed',
    #                    'leg_attempted', 'distance_landed', 'distance_attempted', 'clinch_landed', 'clinch_attempted',
    #                    'ground_landed',
    #                    'ground_attempted', 'body_attempted_career', 'body_landed_career', 'clinch_attempted_career',
    #                    'clinch_landed_career',
    #                    'distance_attempted_career', 'distance_landed_career', 'ground_attempted_career',
    #                    'ground_landed_career',
    #                    'head_attempted_career', 'head_landed_career', 'knockdowns_career', 'leg_attempted_career',
    #                    'leg_landed_career', 'reversals_career', 'significant_strikes_attempted_career',
    #                    'significant_strikes_landed_career',
    #                    'significant_strikes_rate_career', 'submission_attempt_career', 'takedown_attempted_career',
    #                    'takedown_rate_career',
    #                    'takedown_successful_career', 'total_strikes_attempted_career', 'total_strikes_landed_career'
    #                    ]

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


if __name__ == "__main__":
    combine_rounds_stats('data/UFC_STATS_ORIGINAL.csv')
    combine_fighters_stats("data/combined_rounds.csv")
    create_matchup_data("data/combined_sorted_fighter_stats.csv", 2, False)
    split_train_test('data/matchup_data_3_sequence.csv', False)

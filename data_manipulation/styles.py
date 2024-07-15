import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def create_fighter_styles(input_file='../data/combined_rounds.csv', n_clusters=5):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Select the relevant columns for clustering
    style_columns = [
        'winner', 'result',
        'knockdowns_career', 'knockdowns_career_avg',
        'significant_strikes_landed_career', 'significant_strikes_landed_career_avg',
        'significant_strikes_attempted_career', 'significant_strikes_attempted_career_avg',
        'significant_strikes_rate_career', 'significant_strikes_rate_career_avg',
        'total_strikes_landed_career', 'total_strikes_landed_career_avg',
        'total_strikes_attempted_career', 'total_strikes_attempted_career_avg',
        'takedown_successful_career', 'takedown_successful_career_avg',
        'takedown_attempted_career', 'takedown_attempted_career_avg',
        'takedown_rate_career', 'takedown_rate_career_avg',
        'submission_attempt_career', 'submission_attempt_career_avg',
        'reversals_career', 'reversals_career_avg',
        'head_landed_career', 'head_landed_career_avg',
        'head_attempted_career', 'head_attempted_career_avg',
        'body_landed_career', 'body_landed_career_avg',
        'body_attempted_career', 'body_attempted_career_avg',
        'leg_landed_career', 'leg_landed_career_avg',
        'leg_attempted_career', 'leg_attempted_career_avg',
        'distance_landed_career', 'distance_landed_career_avg',
        'distance_attempted_career', 'distance_attempted_career_avg',
        'clinch_landed_career', 'clinch_landed_career_avg',
        'clinch_attempted_career', 'clinch_attempted_career_avg',
        'ground_landed_career', 'ground_landed_career_avg',
        'ground_attempted_career', 'ground_attempted_career_avg',
        'control_career', 'control_career_avg'
    ]

    # Prepare the data for clustering
    X = df[style_columns].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Generate unique integer labels
    unique_labels = np.random.choice(range(100, 1000), size=n_clusters, replace=False)
    label_mapping = dict(zip(range(n_clusters), unique_labels))

    # Add cluster labels to the DataFrame
    df.loc[X.index, 'Style'] = [label_mapping[label] for label in cluster_labels]

    # Fill NaN values in the Style column
    df['Style'] = df['Style'].fillna(-1).astype(int)

    # Save the results back to the input file
    df.to_csv(input_file, index=False)

    print(f"Fighter styles have been added to {input_file}")

    # Print cluster centroids
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    centroid_df = pd.DataFrame(centroids, columns=style_columns)
    print("\nCluster Centroids:")
    print(centroid_df)

    # Print number of fighters and percentage for each cluster
    total_fights = len(df)
    for i, unique_label in label_mapping.items():
        cluster_size = (df['Style'] == unique_label).sum()
        percentage = (cluster_size / total_fights) * 100
        print(f"Style {unique_label}:")
        print(f"Number of fights: {cluster_size}")
        print(f"Percentage: {percentage:.2f}%")

        # Get top 5 unique fighters closest to this style
        cluster_df = df[df['Style'] == unique_label]
        top_fighters = (
            cluster_df.groupby('fighter')
            .size()
            .sort_values(ascending=False)
            .head(5)
        )
        print("Top 5 unique fighters most frequently classified with this style:")
        for fighter, count in top_fighters.items():
            last_fight = cluster_df[cluster_df['fighter'] == fighter].iloc[-1]
            print(f"{fighter} (last fight date: {last_fight['fight_date']}): {count} fights")
        print()

    return df

if __name__ == "__main__":
    create_fighter_styles()
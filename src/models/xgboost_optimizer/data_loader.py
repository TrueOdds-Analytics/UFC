"""
Data loading and preprocessing functions.
"""
import pandas as pd
from config import TRAIN_DATA_PATH, VAL_DATA_PATH, CATEGORY_COLUMNS


def load_and_preprocess_data(odds_type='close_odds'):
    """
    Load and preprocess the training and validation data.

    Args:
        odds_type: Which odds to use in the model: 'open_odds', 'close_odds', 'drop_open' or None

    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    print(f"Loading data with odds type: {odds_type}")

    # Load data from CSV files
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    val_data = pd.read_csv(VAL_DATA_PATH)

    # Extract target variable
    y_train = train_data['winner']
    y_val = val_data['winner']

    # Remove target from feature sets
    X_train = train_data.drop(['winner'], axis=1)
    X_val = val_data.drop(['winner'], axis=1)

    # Determine columns to drop based on odds_type
    if odds_type == 'open_odds':
        columns_to_drop = [
            'fighter_a', 'fighter_b', 'current_fight_date',
            'current_fight_closing_odds', 'current_fight_closing_odds_b',
            'current_fight_closing_odds_ratio', 'current_fight_closing_odds_diff'
        ]
    elif odds_type == 'close_odds':
        columns_to_drop = ['fighter_a', 'fighter_b', 'current_fight_date']
    elif odds_type == 'drop_open':
        base_columns_to_drop = ['fighter_a', 'fighter_b', 'current_fight_date']
        # Find columns with 'open' in the name
        open_columns = [col for col in X_train.columns if 'open' in col.lower()]
        # Combine the lists
        columns_to_drop = base_columns_to_drop + open_columns
    else:
        columns_to_drop = list(X_train.columns[X_train.columns.str.contains('odd', case=False)]) + [
            'fighter_a', 'fighter_b', 'current_fight_date'
        ]

    # Drop specified columns
    X_train = X_train.drop(columns=columns_to_drop)
    X_val = X_val.drop(columns=columns_to_drop)

    # Shuffle data (with consistent random state for reproducibility)
    X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
    y_train = y_train.sample(frac=1, random_state=42).reset_index(drop=True)
    X_val = X_val.sample(frac=1, random_state=42).reset_index(drop=True)
    y_val = y_val.sample(frac=1, random_state=42).reset_index(drop=True)

    # Convert categorical columns
    for df in [X_train, X_val]:
        # Only convert columns that exist in the dataframe
        cat_cols = [col for col in CATEGORY_COLUMNS if col in df.columns]
        df[cat_cols] = df[cat_cols].astype("category")

    print(f"Data loaded and preprocessed. Training features: {X_train.shape[1]}")
    return X_train, X_val, y_train, y_val

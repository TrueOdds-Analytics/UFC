import os
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from data_manipulation.helper import *

from data_manipulation.helper import calculate_complementary_odd


def load_and_preprocess_data(data):
    category_columns = [col for col in data.columns if col.endswith(('fight_1', 'fight_2', 'fight_3'))]
    data[category_columns] = data[category_columns].astype("category")
    return data


def specific_matchup(file_path, fighter_a, fighter_b, current_fight_data, n_past_fights=3, output_dir=''):
    df = pd.read_csv(file_path)
    df['fight_date'] = pd.to_datetime(df['fight_date'])

    columns_to_exclude = ['fighter', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
                          'result', 'winner', 'weight_class', 'scheduled_rounds',
                          'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']

    features_to_include = [col for col in df.columns if
                           col not in columns_to_exclude and col != 'age' and not col.endswith('_age')]

    # Filter past fights for each fighter before the current fight date
    current_fight_date = pd.to_datetime(current_fight_data['current_fight_date'])
    fighter_df = df[(df['fighter'].str.lower() == fighter_a.lower()) & (df['fight_date'] < current_fight_date)] \
        .sort_values(by='fight_date', ascending=False).head(n_past_fights)
    opponent_df = df[(df['fighter'].str.lower() == fighter_b.lower()) & (df['fight_date'] < current_fight_date)] \
        .sort_values(by='fight_date', ascending=False).head(n_past_fights)

    if len(fighter_df) < n_past_fights or len(opponent_df) < n_past_fights:
        print(f"Not enough past fight data for {fighter_a} or {fighter_b}")
        return None

    fighter_features = fighter_df[features_to_include].mean().values
    opponent_features = opponent_df[features_to_include].mean().values

    # Get last n_past_fights results
    results_fighter = fighter_df[['result', 'winner', 'weight_class', 'scheduled_rounds']].head(
        n_past_fights).values.flatten()
    results_opponent = opponent_df[['result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']].head(
        n_past_fights).values.flatten()

    # Pad results if necessary
    results_fighter = np.pad(results_fighter, (0, n_past_fights * 4 - len(results_fighter)), 'constant',
                             constant_values=np.nan)
    results_opponent = np.pad(results_opponent, (0, n_past_fights * 4 - len(results_opponent)), 'constant',
                              constant_values=np.nan)

    # Process open odds
    odds_a = current_fight_data.get('open_odds', np.nan)
    odds_b = current_fight_data.get('open_odds_b', np.nan)

    if pd.notna(odds_a) and pd.notna(odds_b):
        odds_a = round_to_nearest_1(odds_a)
        odds_b = round_to_nearest_1(odds_b)
        current_fight_odds = [odds_a, odds_b]
        current_fight_odds_diff = odds_a - odds_b
        current_fight_odds_ratio = odds_a / odds_b if odds_b != 0 else 0
    elif pd.notna(odds_a):
        odds_a = round_to_nearest_1(odds_a)
        odds_b = calculate_complementary_odd(odds_a)
        current_fight_odds = [odds_a, odds_b]
        current_fight_odds_diff = odds_a - odds_b
        current_fight_odds_ratio = odds_a / odds_b if odds_b != 0 else 0
    elif pd.notna(odds_b):
        odds_b = round_to_nearest_1(odds_b)
        odds_a = calculate_complementary_odd(odds_b)
        current_fight_odds = [odds_a, odds_b]
        current_fight_odds_diff = odds_a - odds_b
        current_fight_odds_ratio = odds_a / odds_b if odds_b != 0 else 0
    else:
        current_fight_odds = [-110, -110]
        current_fight_odds_diff = 0
        current_fight_odds_ratio = 1

    # Process closing odds
    closing_odds_a = current_fight_data.get('closing_range_end', np.nan)
    closing_odds_b = current_fight_data.get('closing_range_end_b', np.nan)

    if pd.notna(closing_odds_a) and pd.notna(closing_odds_b):
        closing_odds_a = round_to_nearest_1(closing_odds_a)
        closing_odds_b = round_to_nearest_1(closing_odds_b)
        current_fight_closing_odds = [closing_odds_a, closing_odds_b]
        current_fight_closing_odds_diff = closing_odds_a - closing_odds_b
        current_fight_closing_odds_ratio = closing_odds_a / closing_odds_b if closing_odds_b != 0 else 0
    elif pd.notna(closing_odds_a):
        closing_odds_a = round_to_nearest_1(closing_odds_a)
        closing_odds_b = calculate_complementary_odd(closing_odds_a)
        current_fight_closing_odds = [closing_odds_a, closing_odds_b]
        current_fight_closing_odds_diff = closing_odds_a - closing_odds_b
        current_fight_closing_odds_ratio = closing_odds_a / closing_odds_b if closing_odds_b != 0 else 0
    elif pd.notna(closing_odds_b):
        closing_odds_b = round_to_nearest_1(closing_odds_b)
        closing_odds_a = calculate_complementary_odd(closing_odds_b)
        current_fight_closing_odds = [closing_odds_a, closing_odds_b]
        current_fight_closing_odds_diff = closing_odds_a - closing_odds_b
        current_fight_closing_odds_ratio = closing_odds_a / closing_odds_b if closing_odds_b != 0 else 0
    else:
        current_fight_closing_odds = [-110, -110]
        current_fight_closing_odds_diff = 0
        current_fight_closing_odds_ratio = 1

    # Calculate fighter stats using helper functions
    def calculate_fighter_stats(fighter_name):
        fighter_all_fights = df[(df['fighter'] == fighter_name) & (df['fight_date'] < current_fight_date)] \
            .sort_values(by='fight_date', ascending=False)
        fighter_recent_fights = fighter_all_fights.head(n_past_fights)

        if fighter_recent_fights.empty:
            return np.nan, np.nan, np.nan, np.nan, np.nan

        last_fight_date = fighter_recent_fights['fight_date'].iloc[0]
        last_known_age = fighter_recent_fights['age'].iloc[0]
        first_fight_date = fighter_all_fights['fight_date'].iloc[-1]  # Earliest fight date

        days_since_last_fight = (current_fight_date - last_fight_date).days
        current_age = last_known_age + days_since_last_fight / 365.25
        years_of_experience = (current_fight_date - first_fight_date).days / 365.25

        win_streak = fighter_recent_fights['win_streak'].iloc[0]
        loss_streak = fighter_recent_fights['loss_streak'].iloc[0]

        most_recent_result = fighter_recent_fights['winner'].iloc[0]
        if most_recent_result == 1:
            win_streak += 1
            loss_streak = 0
        elif most_recent_result == 0:
            loss_streak += 1
            win_streak = 0

        return current_age, years_of_experience, days_since_last_fight, win_streak, loss_streak

    age_a, exp_a, days_a, win_streak_a, loss_streak_a = calculate_fighter_stats(fighter_a)
    age_b, exp_b, days_b, win_streak_b, loss_streak_b = calculate_fighter_stats(fighter_b)

    age_diff = age_a - age_b
    age_ratio = age_a / age_b if age_b != 0 else 0

    exp_diff = exp_a - exp_b
    exp_ratio = exp_a / exp_b if exp_b != 0 else 1

    days_diff = days_a - days_b
    days_ratio = days_a / days_b if days_b != 0 else 1

    win_streak_diff = win_streak_a - win_streak_b
    win_streak_ratio = win_streak_a / win_streak_b if win_streak_b != 0 else 1

    loss_streak_diff = loss_streak_a - loss_streak_b
    loss_streak_ratio = loss_streak_a / loss_streak_b if loss_streak_b != 0 else 1

    # Use pre-fight ELO
    elo_a = fighter_df['fight_outcome_elo'].iloc[0]
    elo_b = opponent_df['fight_outcome_elo'].iloc[0]
    elo_diff = elo_a - elo_b
    elo_a_win_chance = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    elo_b_win_chance = 1 - elo_a_win_chance
    elo_ratio = elo_a / elo_b if elo_b != 0 else 0

    elo_stats = [
        elo_a, elo_b, elo_diff,
        elo_a_win_chance, elo_b_win_chance, elo_ratio
    ]

    feature_columns = [f"{feature}_fighter_avg_last_{n_past_fights}" for feature in features_to_include] + \
                      [f"{feature}_fighter_b_avg_last_{n_past_fights}" for feature in features_to_include]

    results_columns = []
    for i in range(1, n_past_fights + 1):
        results_columns += [f"result_fight_{i}", f"winner_fight_{i}", f"weight_class_fight_{i}",
                            f"scheduled_rounds_fight_{i}",
                            f"result_b_fight_{i}", f"winner_b_fight_{i}", f"weight_class_b_fight_{i}",
                            f"scheduled_rounds_b_fight_{i}"]

    odds_age_columns = ['current_fight_open_odds', 'current_fight_open_odds_b', 'current_fight_open_odds_diff',
                        'current_fight_open_odds_ratio',
                        'current_fight_closing_odds', 'current_fight_closing_odds_b', 'current_fight_closing_odds_diff',
                        'current_fight_closing_odds_ratio',
                        'current_fight_age', 'current_fight_age_b', 'current_fight_age_diff', 'current_fight_age_ratio']

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

    column_names = ['fighter', 'fighter_b'] + feature_columns + results_columns + odds_age_columns + new_columns + [
        'current_fight_date']

    combined_features = np.concatenate([
        fighter_features, opponent_features, results_fighter, results_opponent,
        current_fight_odds, [current_fight_odds_diff, current_fight_odds_ratio],
        current_fight_closing_odds, [current_fight_closing_odds_diff, current_fight_closing_odds_ratio],
        [age_a, age_b, age_diff, age_ratio],
        elo_stats,
        [win_streak_a, win_streak_b, win_streak_diff, win_streak_ratio],
        [loss_streak_a, loss_streak_b, loss_streak_diff, loss_streak_ratio],
        [exp_a, exp_b, exp_diff, exp_ratio],
        [days_a, days_b, days_diff, days_ratio]
    ])

    matchup_df = pd.DataFrame([combined_features], columns=column_names[2:-1])
    matchup_df.insert(0, 'fighter', fighter_a)
    matchup_df.insert(1, 'fighter_b', fighter_b)
    matchup_df['current_fight_date'] = current_fight_data['current_fight_date']

    # Generalized column renaming
    def rename_column(col):
        if col == 'fighter':
            return 'fighter_a'
        if 'fighter' in col and not col.startswith('fighter'):
            if 'b_fighter_b' in col:
                return col.replace('b_fighter_b', 'fighter_b_opponent')
            elif 'b_fighter' in col:
                return col.replace('b_fighter', 'fighter_a_opponent')
            elif 'fighter' in col and 'fighter_b' not in col:
                return col.replace('fighter', 'fighter_a')
        return col

    matchup_df.columns = [rename_column(col) for col in matchup_df.columns]

    # Create diff and ratio columns
    diff_columns = {}
    ratio_columns = {}

    for feature in features_to_include:
        col_a = f"{feature}_fighter_a_avg_last_{n_past_fights}"
        col_b = f"{feature}_fighter_b_avg_last_{n_past_fights}"

        if col_a in matchup_df.columns and col_b in matchup_df.columns:
            diff_columns[f"matchup_{feature}_diff_avg_last_{n_past_fights}"] = matchup_df[col_a] - matchup_df[col_b]
            ratio_columns[f"matchup_{feature}_ratio_avg_last_{n_past_fights}"] = matchup_df[col_a] / matchup_df[
                col_b].replace(0, 1)

    # Add new columns
    matchup_df = pd.concat([matchup_df, pd.DataFrame(diff_columns), pd.DataFrame(ratio_columns)], axis=1)

    # Read the test file to get the correct column order
    test_file_path = 'data/train test data/test_data.csv'
    test_df = pd.read_csv(test_file_path, nrows=0)  # Read only the header
    correct_columns = test_df.columns.tolist()

    # Ensure all columns from test data are present
    for col in correct_columns:
        if col not in matchup_df.columns:
            matchup_df[col] = np.nan

    matchup_df = matchup_df[correct_columns]

    output_file = os.path.join(output_dir, 'specific_matchup_data.csv')
    matchup_df.to_csv(output_file, index=False)
    print(f"Matchup data saved to {output_file}")

    return matchup_df


def load_model(model_path, model_type='xgboost'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(enable_categorical=True)
            model.load_model(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise


def ensemble_prediction(matchup_df, model_dir, val_data_path, use_calibration=True):
    model_files = [
        'model_0.6616_auc_diff_0.0917.json',
        'model_0.6586_auc_diff_0.0906.json',
        'model_0.6586_auc_diff_0.0975.json',
        'model_0.6526_auc_diff_0.0951.json',
        'model_0.6556_auc_diff_0.0846.json'
    ]

    models = []
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        model = load_model(model_path, 'xgboost')
        models.append(model)

    expected_features = models[0].get_booster().feature_names
    X = matchup_df.reindex(columns=expected_features)
    X = load_and_preprocess_data(X)

    if use_calibration:
        val_data = pd.read_csv(val_data_path)
        X_val = val_data.drop(['winner'], axis=1).reindex(columns=expected_features)
        X_val = load_and_preprocess_data(X_val)
        y_val = val_data['winner']

    y_pred_proba_list = []
    for model in models:
        if use_calibration:
            calibrated_model = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')
            calibrated_model.fit(X_val, y_val)
            y_pred_proba = calibrated_model.predict_proba(X)
        else:
            y_pred_proba = model.predict_proba(X)
        y_pred_proba_list.append(y_pred_proba)

    y_pred_proba_avg = np.mean(y_pred_proba_list, axis=0)
    fighter_a_probability = y_pred_proba_avg[0][1]
    fighter_b_probability = 1 - fighter_a_probability

    if fighter_a_probability > fighter_b_probability:
        predicted_winner = matchup_df['fighter_a'].iloc[0]
        winning_probability = fighter_a_probability
    else:
        predicted_winner = matchup_df['fighter_b'].iloc[0]
        winning_probability = fighter_b_probability

    return predicted_winner, winning_probability


if __name__ == "__main__":
    file_path = "data/combined_sorted_fighter_stats.csv"
    fighter_a = "austin hubbard"
    fighter_b = "alexander hernandez"

    current_fight_data = {
        'open_odds': 170,              # Opening odds for fighter_a
        'open_odds_b': -215,           # Opening odds for fighter_b
        'closing_range_end': 174,      # Closing odds for fighter_a
        'closing_range_end_b': -206,   # Closing odds for fighter_b
        'current_fight_date': '2024-10-05'
    }

    output_dir = 'data/matchup data'
    model_dir = 'models/xgboost/jan2024-july2024/125 closed/'
    val_data_path = 'data/train test data/val_data.csv'

    matchup_df = specific_matchup(file_path, fighter_a, fighter_b, current_fight_data, output_dir=output_dir)

    if matchup_df is not None:
        predicted_winner, winning_probability = ensemble_prediction(matchup_df, model_dir, val_data_path)
        print(f"Predicted winner: {predicted_winner}")
        print(f"Winning probability: {winning_probability:.2%}")
    else:
        print("Could not generate matchup data.")


import pandas as pd
import numpy as np
import os


def specific_matchup(file_path, fighter_a, fighter_b, current_fight_data, n_past_fights=3, output_dir=''):
    df = pd.read_csv(file_path)
    df['fight_date'] = pd.to_datetime(df['fight_date'])

    columns_to_exclude = ['fighter', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
                          'result', 'winner', 'weight_class', 'scheduled_rounds',
                          'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']

    features_to_include = [col for col in df.columns if
                           col not in columns_to_exclude and col != 'age' and not col.endswith('_age')]

    fighter_df = df[df['fighter'] == fighter_a].sort_values(by='fight_date', ascending=False)
    opponent_df = df[df['fighter'] == fighter_b].sort_values(by='fight_date', ascending=False)

    if len(fighter_df) < n_past_fights or len(opponent_df) < n_past_fights:
        print(f"Not enough past fight data for {fighter_a} or {fighter_b}")
        return None

    fighter_features = fighter_df.head(n_past_fights)[features_to_include].mean().values
    opponent_features = opponent_df.head(n_past_fights)[features_to_include].mean().values

    results_fighter = fighter_df.head(n_past_fights)[['result', 'winner', 'weight_class', 'scheduled_rounds']].values.flatten()
    results_opponent = opponent_df.head(n_past_fights)[['result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']].values.flatten()

    results_fighter = np.pad(results_fighter, (0, n_past_fights * 4 - len(results_fighter)), 'constant', constant_values=np.nan)
    results_opponent = np.pad(results_opponent, (0, n_past_fights * 4 - len(results_opponent)), 'constant', constant_values=np.nan)

    odds_a, odds_b = current_fight_data['odds']
    odds_diff = odds_a - odds_b
    odds_ratio = odds_a / odds_b if odds_b != 0 else 0
    odds_stats = [odds_a, odds_b, odds_diff, odds_ratio]

    current_fight_date = pd.to_datetime(current_fight_data['current_fight_date'])

    def calculate_fighter_stats(fighter_data):
        last_fight_date = fighter_data['fight_date'].iloc[0]
        last_known_age = fighter_data['age'].iloc[0]
        first_fight_date = fighter_data['fight_date'].iloc[-1]

        days_since_last_fight = (current_fight_date - last_fight_date).days
        current_age = last_known_age + days_since_last_fight / 365.25
        years_of_experience = (current_fight_date - first_fight_date).days / 365.25

        win_streak = fighter_data['win_streak'].iloc[0]
        loss_streak = fighter_data['loss_streak'].iloc[0]

        most_recent_result = fighter_data['winner'].iloc[0]
        if most_recent_result == 1:
            win_streak += 1
            loss_streak = 0
        elif most_recent_result == 0:
            loss_streak += 1
            win_streak = 0

        return current_age, years_of_experience, days_since_last_fight, win_streak, loss_streak

    age_a, exp_a, days_a, win_streak_a, loss_streak_a = calculate_fighter_stats(fighter_df)
    age_b, exp_b, days_b, win_streak_b, loss_streak_b = calculate_fighter_stats(opponent_df)

    age_diff = age_a - age_b
    age_ratio = age_a / age_b if age_b != 0 else 0
    age_stats = [age_a, age_b, age_diff, age_ratio]

    exp_diff = exp_a - exp_b
    exp_ratio = exp_a / exp_b if exp_b != 0 else 0
    exp_stats = [exp_a, exp_b, exp_diff, exp_ratio]

    days_diff = days_a - days_b
    days_ratio = days_a / days_b if days_b != 0 else 0
    days_stats = [days_a, days_b, days_diff, days_ratio]

    win_streak_diff = win_streak_a - win_streak_b
    win_streak_ratio = win_streak_a / win_streak_b if win_streak_b != 0 else 0
    win_streak_stats = [win_streak_a, win_streak_b, win_streak_diff, win_streak_ratio]

    loss_streak_diff = loss_streak_a - loss_streak_b
    loss_streak_ratio = loss_streak_a / loss_streak_b if loss_streak_b != 0 else 0
    loss_streak_stats = [loss_streak_a, loss_streak_b, loss_streak_diff, loss_streak_ratio]

    elo_a = fighter_df['fight_outcome_elo'].iloc[0]
    elo_b = opponent_df['fight_outcome_elo'].iloc[0]
    elo_diff = elo_a - elo_b
    elo_a_win_chance = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    elo_b_win_chance = 1 - elo_a_win_chance
    elo_ratio = elo_a / elo_b if elo_b != 0 else 0
    elo_stats = [elo_a, elo_b, elo_diff, elo_a_win_chance, elo_b_win_chance, elo_ratio]

    combined_features = np.concatenate([
        fighter_features, opponent_features, results_fighter, results_opponent,
        odds_stats, age_stats, elo_stats, exp_stats, days_stats, win_streak_stats, loss_streak_stats
    ])

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
                        'current_fight_age', 'current_fight_age_b', 'current_fight_age_diff',
                        'current_fight_age_ratio']

    elo_columns = [
        'current_fight_pre_fight_elo_a', 'current_fight_pre_fight_elo_b', 'current_fight_pre_fight_elo_diff',
        'current_fight_pre_fight_elo_a_win_chance', 'current_fight_pre_fight_elo_b_win_chance',
        'current_fight_pre_fight_elo_ratio'
    ]

    other_stat_columns = [
        'current_fight_years_experience_a', 'current_fight_years_experience_b', 'current_fight_years_experience_diff',
        'current_fight_years_experience_ratio',
        'current_fight_days_since_last_a', 'current_fight_days_since_last_b', 'current_fight_days_since_last_diff',
        'current_fight_days_since_last_ratio',
        'current_fight_win_streak_a', 'current_fight_win_streak_b', 'current_fight_win_streak_diff',
        'current_fight_win_streak_ratio',
        'current_fight_loss_streak_a', 'current_fight_loss_streak_b', 'current_fight_loss_streak_diff',
        'current_fight_loss_streak_ratio'
    ]

    column_names = ['fighter', 'fighter_b'] + feature_columns + results_columns + odds_age_columns + elo_columns + other_stat_columns

    print(f"Length of combined_features: {len(combined_features)}")
    print(f"Length of column_names: {len(column_names)}")

    matchup_df = pd.DataFrame([combined_features], columns=column_names[2:])
    matchup_df.insert(0, 'fighter', fighter_a)
    matchup_df.insert(1, 'fighter_b', fighter_b)

    matchup_df['current_fight_date'] = current_fight_data['current_fight_date']

    output_file = os.path.join(output_dir, 'specific_matchup_data.csv')
    matchup_df.to_csv(output_file, index=False)
    print(f"Matchup data saved to {output_file}")

    return matchup_df


if __name__ == "__main__":
    file_path = "data/combined_sorted_fighter_stats.csv"
    fighter_a = "alex pereira"
    fighter_b = "jamahal hill"

    current_fight_data = {
        'odds': [-110, -110],
        'current_fight_date': '2024-07-15'
    }

    output_dir = 'data/matchup data'

    matchup = specific_matchup(file_path, fighter_a, fighter_b, current_fight_data, output_dir=output_dir)
    print(matchup)
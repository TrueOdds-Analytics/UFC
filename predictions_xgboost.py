from predict_testset import *

import pandas as pd
import numpy as np
from datetime import datetime


def create_specific_matchup_data(file_path, fighter_name, opponent_name, fight_date, odds_a, odds_b, age_a, age_b,
                                 n_past_fights=3):
    df = pd.read_csv(file_path)
    df['fight_date'] = pd.to_datetime(df['fight_date'])

    fighter_name = fighter_name.lower()
    opponent_name = opponent_name.lower()
    df['fighter'] = df['fighter'].str.lower()
    df['fighter_b'] = df['fighter_b'].str.lower()

    columns_to_exclude = ['fighter', 'id', 'fighter_b', 'fight_date', 'fight_date_b',
                          'result', 'winner', 'weight_class', 'scheduled_rounds',
                          'result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']

    features_to_include = [col for col in df.columns if col not in columns_to_exclude and 'age' not in col.lower()]

    fighter_df = df[(df['fighter'] == fighter_name)].sort_values(by='fight_date', ascending=False).head(n_past_fights)
    opponent_df = df[(df['fighter'] == opponent_name)].sort_values(by='fight_date', ascending=False).head(n_past_fights)

    if len(fighter_df) < n_past_fights or len(opponent_df) < n_past_fights:
        print("Specific matchup failure: One of the fighters doesn't have enough fights.")
        return None

    fighter_features = fighter_df[features_to_include].mean().values
    opponent_features = opponent_df[features_to_include].mean().values

    results_fighter = fighter_df[['result', 'winner', 'weight_class', 'scheduled_rounds']].head(3).values.flatten()
    results_opponent = opponent_df[['result_b', 'winner_b', 'weight_class_b', 'scheduled_rounds_b']].head(
        3).values.flatten()

    fighter_recent = fighter_df.iloc[0]
    opponent_recent = opponent_df.iloc[0]

    current_fight_date = pd.to_datetime(fight_date)
    current_fight_open_odds = odds_a
    current_fight_open_odds_b = odds_b
    current_fight_open_odds_diff = current_fight_open_odds - current_fight_open_odds_b

    current_fight_age = age_a
    current_fight_age_b = age_b
    current_fight_age_diff = current_fight_age - current_fight_age_b

    # Helper functions for Elo calculation
    def get_margin_factor(method):
        margin_factors = {3: 6, 0: 6, 1: 5, 2: 3}
        return margin_factors.get(method, 1.0)

    def get_age_factor(age):
        if age < 27:
            return 1.15
        elif 27 <= age < 32:
            return 1.0
        else:
            return 0.85

    def get_additional_factors(win_streak, loss_streak, years_experience, days_since_last_fight):
        streak_factor = 1 + (win_streak * 0.02) - (loss_streak * 0.02)
        experience_factor = min(1 + (years_experience * 0.01), 1.2)
        inactivity_factor = max(1 - (days_since_last_fight / 365 * 0.1), 0.9)
        return streak_factor * experience_factor * inactivity_factor

    weight_class_map = {
        0: 'Featherweight', 1: 'Lightweight', 2: 'Heavyweight', 3: 'Bantamweight',
        4: 'Welterweight', 5: 'Light Heavyweight', 6: 'Middleweight', 7: 'Catch Weight Bout',
        8: 'Open Weight Bout', 9: 'Flyweight', 10: 'Tournament',
        11: 'UFC Superfight Championship Bout'
    }
    weight_class_factors = {
        'Flyweight': {'ko': 1.3, 'decision': 0.7},
        'Bantamweight': {'ko': 1.25, 'decision': 0.80},
        'Featherweight': {'ko': 1.2, 'decision': 0.9},
        'Lightweight': {'ko': 1.15, 'decision': 0.95},
        'Welterweight': {'ko': 1.1, 'decision': 1.0},
        'Middleweight': {'ko': 1.0, 'decision': 1.0},
        'Light Heavyweight': {'ko': 0.9, 'decision': 1.0},
        'Heavyweight': {'ko': 0.8, 'decision': 1.0},
        'Catch Weight Bout': {'ko': 1.0, 'decision': 1.0},
        'Open Weight Bout': {'ko': 1.0, 'decision': 1.0},
        'Tournament': {'ko': 1.0, 'decision': 1.0},
        'UFC Superfight Championship Bout': {'ko': 1.0, 'decision': 1.0}
    }

    def calculate_elo_change(winner, loser, k=20):
        winner_rating = winner['elo']
        loser_rating = loser['elo']

        weight_class = weight_class_map[winner['weight_class']]
        margin_factor = get_margin_factor(winner['result'])
        is_ko = winner['result'] in [0, 3]
        weight_class_factor = weight_class_factors[weight_class]['ko' if is_ko else 'decision']
        age_factor = get_age_factor(winner['age'])
        additional_factor = get_additional_factors(
            winner['win_streak'], winner['loss_streak'],
            winner['years_of_experience'], winner['days_since_last_fight']
        )

        expected_score = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
        actual_score = 1  # Winner always gets a score of 1
        elo_change = k * margin_factor * weight_class_factor * age_factor * additional_factor * (
                actual_score - expected_score)

        return elo_change

    # Calculate new Elo ratings based on the most recent fight
    if fighter_recent['winner'] == 1:
        elo_change = calculate_elo_change(fighter_recent, opponent_recent)
        current_fight_elo_a = fighter_recent['elo'] + elo_change
        current_fight_elo_b = opponent_recent['elo'] - elo_change
    else:
        elo_change = calculate_elo_change(opponent_recent, fighter_recent)
        current_fight_elo_a = fighter_recent['elo'] - elo_change
        current_fight_elo_b = opponent_recent['elo'] + elo_change

    current_fight_elo_diff = current_fight_elo_a - current_fight_elo_b

    current_fight_elo_a_win_chance = 1 / (1 + 10 ** ((current_fight_elo_b - current_fight_elo_a) / 400))
    current_fight_elo_b_win_chance = 1 - current_fight_elo_a_win_chance
    current_fight_elo_chance_diff = current_fight_elo_a_win_chance - current_fight_elo_b_win_chance

    # Correct win/loss streak calculation
    if fighter_recent['winner'] == 1:
        current_fight_win_streak_a = fighter_recent['win_streak'] + 1
        current_fight_loss_streak_a = 0
    else:
        current_fight_win_streak_a = 0
        current_fight_loss_streak_a = fighter_recent['loss_streak'] + 1

    if opponent_recent['winner'] == 1:
        current_fight_win_streak_b = opponent_recent['win_streak'] + 1
        current_fight_loss_streak_b = 0
    else:
        current_fight_win_streak_b = 0
        current_fight_loss_streak_b = opponent_recent['loss_streak'] + 1

    current_fight_win_streak_diff = current_fight_win_streak_a - current_fight_win_streak_b
    current_fight_loss_streak_diff = current_fight_loss_streak_a - current_fight_loss_streak_b

    fighter_first_fight = df[df['fighter'] == fighter_name].sort_values('fight_date').iloc[0]['fight_date']
    opponent_first_fight = df[df['fighter'] == opponent_name].sort_values('fight_date').iloc[0]['fight_date']
    current_fight_years_experience_a = (current_fight_date - fighter_first_fight).days / 365.25
    current_fight_years_experience_b = (current_fight_date - opponent_first_fight).days / 365.25
    current_fight_years_experience_diff = current_fight_years_experience_a - current_fight_years_experience_b

    current_fight_days_since_last_a = (current_fight_date - fighter_recent['fight_date']).days
    current_fight_days_since_last_b = (current_fight_date - opponent_recent['fight_date']).days
    current_fight_days_since_last_diff = current_fight_days_since_last_a - current_fight_days_since_last_b

    current_fight_info = [
        current_fight_open_odds, current_fight_open_odds_b, current_fight_open_odds_diff,
        current_fight_age, current_fight_age_b, current_fight_age_diff,
        current_fight_elo_a, current_fight_elo_b, current_fight_elo_diff,
        current_fight_elo_a_win_chance, current_fight_elo_b_win_chance, current_fight_elo_chance_diff,
        current_fight_win_streak_a, current_fight_win_streak_b, current_fight_win_streak_diff,
        current_fight_loss_streak_a, current_fight_loss_streak_b, current_fight_loss_streak_diff,
        current_fight_years_experience_a, current_fight_years_experience_b, current_fight_years_experience_diff,
        current_fight_days_since_last_a, current_fight_days_since_last_b, current_fight_days_since_last_diff
    ]

    combined_features = np.concatenate(
        [fighter_features, opponent_features, results_fighter, results_opponent, current_fight_info])

    most_recent_date = max(fighter_df['fight_date'].max(), opponent_df['fight_date'].max())

    matchup_data = [[fighter_name, opponent_name, most_recent_date] + combined_features.tolist() + [current_fight_date]]

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
                    'current_fight_age', 'current_fight_age_b', 'current_fight_age_diff',
                    'current_fight_elo_a', 'current_fight_elo_b', 'current_fight_elo_diff',
                    'current_fight_elo_a_win_chance', 'current_fight_elo_b_win_chance', 'current_fight_elo_chance_diff',
                    'current_fight_win_streak_a', 'current_fight_win_streak_b', 'current_fight_win_streak_diff',
                    'current_fight_loss_streak_a', 'current_fight_loss_streak_b', 'current_fight_loss_streak_diff',
                    'current_fight_years_experience_a', 'current_fight_years_experience_b',
                    'current_fight_years_experience_diff',
                    'current_fight_days_since_last_a', 'current_fight_days_since_last_b',
                    'current_fight_days_since_last_diff',
                    'current_fight_date']

    matchup_df = pd.DataFrame(matchup_data, columns=column_names)

    csv_name = f'specific_matchup_data.csv'
    matchup_df.to_csv(f'data/matchup data/{csv_name}', index=False)

    print("Specific matchup success. Data saved to CSV.")
    return matchup_df

def predict_outcome(model, specific_data, display_data, fighter_name, opponent_name, initial_bankroll=10000,
                    kelly_fraction=0.125,
                    fixed_bet_fraction=0.1, default_bet=0.05, min_odds=-300, confidence_threshold=0.60):
    console = Console(width=160)

    predictions = model.predict(specific_data)
    probabilities = model.predict_proba(specific_data)

    winner = fighter_name if predictions[0] == 1 else opponent_name
    win_probability = probabilities[0][1] if predictions[0] == 1 else probabilities[0][0]

    # Check if the win probability meets the confidence threshold
    if win_probability < confidence_threshold:
        console.print(
            f"[red]Win probability ({win_probability:.2%}) is below the confidence threshold ({confidence_threshold:.2%}). No bet recommended.[/red]")
        return

    odds_a = specific_data['current_fight_open_odds'].values[0]
    odds_b = specific_data['current_fight_open_odds_b'].values[0]
    odds = odds_a if winner == fighter_name else odds_b

    # Check if odds are better than -300
    if odds < min_odds:
        console.print(f"[red]Odds ({odds}) are worse than -300. No bet recommended.[/red]")
        return

    # Fixed Fraction Betting
    fixed_stake = initial_bankroll * fixed_bet_fraction
    fixed_profit = calculate_profit(odds, fixed_stake)
    fixed_units = fixed_stake / (initial_bankroll * 0.01)  # Calculate units

    # Kelly Criterion Betting
    b = odds / 100 if odds > 0 else 100 / abs(odds)
    kelly_bet_size = calculate_kelly_fraction(win_probability, b, kelly_fraction)

    # If Kelly bet size is 0, use default_bet (5% of bankroll)
    if kelly_bet_size == 0:
        kelly_bet_size = default_bet
        console.print("[yellow]Kelly formula suggested 0% bet. Using 1% of bankroll instead.[/yellow]")

    kelly_stake = initial_bankroll * kelly_bet_size
    kelly_profit = calculate_profit(odds, kelly_stake)
    kelly_units = kelly_stake / (initial_bankroll * 0.01)  # Calculate units

    # Create rich text output
    fighter_a = fighter_name.title()
    fighter_b = opponent_name.title()
    fight_date_str = display_data['current_fight_date'].values[0]

    # Convert numpy.datetime64 to string, then to datetime
    fight_date_str = pd.to_datetime(fight_date_str).strftime('%Y-%m-%d')
    fight_date = datetime.datetime.strptime(fight_date_str, '%Y-%m-%d')
    formatted_fight_date = fight_date.strftime('%B %d, %Y')

    fight_info = Panel(
        Group(
            Text(f"Predicted Winner: {winner.title()}", style="green"),
            Text(f"Win Probability: {win_probability:.2%}", style="blue"),
        ),
        title="Fight Prediction",
        expand=True
    )

    fixed_panel = Panel(
        f"Initial Bankroll: ${initial_bankroll:.2f}\n"
        f"Stake: ${fixed_stake:.2f} ({fixed_units:.2f} units)\n"
        f"Potential Profit: ${fixed_profit:.2f}\n"
        f"Potential Bankroll After: ${initial_bankroll + fixed_profit:.2f}\n"
        f"Potential ROI: {(fixed_profit / initial_bankroll) * 100:.2f}%",
        title="Fixed Fraction",
        expand=True
    )
    kelly_panel = Panel(
        f"Initial Bankroll: ${initial_bankroll:.2f}\n"
        f"Stake: ${kelly_stake:.2f} ({kelly_units:.2f} units)\n"
        f"Potential Profit: ${kelly_profit:.2f}\n"
        f"Potential Bankroll After: ${initial_bankroll + kelly_profit:.2f}\n"
        f"Potential ROI: {(kelly_profit / initial_bankroll) * 100:.2f}%",
        title="Kelly",
        expand=True
    )

    main_panel = Panel(
        Group(
            fight_info,
            Columns([fixed_panel, kelly_panel], equal=True, expand=True)
        ),
        title=f"{fighter_a} vs {fighter_b} on {formatted_fight_date}",
        subtitle=f"Odds: {odds}",
        width=100
    )

    console.print(main_panel, style="magenta")


# In the main part of the script, update the function call:
if __name__ == "__main__":
    matchup_data = create_specific_matchup_data(
        file_path='data/combined_sorted_fighter_stats.csv',
        fighter_name='johnny walker',
        opponent_name='volkan oezdemir',
        fight_date='2024-06-22',
        odds_a=-150,
        odds_b=110,
        age_a=32,
        age_b=35,
        n_past_fights=3
    )
    # model_path = os.path.abspath('models/xgboost/jun2022-jun2024/model_0.7632_auc_diff_0.0439.json')
    # model = load_model(model_path)
    #
    # console = Console()
    #
    # while True:
    #     fighter_name = console.input("[cyan]Enter the name of Fighter A (or 'q' to quit): [/cyan]")
    #     if fighter_name.lower() == 'q':
    #         break
    #
    #     opponent_name = console.input("[magenta]Enter the name of Fighter B: [/magenta]")
    #
    #     with console.status("[yellow]Generating matchup data...[/yellow]"):
    #         specific_data = create_specific_matchup_data('data/combined_sorted_fighter_stats.csv', fighter_name,
    #                                                      opponent_name, 3, True)
    #
    #     if specific_data is None:
    #         console.print("[red]Insufficient data for the specified fighters. Please try again.[/red]\n")
    #         continue
    #
    #     display_columns = ['current_fight_date', 'fighter', 'fighter_b']
    #     display_data = specific_data[display_columns]
    #     specific_data = preprocess_data(specific_data)
    #     specific_data = specific_data.drop(display_columns, axis=1)
    #     expected_features = model.get_booster().feature_names
    #     specific_data = specific_data.reindex(columns=expected_features)
    #
    #     predict_outcome(model, specific_data, display_data, fighter_name, opponent_name, 10000, 1,
    #                     0.1, 0.01, -300, 0.50)
    #     console.print()

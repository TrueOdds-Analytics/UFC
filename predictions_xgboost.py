from data_cleaner import create_specific_matchup_data
from predict_testset import *


def predict_outcome(model, specific_data, display_data, fighter_name, opponent_name, initial_bankroll=10000, kelly_fraction=0.125,
                    fixed_bet_fraction=0.1, default_bet=0.05, min_odds=-300, confidence_threshold=0.60):
    console = Console(width=160)

    predictions = model.predict(specific_data)
    probabilities = model.predict_proba(specific_data)

    winner = fighter_name if predictions[0] == 1 else opponent_name
    win_probability = probabilities[0][1] if predictions[0] == 1 else probabilities[0][0]

    # Check if the win probability meets the confidence threshold
    if win_probability < confidence_threshold:
        console.print(f"[red]Win probability ({win_probability:.2%}) is below the confidence threshold ({confidence_threshold:.2%}). No bet recommended.[/red]")
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
    model_path = os.path.abspath('models/xgboost/model_0.6866_342_features_auc_diff_0.0616_good.json')
    model = load_model(model_path)

    console = Console()

    while True:
        fighter_name = console.input("[cyan]Enter the name of Fighter A (or 'q' to quit): [/cyan]")
        if fighter_name.lower() == 'q':
            break

        opponent_name = console.input("[magenta]Enter the name of Fighter B: [/magenta]")

        with console.status("[yellow]Generating matchup data...[/yellow]"):
            specific_data = create_specific_matchup_data('data/combined_sorted_fighter_stats.csv', fighter_name,
                                                         opponent_name, 3, True)

        if specific_data is None:
            console.print("[red]Insufficient data for the specified fighters. Please try again.[/red]\n")
            continue

        display_columns = ['current_fight_date', 'fighter', 'fighter_b']
        display_data = specific_data[display_columns]
        specific_data = preprocess_data(specific_data)
        specific_data = specific_data.drop(display_columns, axis=1)
        expected_features = model.get_booster().feature_names
        specific_data = specific_data.reindex(columns=expected_features)

        predict_outcome(model, specific_data, display_data, fighter_name, opponent_name, 13000, 1,
                        0.1, 0.01, -300, 0.5488)
        console.print()

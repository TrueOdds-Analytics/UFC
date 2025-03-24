"""
Betting utilities for MMA fight prediction
"""
import datetime
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns


def calculate_profit(odds, stake):
    """
    Calculate potential profit based on American odds

    Args:
        odds: American format odds
        stake: Amount wagered

    Returns:
        Potential profit
    """
    if odds < 0:
        return (100 / abs(odds)) * stake
    else:
        return (odds / 100) * stake


def calculate_kelly_fraction(p, b):
    """
    Calculate Kelly criterion optimal bet size

    Args:
        p: Probability of winning
        b: Decimal odds-1 (payout per unit wagered)

    Returns:
        Kelly fraction (0 to 1)
    """
    q = 1 - p
    return max(0, (p - (q / b)))


def calculate_average_odds(open_odds, close_odds):
    """
    Calculate average odds between opening and closing lines

    Args:
        open_odds: Opening odds in American format
        close_odds: Closing odds in American format

    Returns:
        Average odds in American format
    """

    def american_to_decimal(odds):
        return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1

    avg_decimal = (american_to_decimal(open_odds) + american_to_decimal(close_odds)) / 2

    if avg_decimal > 2:
        return round((avg_decimal - 1) * 100)
    else:
        return round(-100 / (avg_decimal - 1))


def print_fight_results(confident_bets):
    """
    Print detailed results for each bet

    Args:
        confident_bets: List of bet result dictionaries
    """
    console = Console(width=160)
    for bet in confident_bets:
        fighter_a = bet['Fighter A'].title()
        fighter_b = bet['Fighter B'].title()
        date_obj = datetime.datetime.strptime(bet['Date'], '%Y-%m-%d')
        formatted_date = date_obj.strftime('%B %d, %Y')

        fixed_available_bankroll_before_bet = float(bet.get('Fixed Fraction Available Bankroll', '0').replace('$', ''))
        fixed_stake = float(bet.get('Fixed Fraction Stake', '0').replace('$', ''))
        fixed_stake_percentage = (
                                         fixed_stake / fixed_available_bankroll_before_bet) * 100 if fixed_available_bankroll_before_bet > 0 else 0

        fixed_panel = Panel(
            f"Starting Bankroll: {bet.get('Fixed Fraction Starting Bankroll', 'N/A')}\n"
            f"Available Bankroll: {bet.get('Fixed Fraction Available Bankroll', 'N/A')}\n"
            f"Stake: {bet.get('Fixed Fraction Stake', 'N/A')}\n"
            f"Stake Percentage: {fixed_stake_percentage:.2f}%\n"
            f"Potential Profit: {bet.get('Fixed Fraction Potential Profit', 'N/A')}\n"
            f"Bankroll After Bet: {bet.get('Fixed Fraction Bankroll After', 'N/A')}\n"
            f"Profit: ${bet.get('Fixed Fraction Profit', 0):.2f}\n"
            f"ROI (of available bankroll): {bet.get('Fixed Fraction ROI', 0):.2f}%",
            title="Fixed Fraction",
            expand=True,
            width=42
        )

        kelly_available_bankroll_before_bet = float(bet.get('Kelly Available Bankroll', '0').replace('$', ''))
        kelly_stake = float(bet.get('Kelly Stake', '0').replace('$', ''))
        kelly_stake_percentage = (
                                         kelly_stake / kelly_available_bankroll_before_bet) * 100 if kelly_available_bankroll_before_bet > 0 else 0

        kelly_panel = Panel(
            f"Starting Bankroll: {bet.get('Kelly Starting Bankroll', 'N/A')}\n"
            f"Available Bankroll: {bet.get('Kelly Available Bankroll', 'N/A')}\n"
            f"Stake: {bet.get('Kelly Stake', 'N/A')}\n"
            f"Stake Percentage: {kelly_stake_percentage:.2f}%\n"
            f"Potential Profit: {bet.get('Kelly Potential Profit', 'N/A')}\n"
            f"Bankroll After Bet: {bet.get('Kelly Bankroll After', 'N/A')}\n"
            f"Profit: ${bet.get('Kelly Profit', 0):.2f}\n"
            f"ROI (of available bankroll): {bet.get('Kelly ROI', 0):.2f}%",
            title="Kelly",
            expand=True,
            width=42
        )

        fight_info = Group(
            Text(f"True Winner: {bet['True Winner'].title()}", style="green"),
            Text(f"Predicted Winner: {bet['Predicted Winner'].title()}", style="blue"),
            Text(f"Confidence: {bet['Confidence']}", style="yellow"),
            Text(f"Models Agreeing: {bet['Models Agreeing']}/5", style="cyan")
        )

        main_panel = Panel(
            Group(
                Panel(fight_info, title="Fight Information"),
                Columns([fixed_panel, kelly_panel], equal=False, expand=False, align="left")
            ),
            title=f"Fight {bet['Fight']}: {fighter_a} vs {fighter_b} on {formatted_date}",
            subtitle=f"Odds: {bet['Odds']}",
            width=89
        )

        console.print(main_panel, style="magenta")
        console.print()


def calculate_daily_roi(daily_bankrolls, initial_bankroll):
    """
    Calculate daily ROI based on bankroll changes

    Args:
        daily_bankrolls: Dictionary mapping dates to bankroll values
        initial_bankroll: Starting bankroll

    Returns:
        Dictionary mapping dates to ROI percentages
    """
    daily_roi = {}
    previous_bankroll = initial_bankroll

    for date, bankroll in sorted(daily_bankrolls.items()):
        daily_profit = bankroll - previous_bankroll
        daily_roi[date] = (daily_profit / previous_bankroll) * 100
        previous_bankroll = bankroll

    return daily_roi


def print_daily_roi(daily_fixed_roi, daily_kelly_roi):
    """
    Print a table of daily ROI values

    Args:
        daily_fixed_roi: Dictionary of fixed betting daily ROI
        daily_kelly_roi: Dictionary of Kelly betting daily ROI
    """
    console = Console()
    console.print("\nDaily ROI:")
    table = Table(title="Daily Return on Investment")
    table.add_column("Date", style="cyan")
    table.add_column("Fixed Fraction ROI", justify="right", style="magenta")
    table.add_column("Kelly ROI", justify="right", style="green")

    for date in sorted(daily_fixed_roi.keys()):
        fixed_roi = f"{daily_fixed_roi[date]:.2f}%"
        kelly_roi = f"{daily_kelly_roi[date]:.2f}%"
        table.add_row(date, fixed_roi, kelly_roi)

    console.print(table)


def calculate_monthly_roi(daily_bankrolls, initial_bankroll, kelly):
    """
    Calculate monthly ROI and profits

    Args:
        daily_bankrolls: Dictionary mapping dates to bankroll values
        initial_bankroll: Starting bankroll
        kelly: Boolean flag for Kelly vs Fixed output formatting

    Returns:
        Tuple of (monthly_roi, monthly_profit, total_roi)
    """
    monthly_roi = {}
    monthly_profit = {}
    current_month = None
    current_bankroll = initial_bankroll
    month_start_bankroll = initial_bankroll
    total_profit = 0
    if kelly:
        print("\nDetailed Kelly ROI Calculation:")
    else:
        print("\nDetailed Fixed Fraction ROI Calculation:")
    print(f"{'Month':<10}{'Profit':<15}{'ROI':<10}{'Start Bankroll':<20}{'End Bankroll':<20}")
    print("-" * 80)

    sorted_dates = sorted(daily_bankrolls.keys())
    for date in sorted_dates:
        bankroll = daily_bankrolls[date]
        month = date[:7]  # Extract YYYY-MM

        if month != current_month:
            if current_month is not None:
                profit = current_bankroll - month_start_bankroll
                monthly_profit[current_month] = profit
                total_profit += profit
                roi = (profit / month_start_bankroll) * 100
                monthly_roi[current_month] = roi
                print(
                    f"{current_month:<10}${profit:<14.2f}{roi:<10.2f}${month_start_bankroll:<19.2f}${current_bankroll:<19.2f}")

            current_month = month
            month_start_bankroll = current_bankroll

        current_bankroll = bankroll

    # Handle the last month
    if current_month is not None:
        profit = current_bankroll - month_start_bankroll
        monthly_profit[current_month] = profit
        total_profit += profit
        roi = (profit / month_start_bankroll) * 100
        monthly_roi[current_month] = roi
        print(
            f"{current_month:<10}${profit:<14.2f}{roi:<10.2f}${month_start_bankroll:<19.2f}${current_bankroll:<19.2f}")

    total_roi = (total_profit / initial_bankroll) * 100
    sum_monthly_roi = sum(monthly_roi.values())

    print("-" * 80)
    print(f"{'Total':<10}${total_profit:<14.2f}{total_roi:<10.2f}")
    print(f"\nSum of monthly ROIs: {sum_monthly_roi:.2f}%")
    print(f"Total ROI: {total_roi:.2f}%")
    print(f"Difference: {total_roi - sum_monthly_roi:.2f}%")

    print("\nDebug Information:")
    print(f"Number of events in dataset: {len(sorted_dates)}")
    print(f"First date: {sorted_dates[0]}, Last date: {sorted_dates[-1]}")
    print(f"Initial bankroll: ${initial_bankroll:.2f}, Final bankroll: ${current_bankroll:.2f}")

    return monthly_roi, monthly_profit, total_roi


def print_betting_results(total_fights, confident_predictions, correct_confident_predictions,
                          fixed_total_bets, fixed_correct_bets, initial_bankroll, fixed_final_bankroll,
                          fixed_total_volume, confidence_threshold, kelly_final_bankroll, kelly_total_volume,
                          kelly_correct_bets, kelly_total_bets, kelly_fraction, fixed_bet_fraction,
                          earliest_fight_date, fixed_monthly_profit, kelly_monthly_profit):
    """
    Print comprehensive betting results summary

    Args:
        Various betting performance metrics
    """
    confident_accuracy = correct_confident_predictions / confident_predictions if confident_predictions > 0 else 0
    fixed_accuracy = fixed_correct_bets / fixed_total_bets if fixed_total_bets > 0 else 0
    kelly_accuracy = kelly_correct_bets / kelly_total_bets if kelly_total_bets > 0 else 0

    fixed_net_profit = sum(fixed_monthly_profit.values())
    fixed_roi = (fixed_net_profit / initial_bankroll) * 100

    kelly_net_profit = sum(kelly_monthly_profit.values())
    kelly_roi = (kelly_net_profit / initial_bankroll) * 100

    avg_fixed_bet_size = fixed_total_volume / fixed_total_bets if fixed_total_bets > 0 else 0
    avg_kelly_bet_size = kelly_total_volume / kelly_total_bets if kelly_total_bets > 0 else 0

    fixed_scale = (avg_fixed_bet_size / fixed_net_profit) * 100 if fixed_net_profit != 0 else 0
    kelly_scale = (avg_kelly_bet_size / kelly_net_profit) * 100 if kelly_net_profit != 0 else 0

    earliest_date = datetime.datetime.strptime(earliest_fight_date, '%Y-%m-%d')
    today = datetime.datetime.now()
    months_diff = (today.year - earliest_date.year) * 12 + today.month - earliest_date.month

    console = Console()

    console.print(Panel(f"Best confidence threshold: {confidence_threshold:.4f}\n"
                        f"Best Kelly ROI: {kelly_roi:.2f}%",
                        title="Optimal Parameters"))

    table = Table(title=f"Betting Results ({confidence_threshold:.0%} confidence threshold)")
    table.add_column("Metric", style="cyan")
    table.add_column("Fixed Fraction", justify="right", style="magenta")
    table.add_column("Kelly", justify="right", style="green")

    table.add_row("Total fights", str(total_fights), str(total_fights))
    table.add_row("Confident predictions", str(confident_predictions), str(confident_predictions))
    table.add_row("Correct predictions", str(correct_confident_predictions), str(correct_confident_predictions))
    table.add_row("Total bets", str(fixed_total_bets), str(kelly_total_bets))
    table.add_row("Correct bets", str(fixed_correct_bets), str(kelly_correct_bets))
    table.add_row("Betting Accuracy", f"{fixed_accuracy:.2%}", f"{kelly_accuracy:.2%}")
    table.add_row("Confident Prediction Accuracy", f"{confident_accuracy:.2%}", f"{confident_accuracy:.2%}")

    console.print(table)

    fixed_panel = Panel(
        f"Initial bankroll: ${initial_bankroll:.2f}\n"
        f"Final bankroll: ${fixed_final_bankroll:.2f}\n"
        f"Total volume: ${fixed_total_volume:.2f}\n"
        f"Net profit: ${fixed_net_profit:.2f}\n"
        f"ROI: {fixed_roi:.2f}%\n"
        f"Fixed bet fraction: {fixed_bet_fraction:.3f}\n"
        f"Average bet size: ${avg_fixed_bet_size:.2f}\n"
        f"Risk: {fixed_scale:.2f}%",
        title="Fixed Fraction Betting Results"
    )

    kelly_panel = Panel(
        f"Initial bankroll: ${initial_bankroll:.2f}\n"
        f"Final bankroll: ${kelly_final_bankroll:.2f}\n"
        f"Total volume: ${kelly_total_volume:.2f}\n"
        f"Net profit: ${kelly_net_profit:.2f}\n"
        f"ROI: {kelly_roi:.2f}%\n"
        f"Kelly fraction: {kelly_fraction:.3f}\n"
        f"Average bet size: ${avg_kelly_bet_size:.2f}\n"
        f"Risk: {kelly_scale:.2f}%",
        title="Kelly Criterion Betting Results"
    )

    console.print(Columns([fixed_panel, kelly_panel]))
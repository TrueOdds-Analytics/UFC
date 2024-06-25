import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import random
import pandas as pd


class BestFightOddsScraperSelenium:
    def __init__(self, fighters):
        self.fighters = fighters
        self.driver = self.initialize_driver()

    def initialize_driver(self):
        # Initialize the Selenium webdriver (e.g., Chrome)
        driver = webdriver.Chrome()  # Make sure you have the appropriate webdriver installed
        return driver

    def clean_movement(self, movement):
        # Extract numerical value and sign using regex
        match = re.search(r'([+-]?\d+(\.\d+)?)', movement)
        if match:
            movement_value = match.group(1)
            try:
                movement_value = float(movement_value) / 100  # Convert to a decimal
            except ValueError:
                print(f"Failed to convert movement: {movement_value}")  # Debug output
                movement_value = None  # If conversion fails, set it to None
        else:
            print(f"No numerical value found in movement: {movement}")  # Debug output
            movement_value = None
        return movement_value

    def scrape(self):
        odds_data = []

        for fighter in self.fighters:
            # Navigate to the search page
            search_url = "https://www.bestfightodds.com/search"
            self.driver.get(search_url)

            # Find the search input field and enter the fighter's name
            search_input = self.driver.find_element(By.XPATH, "//input[@name='query']")
            search_input.send_keys(fighter)

            # Submit the search form
            search_button = self.driver.find_element(By.XPATH, "//input[@type='submit']")
            search_button.click()

            try:
                # Wait for the search results to load
                search_results = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//table[@class='content-list']"))
                )

                # Find the first fighter link in the search results
                fighter_link = search_results.find_element(By.XPATH, ".//a[contains(@href, '/fighters/')]")
                fighter_link.click()  # Click on the fighter link

                # Wait for the odds table to load
                odds_table = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//table[@class='team-stats-table']"))
                )

                # Extract the odds data from the table
                rows = odds_table.find_elements(By.XPATH, ".//tr")
                for row in rows[1:]:  # Skip the header row
                    try:
                        matchup_link = row.find_element(By.XPATH, ".//th[@class='oppcell']/a")
                        matchup = matchup_link.text
                    except NoSuchElementException:
                        continue  # Skip this row if the matchup link is not found

                    try:
                        open_odds = row.find_element(By.XPATH, ".//td[@class='moneyline']/span").text
                    except NoSuchElementException:
                        open_odds = ""  # Assign an empty string if the open odds element is not found

                    try:
                        closing_range_low = row.find_element(By.XPATH, ".//td[@class='moneyline'][2]/span").text
                        closing_range_high = row.find_element(By.XPATH, ".//td[@class='moneyline'][3]/span").text
                    except NoSuchElementException:
                        closing_range_low = closing_range_high = ""  # Assign empty strings if the closing range elements are not found

                    closing_range_start = closing_range_low
                    closing_range_end = closing_range_high

                    try:
                        movement = row.find_element(By.XPATH, ".//td[@class='change-cell']/span").text
                        print(f"Raw movement: {movement}")  # Debug output
                        movement = self.clean_movement(movement)
                        print(f"Cleaned movement: {movement}")  # Debug output
                    except NoSuchElementException:
                        movement = None  # Assign None if the movement element is not found

                    try:
                        event_link = row.find_element(By.XPATH, ".//td[@class='item-non-mobile']/a")
                        event = event_link.text
                    except NoSuchElementException:
                        event = ""  # Assign an empty string if the event element is not found

                    odds_data.append({
                        "Matchup": matchup,
                        "Open": open_odds,
                        "Closing Range Start": closing_range_start,
                        "Closing Range End": closing_range_end,
                        "Movement": movement,
                        "Event": event
                    })

            except TimeoutException:
                print(f"No odds data found for {fighter}")

            # Add a random delay between requests to avoid overwhelming the server
            time.sleep(random.uniform(1, 3))

        self.driver.quit()

        # Convert the odds data to a pandas DataFrame
        df = pd.DataFrame(odds_data)

        # Save the DataFrame to an Excel file
        df.to_csv("data/fight_odds.csv", index=False)

        return df


def clean_fight_odds_from_csv(input_csv_path, output_csv_path):
    # Read the CSV file
    fight_odds_df = pd.read_csv(input_csv_path)

    # Remove rows where Event doesn't contain 'UFC'
    fight_odds_df = fight_odds_df[
        fight_odds_df['Event'].str.contains('UFC', case=False, na=False) | fight_odds_df['Event'].isna()]

    # Create a list to store the modified rows
    modified_rows = []

    # Iterate through the DataFrame
    i = 0
    while i < len(fight_odds_df):
        row1 = fight_odds_df.iloc[i].copy()

        if pd.notna(row1['Event']):
            # This is a row with Event information
            modified_rows.append(row1)

            # Check if there's a next row
            if i + 1 < len(fight_odds_df):
                row2 = fight_odds_df.iloc[i + 1].copy()
                # Carry over Movement and Event from row1 to row2
                row2['Movement'] = row1['Movement']
                row2['Event'] = row1['Event']
                modified_rows.append(row2)
                i += 2  # Move to the next pair
            else:
                i += 1  # Move to the next row if there's no pair
        else:
            # This is a row without Event information, skip it
            i += 1

    # Create the modified DataFrame
    modified_df = pd.DataFrame(modified_rows, columns=fight_odds_df.columns)

    # Save the cleaned and modified DataFrame to a new CSV file
    modified_df.to_csv(output_csv_path, index=False)

    return modified_df


if __name__ == "__main__":
    # combined_rounds_df = pd.read_csv("data/combined_rounds.csv")
    # fighters = combined_rounds_df["fighter"].unique().tolist()
    # fighters = list(set(fighters))
    # scraper = BestFightOddsScraperSelenium(fighters)
    # odds_df = scraper.scrape()
    # print(odds_df)

    # Clean the fight odds data
    input_file = "data/odds data/fight_odds.csv"
    output_file = "data/odds data/cleaned_fight_odds.csv"
    cleaned_odds_df = clean_fight_odds_from_csv(input_file, output_file)

    print(cleaned_odds_df)

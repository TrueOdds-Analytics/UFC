import re
import time
import random
import pandas as pd
import os
import traceback
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options  # For headless option if needed
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    WebDriverException,
)
from datetime import datetime
from rapidfuzz.fuzz import ratio  # Import the ratio function for fuzzy matching


class BestFightOddsScraperSelenium:
    def __init__(self, fighters, existing_odds_df=None):
        self.fighters = fighters
        self.driver = self.initialize_driver()
        self.unprocessed_fighters = []  # List to keep track of unprocessed fighters
        if existing_odds_df is not None and not existing_odds_df.empty:
            self.existing_data = existing_odds_df
            # Ensure 'Event' and 'Matchup' are strings
            self.existing_data['Event'] = self.existing_data['Event'].astype(str)
            self.existing_data['Matchup'] = self.existing_data['Matchup'].astype(str)
            self.existing_matchups = set(
                zip(self.existing_data['Event'], self.existing_data['Matchup'])
            )
        else:
            self.existing_data = pd.DataFrame()
            self.existing_matchups = set()

    def initialize_driver(self):
        # Optionally, run in headless mode
        options = Options()
        # options.add_argument('--headless')  # Uncomment to run in headless mode
        driver = webdriver.Chrome(options=options)
        return driver

    @staticmethod
    def clean_movement(movement):
        match = re.search(r'([+-]?\d+(\.\d+)?)', movement)
        if match:
            try:
                return float(match.group(1)) / 100
            except ValueError:
                print(f"Failed to convert movement: {movement}")
        print(f"No numerical value found in movement: {movement}")
        return None

    def scrape(self):
        odds_data = []
        similarity_threshold = 85  # Adjust the threshold as needed

        for fighter in self.fighters:
            print(f"Processing fighter: {fighter}")
            success = False
            for attempt in range(1, 6):  # Try up to 5 times
                try:
                    print(f"Attempt {attempt} for fighter: {fighter}")
                    self.driver.get("https://www.bestfightodds.com/search")

                    # Wait for the search input field
                    search_input = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.NAME, "query"))
                    )
                    search_input.clear()
                    search_input.send_keys(fighter)

                    # Wait for the search button
                    search_button = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, "//input[@type='submit']"))
                    )
                    search_button.click()

                    # Shorter fixed sleep to allow page load
                    time.sleep(1)

                    # Check if we're on the fighter's page directly
                    try:
                        # Attempt to locate the odds table directly
                        odds_table = WebDriverWait(self.driver, 5).until(
                            EC.presence_of_element_located(
                                (By.XPATH, "//table[@class='team-stats-table']")
                            )
                        )
                        print(f"Directly navigated to fighter page for '{fighter}'")
                        # Proceed to scrape the odds table
                    except TimeoutException:
                        # If odds table not found, check for search results
                        try:
                            search_results = WebDriverWait(self.driver, 5).until(
                                EC.presence_of_element_located(
                                    (By.XPATH, "//table[@class='content-list']")
                                )
                            )
                            # We're on the search results page
                            fighter_links = search_results.find_elements(
                                By.XPATH, ".//a[contains(@href, '/fighters/')]"
                            )
                            fighter_found = False
                            for link in fighter_links:
                                link_text = link.text.strip()
                                similarity = ratio(link_text.lower(), fighter.lower())
                                if similarity >= similarity_threshold:
                                    print(f"Found match: '{link_text}' with similarity {similarity}")
                                    link.click()
                                    fighter_found = True
                                    break
                            if not fighter_found:
                                print(
                                    f"Fighter '{fighter}' not found in search results on attempt {attempt}."
                                )
                                continue  # Try again

                            # Wait for the odds table on fighter's page
                            odds_table = WebDriverWait(self.driver, 10).until(
                                EC.presence_of_element_located(
                                    (By.XPATH, "//table[@class='team-stats-table']")
                                )
                            )
                        except TimeoutException:
                            print(
                                f"No search results or fighter page found for '{fighter}' on attempt {attempt}."
                            )
                            continue  # Try again

                    # Now, odds_table should be available
                    rows = odds_table.find_elements(By.XPATH, ".//tr")
                    for row in rows[1:]:
                        try:
                            matchup = row.find_element(
                                By.XPATH, ".//th[@class='oppcell']/a"
                            ).text
                        except NoSuchElementException:
                            continue

                        try:
                            event_cell = row.find_element(
                                By.XPATH, ".//td[@class='item-non-mobile'][1]"
                            )
                            event = event_cell.find_element(By.XPATH, ".//a").text
                        except NoSuchElementException:
                            event = ""

                        # Check if this event and matchup have already been scraped
                        if (event, matchup) in self.existing_matchups:
                            print(
                                f"Skipping already scraped event: {event}, matchup: {matchup}"
                            )
                            continue

                        # Proceed with scraping
                        try:
                            open_odds = row.find_element(
                                By.XPATH, ".//td[@class='moneyline']/span"
                            ).text
                        except NoSuchElementException:
                            open_odds = ""

                        try:
                            closing_range_low = row.find_element(
                                By.XPATH, ".//td[@class='moneyline'][2]/span"
                            ).text
                            closing_range_high = row.find_element(
                                By.XPATH, ".//td[@class='moneyline'][3]/span"
                            ).text
                        except NoSuchElementException:
                            closing_range_low = closing_range_high = ""

                        try:
                            movement = row.find_element(
                                By.XPATH, ".//td[@class='change-cell']/span"
                            ).text
                            movement = self.clean_movement(movement)
                        except NoSuchElementException:
                            movement = None

                        try:
                            date_cell = row.find_element(
                                By.XPATH,
                                ".//td[@class='item-non-mobile'][@style='padding-left: 20px; color: #767676']",
                            )
                            date = date_cell.text.strip()
                        except NoSuchElementException:
                            date = ""

                        odds_data.append(
                            {
                                "Matchup": matchup,
                                "Open": open_odds,
                                "Closing Range Start": closing_range_low,
                                "Closing Range End": closing_range_high,
                                "Movement": movement,
                                "Event": event,
                                "Date": date,
                            }
                        )
                    success = True  # Scraping was successful
                    print(f"Successfully processed fighter: {fighter} on attempt {attempt}")
                    break  # Exit the retry loop
                except Exception as e:
                    print(
                        f"An error occurred while processing fighter '{fighter}' on attempt {attempt}: {type(e).__name__}: {e}"
                    )
                    traceback.print_exc()
                    # Shorter sleep before next retry
                    time.sleep(random.uniform(0.5, 1))
                    continue  # Try again
            if not success:
                print(f"Failed to process fighter '{fighter}' after 5 attempts. Skipping.")
                self.unprocessed_fighters.append(fighter)
            # Shorter sleep between processing fighters
            time.sleep(random.uniform(0.5, 1))

        self.driver.quit()
        return pd.DataFrame(odds_data)

    @staticmethod
    def parse_custom_date(date_string):
        if pd.isna(date_string):
            return pd.NaT
        date_string = str(date_string)
        date_string = date_string.replace('th', '').replace('st', '').replace('nd', '').replace('rd', '')
        try:
            return datetime.strptime(date_string, '%b %d %Y')
        except ValueError:
            return pd.NaT

    @staticmethod
    def clean_fight_odds_from_csv(input_csv_path, output_csv_path):
        fight_odds_df = pd.read_csv(input_csv_path)

        if 'Date' not in fight_odds_df.columns:
            fight_odds_df['Date'] = ""

        fight_odds_df = fight_odds_df[
            fight_odds_df['Event'].str.contains('UFC', case=False, na=False)
            | fight_odds_df['Event'].isna()
        ]

        modified_rows = []
        i = 0
        while i < len(fight_odds_df):
            row1 = fight_odds_df.iloc[i].copy()

            if pd.notna(row1['Event']):
                modified_rows.append(row1)

                if i + 1 < len(fight_odds_df):
                    row2 = fight_odds_df.iloc[i + 1].copy()
                    row2['Movement'] = row1['Movement']
                    row2['Event'] = row1['Event']
                    row1['Date'] = row2['Date']
                    modified_rows.append(row2)
                    i += 2
                else:
                    i += 1
            else:
                i += 1

        modified_df = pd.DataFrame(modified_rows, columns=fight_odds_df.columns)

        # Convert Date column to datetime using the custom parser
        modified_df['Date'] = modified_df['Date'].apply(BestFightOddsScraperSelenium.parse_custom_date)

        # Remove rows with no valid date entry
        modified_df = modified_df[modified_df['Date'].notna()]

        # Extract fighter names and create a new column
        modified_df['Fighter'] = modified_df['Matchup'].str.split(' vs ', expand=True)[0]

        # Remove duplicate rows based on 'Matchup' and 'Date'
        modified_df = modified_df.drop_duplicates(subset=['Matchup', 'Date'], keep='first')

        # Sort the DataFrame by Fighter name and then by Date
        modified_df = modified_df.sort_values(['Fighter', 'Date'])

        # Drop the temporary Fighter column
        modified_df = modified_df.drop('Fighter', axis=1)

        # Reset the index
        modified_df = modified_df.reset_index(drop=True)

        # Convert Date back to string format for CSV output
        modified_df['Date'] = modified_df['Date'].dt.strftime('%b %d %Y')

        # Drop the "Event" column as requested
        if 'Event' in modified_df.columns:
            modified_df = modified_df.drop('Event', axis=1)

        modified_df.to_csv(output_csv_path, index=False)
        return modified_df


if __name__ == "__main__":
    # Read the combined rounds CSV and extract unique fighter names
    combined_rounds_df = pd.read_csv("../data/combined_rounds.csv")
    fighters = combined_rounds_df["fighter"].unique().tolist()
    fighters = list(set(fighters))

    existing_odds_file = "../data/odds data/fight_odds.csv"
    if os.path.exists(existing_odds_file):
        existing_odds_df = pd.read_csv(existing_odds_file)
    else:
        existing_odds_df = pd.DataFrame()

    scraper = BestFightOddsScraperSelenium(fighters, existing_odds_df)
    odds_df = scraper.scrape()

    if not existing_odds_df.empty:
        combined_odds_df = pd.concat([existing_odds_df, odds_df], ignore_index=True)
    else:
        combined_odds_df = odds_df

    # Remove duplicates if any
    combined_odds_df = combined_odds_df.drop_duplicates(
        subset=['Event', 'Matchup'], keep='first'
    )

    combined_odds_df.to_csv(existing_odds_file, index=False)
    print(combined_odds_df)

    # Clean the fight odds data and drop the Event column in the cleaned CSV
    input_file = existing_odds_file
    output_file = "../data/odds data/cleaned_fight_odds.csv"
    cleaned_odds_df = BestFightOddsScraperSelenium.clean_fight_odds_from_csv(input_file, output_file)

    print(cleaned_odds_df)

    # Print out the fighters that were not processed
    if scraper.unprocessed_fighters:
        print("\nThe following fighters were not processed after 5 attempts:")
        for fighter in scraper.unprocessed_fighters:
            print(f"- {fighter}")
    else:
        print("\nAll fighters were processed successfully.")

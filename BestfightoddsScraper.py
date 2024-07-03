import re
import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from datetime import datetime

class BestFightOddsScraperSelenium:
    def __init__(self, fighters):
        self.fighters = fighters
        self.driver = self.initialize_driver()

    def initialize_driver(self):
        driver = webdriver.Chrome()
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

        for fighter in self.fighters:
            self.driver.get("https://www.bestfightodds.com/search")
            search_input = self.driver.find_element(By.XPATH, "//input[@name='query']")
            search_input.send_keys(fighter)
            search_button = self.driver.find_element(By.XPATH, "//input[@type='submit']")
            search_button.click()

            try:
                search_results = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//table[@class='content-list']"))
                )
                fighter_link = search_results.find_element(By.XPATH, ".//a[contains(@href, '/fighters/')]")
                fighter_link.click()

                odds_table = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.XPATH, "//table[@class='team-stats-table']"))
                )

                rows = odds_table.find_elements(By.XPATH, ".//tr")
                for row in rows[1:]:
                    try:
                        matchup = row.find_element(By.XPATH, ".//th[@class='oppcell']/a").text
                    except NoSuchElementException:
                        continue

                    try:
                        open_odds = row.find_element(By.XPATH, ".//td[@class='moneyline']/span").text
                    except NoSuchElementException:
                        open_odds = ""

                    try:
                        closing_range_low = row.find_element(By.XPATH, ".//td[@class='moneyline'][2]/span").text
                        closing_range_high = row.find_element(By.XPATH, ".//td[@class='moneyline'][3]/span").text
                    except NoSuchElementException:
                        closing_range_low = closing_range_high = ""

                    try:
                        movement = row.find_element(By.XPATH, ".//td[@class='change-cell']/span").text
                        print(f"Raw movement: {movement}")
                        movement = self.clean_movement(movement)
                        print(f"Cleaned movement: {movement}")
                    except NoSuchElementException:
                        movement = None

                    try:
                        event_cell = row.find_element(By.XPATH, ".//td[@class='item-non-mobile'][1]")
                        event = event_cell.find_element(By.XPATH, ".//a").text
                    except NoSuchElementException:
                        event = ""

                    try:
                        date_cell = row.find_element(By.XPATH,
                                                     ".//td[@class='item-non-mobile'][@style='padding-left: 20px; color: #767676']")
                        date = date_cell.text.strip()
                        print(f"Extracted date: {date}")
                    except NoSuchElementException:
                        date = ""

                    odds_data.append({
                        "Matchup": matchup,
                        "Open": open_odds,
                        "Closing Range Start": closing_range_low,
                        "Closing Range End": closing_range_high,
                        "Movement": movement,
                        "Event": event,
                        "Date": date
                    })

            except TimeoutException:
                print(f"No odds data found for {fighter}")

            time.sleep(random.uniform(1, 3))

        self.driver.quit()
        return pd.DataFrame(odds_data)


def parse_custom_date(date_string):
    try:
        return datetime.strptime(date_string, '%b %d %Y')
    except ValueError:
        try:
            return datetime.strptime(
                date_string.replace('th', '').replace('st', '').replace('nd', '').replace('rd', ''), '%b %d %Y')
        except ValueError:
            return pd.NaT


def clean_fight_odds_from_csv(input_csv_path, output_csv_path):
    fight_odds_df = pd.read_csv(input_csv_path)

    if 'Date' not in fight_odds_df.columns:
        fight_odds_df['Date'] = ""

    fight_odds_df = fight_odds_df[
        fight_odds_df['Event'].str.contains('UFC', case=False, na=False) | fight_odds_df['Event'].isna()]

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
    modified_df['Date'] = modified_df['Date'].apply(parse_custom_date)

    # Extract fighter names and create a new column
    modified_df['Fighter'] = modified_df['Matchup'].str.split(' vs ', expand=True)[0]

    # Sort the DataFrame by Fighter name and then by Date
    modified_df = modified_df.sort_values(['Fighter', 'Date'])

    # Drop the temporary Fighter column
    modified_df = modified_df.drop('Fighter', axis=1)

    # Reset the index
    modified_df = modified_df.reset_index(drop=True)

    # Convert Date back to string format for CSV output
    modified_df['Date'] = modified_df['Date'].dt.strftime('%b %d %Y')

    modified_df.to_csv(output_csv_path, index=False)
    return modified_df


if __name__ == "__main__":
    # combined_rounds_df = pd.read_csv("data/combined_rounds.csv")
    # fighters = combined_rounds_df["fighter"].unique().tolist()
    # fighters = list(set(fighters))
    # scraper = BestFightOddsScraperSelenium(fighters)
    # odds_df = scraper.scrape()
    # odds_df.to_csv("data/odds data/fight_odds.csv", index=False)
    # print(odds_df)

    # Clean the fight odds data
    input_file = "data/odds data/fight_odds.csv"
    output_file = "data/odds data/cleaned_fight_odds.csv"
    cleaned_odds_df = clean_fight_odds_from_csv(input_file, output_file)

    print(cleaned_odds_df)
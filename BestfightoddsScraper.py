from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
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
                    cells = row.find_elements(By.XPATH, ".//td")
                    if len(cells) >= 5:
                        matchup_link = row.find_element(By.XPATH, ".//th[@class='oppcell']/a")
                        matchup = matchup_link.text
                        open_odds = row.find_element(By.XPATH, ".//td[@class='moneyline']/span").text
                        closing_range_low = row.find_element(By.XPATH, ".//td[@class='moneyline'][2]/span").text
                        closing_range_high = row.find_element(By.XPATH, ".//td[@class='moneyline'][3]/span").text
                        closing_range = f"{closing_range_low} ... {closing_range_high}"

                        try:
                            movement = row.find_element(By.XPATH, ".//td[@class='change-cell']/span").text
                        except:
                            movement = ""  # Assign an empty string if the movement element is not found

                        try:
                            event_link = row.find_element(By.XPATH, ".//td[@class='item-non-mobile']/a")
                            event = event_link.text
                        except:
                            event = ""  # Assign an empty string if the event element is not found

                        odds_data.append({
                            "Matchup": matchup,
                            "Open": open_odds,
                            "Closing Range": closing_range,
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


# Usage example
fighters = ["Jon Jones"]
scraper = BestFightOddsScraperSelenium(fighters)
odds_df = scraper.scrape()
print(odds_df)

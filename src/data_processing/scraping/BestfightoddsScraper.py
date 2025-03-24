import re
import time
import random
import pandas as pd
import os
import traceback
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from datetime import datetime
from rapidfuzz.fuzz import ratio
import concurrent.futures
from tqdm import tqdm
import threading
import queue


class BestFightOddsScraperSelenium:
    def __init__(self, num_workers=24):
        self.num_workers = num_workers
        self.unprocessed_fighters = []
        self.progress_queue = queue.Queue()
        self.lock = threading.Lock()
        self.total_fighters = 0
        self.processed_count = 0

    def initialize_driver(self):
        options = Options()
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--log-level=3')
        return webdriver.Chrome(options=options)

    @staticmethod
    def clean_movement(movement):
        if pd.isna(movement) or movement == "":
            return None
        match = re.search(r'([+-]?\d+(\.\d+)?)', str(movement))
        if match:
            try:
                return float(match.group(1)) / 100
            except ValueError:
                return None
        return None

    def scrape_fighter(self, fighter, driver):
        odds_data = []
        similarity_threshold = 85

        for attempt in range(1, 6):
            try:
                print(f"\nAttempting to scrape {fighter} (Attempt {attempt})")
                driver.get("https://www.bestfightodds.com/search")

                # Search for fighter
                search_input = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.NAME, "query"))
                )
                search_input.clear()
                search_input.send_keys(fighter)

                search_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//input[@type='submit']"))
                )
                search_button.click()
                time.sleep(1)

                # Find odds table
                try:
                    odds_table = WebDriverWait(driver, 5).until(
                        EC.presence_of_element_located((By.XPATH, "//table[@class='team-stats-table']"))
                    )
                except TimeoutException:
                    try:
                        # Search results page
                        search_results = WebDriverWait(driver, 5).until(
                            EC.presence_of_element_located((By.XPATH, "//table[@class='content-list']"))
                        )
                        fighter_links = search_results.find_elements(By.XPATH, ".//a[contains(@href, '/fighters/')]")

                        # Find matching fighter
                        for link in fighter_links:
                            link_text = link.text.strip()
                            similarity = ratio(link_text.lower(), fighter.lower())
                            if similarity >= similarity_threshold:
                                link.click()
                                break
                        else:
                            print(f"No match found for {fighter}")
                            continue

                        odds_table = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.XPATH, "//table[@class='team-stats-table']"))
                        )
                    except TimeoutException:
                        print(f"No odds table found for {fighter}")
                        continue

                # Process odds table
                rows = odds_table.find_elements(By.XPATH, ".//tr")[1:]
                for row in rows:
                    try:
                        fight_data = {
                            "Matchup": row.find_element(By.XPATH, ".//th[@class='oppcell']/a").text,
                            "Event": "",
                            "Open": "",
                            "Closing Range Start": "",
                            "Closing Range End": "",
                            "Movement": None,
                            "Date": ""
                        }

                        # Get event
                        try:
                            event_cell = row.find_element(By.XPATH, ".//td[@class='item-non-mobile'][1]")
                            fight_data["Event"] = event_cell.find_element(By.XPATH, ".//a").text
                        except NoSuchElementException:
                            pass

                        # Get odds
                        try:
                            fight_data["Open"] = row.find_element(By.XPATH, ".//td[@class='moneyline']/span").text
                            fight_data["Closing Range Start"] = row.find_element(By.XPATH,
                                                                                 ".//td[@class='moneyline'][2]/span").text
                            fight_data["Closing Range End"] = row.find_element(By.XPATH,
                                                                               ".//td[@class='moneyline'][3]/span").text
                        except NoSuchElementException:
                            pass

                        # Get movement
                        try:
                            movement = row.find_element(By.XPATH, ".//td[@class='change-cell']/span").text
                            fight_data["Movement"] = self.clean_movement(movement)
                        except NoSuchElementException:
                            pass

                        # Get date
                        try:
                            date_cell = row.find_element(
                                By.XPATH,
                                ".//td[@class='item-non-mobile'][@style='padding-left: 20px; color: #767676']"
                            )
                            fight_data["Date"] = date_cell.text.strip()
                        except NoSuchElementException:
                            pass

                        odds_data.append(fight_data)

                    except NoSuchElementException:
                        continue

                # Update progress
                with self.lock:
                    self.processed_count += 1
                    self.progress_queue.put(1)

                print(f"Successfully scraped {fighter}")
                return odds_data

            except Exception as e:
                print(f"Error processing {fighter}: {str(e)}")
                if attempt == 5:
                    self.unprocessed_fighters.append(fighter)
                time.sleep(random.uniform(0.5, 1))

        return []

    def process_batch(self, batch):
        driver = self.initialize_driver()
        batch_data = []
        try:
            for fighter in batch:
                fighter_data = self.scrape_fighter(fighter, driver)
                batch_data.extend(fighter_data)
                time.sleep(random.uniform(0.5, 1))
        finally:
            driver.quit()
        return batch_data

    def scrape_all(self, fighters):
        self.total_fighters = len(fighters)
        print(f"\nStarting scrape of {self.total_fighters} fighters using {self.num_workers} workers")

        # Create batches
        batch_size = max(1, len(fighters) // self.num_workers)
        batches = [fighters[i:i + batch_size] for i in range(0, len(fighters), batch_size)]

        # Progress bar setup
        pbar = tqdm(total=self.total_fighters, desc="Processing fighters")

        # Process batches
        all_data = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_batch = {executor.submit(self.process_batch, batch): batch for batch in batches}

            # Monitor progress
            while future_to_batch:
                done, _ = concurrent.futures.wait(future_to_batch, timeout=0.1)

                # Update progress bar
                while True:
                    try:
                        progress = self.progress_queue.get_nowait()
                        pbar.update(progress)
                    except queue.Empty:
                        break

                # Process completed futures
                for future in done:
                    try:
                        batch_data = future.result()
                        all_data.extend(batch_data)
                    except Exception as e:
                        print(f"\nBatch processing error: {str(e)}")
                    future_to_batch.pop(future)

        pbar.close()
        return pd.DataFrame(all_data)

    @staticmethod
    def clean_fight_odds_from_csv(input_csv_path, output_csv_path):
        fight_odds_df = pd.read_csv(input_csv_path)

        if 'Date' not in fight_odds_df.columns:
            fight_odds_df['Date'] = ""

        # Filter for UFC events
        fight_odds_df = fight_odds_df[
            fight_odds_df['Event'].str.contains('UFC', case=False, na=False)
            | fight_odds_df['Event'].isna()
            ]

        # Process rows
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

        # Create final DataFrame
        modified_df = pd.DataFrame(modified_rows, columns=fight_odds_df.columns)

        # Parse dates
        modified_df['Date'] = modified_df['Date'].apply(lambda x: BestFightOddsScraperSelenium.parse_custom_date(x))
        modified_df = modified_df[modified_df['Date'].notna()]

        # -------------------------
        # CHANGED THIS LINE ONLY:
        # -------------------------
        # Instead of '%b %d %Y', use '%Y-%m-%d'
        modified_df['Date'] = modified_df['Date'].dt.strftime('%Y-%m-%d')

        # Sort and clean
        modified_df = modified_df.sort_values(['Matchup', 'Date'])
        if 'Event' in modified_df.columns:
            modified_df = modified_df.drop('Event', axis=1)

        # Remove duplicates based on 'Matchup' and 'Date', keeping the first occurrence only
        modified_df = modified_df.drop_duplicates(subset=['Matchup', 'Date'], keep='first')

        modified_df.to_csv(output_csv_path, index=False)
        return modified_df

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


if __name__ == "__main__":
    start_time = time.time()

    # ---------------------------------------------------------------------
    # Set these flags to control which parts of the process to run:
    #   run_scraping: Scrape data and save raw CSV.
    #   run_cleaning: Clean the raw CSV and save the cleaned version.
    #
    # To run only one part, simply set the other flag to False.
    # ---------------------------------------------------------------------
    run_scraping = True   # Change to False to skip scraping
    run_cleaning = True   # Change to False to skip cleaning

    if run_scraping:
        # Read fighter list
        combined_rounds_df = pd.read_csv("data/processed/combined_rounds.csv")
        fighters = list(set(combined_rounds_df["fighter"].unique().tolist()))

        # For testing (optional):
        # fighters = fighters[:10]

        # Initialize scraper
        scraper = BestFightOddsScraperSelenium(num_workers=24)

        # Scrape data
        odds_df = scraper.scrape_all(fighters)

        # Save raw data
        odds_df.to_csv("data/raw/fight_odds.csv", index=False)
        print("\nRaw data saved!")

        # Print scraping summary
        print("\nScraping Summary:")
        total_processed = len(fighters) - len(scraper.unprocessed_fighters)
        print(f"Total fighters processed: {total_processed}")
        success_rate = (total_processed / len(fighters)) * 100 if fighters else 0
        print(f"Success rate: {success_rate:.2f}%")
        if scraper.unprocessed_fighters:
            print("\nUnprocessed fighters:")
            for fighter in scraper.unprocessed_fighters:
                print(f"- {fighter}")
        else:
            print("\nAll fighters processed successfully!")

    if run_cleaning:
        # Clean and save processed data
        input_file = "data/raw/fight_odds.csv"
        output_file = "data/processed/cleaned_fight_odds.csv"
        cleaned_odds_df = BestFightOddsScraperSelenium.clean_fight_odds_from_csv(input_file, output_file)
        print("\nCleaned data saved!")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal time: {elapsed_time / 60:.2f} minutes")

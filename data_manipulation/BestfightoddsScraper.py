import re
import time
import random
import pandas as pd
import os
import json
import logging
from datetime import datetime
from typing import List, Set, Dict, Optional, Tuple
from dataclasses import dataclass
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from rapidfuzz.fuzz import ratio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)


@dataclass
class FightOdds:
    matchup: str
    open_odds: str
    closing_range_start: str
    closing_range_end: str
    movement: Optional[float]
    event: str
    date: str
    fighter1: str
    fighter2: str


class FightOddsScraper:
    """
    A robust scraper for collecting UFC fight odds from BestFightOdds.com
    """

    def __init__(self,
                 fighters: List[str],
                 existing_odds_df: Optional[pd.DataFrame] = None,
                 headless: bool = False,
                 retry_failed: bool = True):
        self.fighters = self._normalize_fighter_list(fighters)
        self.headless = headless
        self.retry_failed = retry_failed
        self.driver = self._initialize_driver()
        self.wait = WebDriverWait(self.driver, 10)

        # Initialize tracking sets and dictionaries
        self.processed_fighters: Set[str] = set()
        self.failed_fighters: Set[str] = set()
        self.existing_matchups: Set[Tuple[str, str]] = set()
        self.odds_cache: Dict[str, List[FightOdds]] = {}

        # Load existing data if provided
        if existing_odds_df is not None and not existing_odds_df.empty:
            self._load_existing_data(existing_odds_df)

        # Configure delays for rate limiting
        self.base_delay = 2
        self.max_delay = 5

    def _normalize_fighter_list(self, fighters: List[str]) -> List[str]:
        """Normalize and deduplicate fighter names."""
        normalized = set()
        for fighter in fighters:
            norm_name = self._normalize_name(fighter)
            if norm_name:
                normalized.add(norm_name)
        return list(normalized)

    def _normalize_name(self, name: str) -> str:
        """Normalize a fighter name for consistent matching."""
        if not isinstance(name, str):
            return ""
        # Remove special characters and extra spaces
        name = re.sub(r'[^\w\s]', '', name)
        # Convert to lowercase and strip whitespace
        name = name.lower().strip()
        # Replace multiple spaces with single space
        name = re.sub(r'\s+', ' ', name)
        return name

    def _initialize_driver(self) -> webdriver.Chrome:
        """Initialize and configure Chrome WebDriver."""
        options = Options()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--window-size=1920,1080')

        # Use webdriver_manager to handle driver installation
        service = Service(ChromeDriverManager().install())
        return webdriver.Chrome(service=service, options=options)

    def _load_existing_data(self, df: pd.DataFrame) -> None:
        """Load and process existing fight odds data."""
        for _, row in df.iterrows():
            event = str(row.get('Event', '')).strip()
            matchup = str(row.get('Matchup', '')).strip()
            if event and matchup:
                self.existing_matchups.add((event, matchup))
                # Also add reversed matchup
                fighters = matchup.split(' vs ')
                if len(fighters) == 2:
                    reversed_matchup = f"{fighters[1]} vs {fighters[0]}"
                    self.existing_matchups.add((event, reversed_matchup))

    def _parse_fighter_names(self, matchup: str) -> Tuple[str, str]:
        """Extract individual fighter names from a matchup string."""
        fighters = matchup.split(' vs ')
        if len(fighters) == 2:
            return fighters[0].strip(), fighters[1].strip()
        return "", ""

    def _clean_odds(self, odds_str: str) -> str:
        """Clean and normalize odds string."""
        if not odds_str:
            return ""
        # Remove any non-numeric characters except + and -
        odds_str = re.sub(r'[^\d+-]', '', odds_str)
        return odds_str

    def _parse_movement(self, movement_str: str) -> Optional[float]:
        """Parse movement string to float value."""
        if not movement_str:
            return None
        try:
            # Extract numeric value and convert to decimal
            match = re.search(r'([+-]?\d+(\.\d+)?)', movement_str)
            if match:
                return float(match.group(1)) / 100
        except (ValueError, TypeError):
            logging.warning(f"Failed to parse movement value: {movement_str}")
        return None

    def _parse_date(self, date_str: str) -> str:
        """Parse and normalize date string."""
        if not date_str:
            return ""
        # Remove ordinal indicators
        date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
        try:
            # Parse date and return standardized format
            date_obj = datetime.strptime(date_str.strip(), '%b %d %Y')
            return date_obj.strftime('%Y-%m-%d')
        except ValueError:
            logging.warning(f"Failed to parse date: {date_str}")
            return date_str.strip()

    async def search_fighter(self, fighter: str) -> bool:
        """Search for a fighter and navigate to their page."""
        try:
            self.driver.get("https://www.bestfightodds.com/search")

            # Wait for and fill search input
            search_input = self.wait.until(
                EC.presence_of_element_located((By.NAME, "query"))
            )
            search_input.clear()
            search_input.send_keys(fighter)

            # Submit search
            search_button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//input[@type='submit']"))
            )
            search_button.click()

            # Wait for results
            time.sleep(self.base_delay)

            return True
        except Exception as e:
            logging.error(f"Error searching for fighter {fighter}: {str(e)}")
            return False

    def _extract_odds_from_row(self, row) -> Optional[FightOdds]:
        """Extract odds data from a table row."""
        try:
            matchup = row.find_element(By.XPATH, ".//th[@class='oppcell']/a").text
            event = row.find_element(By.XPATH, ".//td[@class='item-non-mobile'][1]/a").text

            # Skip if already processed
            if (event, matchup) in self.existing_matchups:
                return None

            # Extract odds data
            open_odds = self._clean_odds(
                row.find_element(By.XPATH, ".//td[@class='moneyline']/span").text
            )
            closing_start = self._clean_odds(
                row.find_element(By.XPATH, ".//td[@class='moneyline'][2]/span").text
            )
            closing_end = self._clean_odds(
                row.find_element(By.XPATH, ".//td[@class='moneyline'][3]/span").text
            )

            # Extract movement and date
            try:
                movement_str = row.find_element(By.XPATH, ".//td[@class='change-cell']/span").text
                movement = self._parse_movement(movement_str)
            except NoSuchElementException:
                movement = None

            try:
                date = row.find_element(
                    By.XPATH,
                    ".//td[@class='item-non-mobile'][@style='padding-left: 20px; color: #767676']"
                ).text
            except NoSuchElementException:
                date = ""

            # Parse fighter names
            fighter1, fighter2 = self._parse_fighter_names(matchup)

            return FightOdds(
                matchup=matchup,
                open_odds=open_odds,
                closing_range_start=closing_start,
                closing_range_end=closing_end,
                movement=movement,
                event=event,
                date=self._parse_date(date),
                fighter1=fighter1,
                fighter2=fighter2
            )

        except NoSuchElementException as e:
            logging.warning(f"Failed to extract odds from row: {str(e)}")
            return None

    def scrape_fighter_odds(self, fighter: str) -> List[FightOdds]:
        """Scrape all odds for a specific fighter."""
        odds_list = []

        try:
            # Search for fighter
            if not self.search_fighter(fighter):
                return odds_list

            # Try to find odds table
            try:
                odds_table = self.wait.until(
                    EC.presence_of_element_located((By.XPATH, "//table[@class='team-stats-table']"))
                )
            except TimeoutException:
                logging.warning(f"No odds table found for {fighter}")
                return odds_list

            # Process each row
            rows = odds_table.find_elements(By.XPATH, ".//tr")[1:]  # Skip header
            for row in rows:
                odds = self._extract_odds_from_row(row)
                if odds:
                    odds_list.append(odds)
                    # Add to existing matchups to prevent duplicates
                    self.existing_matchups.add((odds.event, odds.matchup))

            return odds_list

        except Exception as e:
            logging.error(f"Error scraping odds for {fighter}: {str(e)}")
            return odds_list

    def scrape_all(self) -> pd.DataFrame:
        """Scrape odds for all fighters."""
        all_odds = []

        for fighter in self.fighters:
            if fighter in self.processed_fighters:
                continue

            logging.info(f"Processing fighter: {fighter}")

            # Add random delay for rate limiting
            time.sleep(random.uniform(self.base_delay, self.max_delay))

            odds_list = self.scrape_fighter_odds(fighter)

            if odds_list:
                all_odds.extend(odds_list)
                self.processed_fighters.add(fighter)
                self.odds_cache[fighter] = odds_list
            else:
                self.failed_fighters.add(fighter)

        # Retry failed fighters if enabled
        if self.retry_failed and self.failed_fighters:
            self._retry_failed_fighters(all_odds)

        # Save processing results
        self._save_processing_results()

        # Convert to DataFrame
        return self._create_dataframe(all_odds)

    def _retry_failed_fighters(self, all_odds: List[FightOdds]) -> None:
        """Retry scraping for failed fighters with modified parameters."""
        logging.info(f"Retrying {len(self.failed_fighters)} failed fighters...")

        failed_copy = self.failed_fighters.copy()
        self.failed_fighters.clear()

        for fighter in failed_copy:
            logging.info(f"Retrying fighter: {fighter}")
            time.sleep(random.uniform(self.base_delay * 2, self.max_delay * 2))

            odds_list = self.scrape_fighter_odds(fighter)

            if odds_list:
                all_odds.extend(odds_list)
                self.processed_fighters.add(fighter)
                self.odds_cache[fighter] = odds_list
            else:
                self.failed_fighters.add(fighter)

    def _save_processing_results(self) -> None:
        """Save processing results to files."""
        # Save failed fighters
        if self.failed_fighters:
            with open('failed_fighters.txt', 'w') as f:
                for fighter in sorted(self.failed_fighters):
                    f.write(f"{fighter}\n")

        # Save processing statistics
        stats = {
            'total_fighters': len(self.fighters),
            'processed_fighters': len(self.processed_fighters),
            'failed_fighters': len(self.failed_fighters),
            'total_matchups': len(self.existing_matchups),
            'timestamp': datetime.now().isoformat()
        }

        with open('scraping_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

    def _create_dataframe(self, odds_list: List[FightOdds]) -> pd.DataFrame:
        """Convert odds list to DataFrame and clean data."""
        if not odds_list:
            return pd.DataFrame()

        df = pd.DataFrame([vars(odds) for odds in odds_list])

        # Clean and standardize data
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['movement'] = pd.to_numeric(df['movement'], errors='coerce')

        # Sort by date and reset index
        df = df.sort_values('date').reset_index(drop=True)

        return df

    def close(self) -> None:
        """Close the WebDriver and clean up resources."""
        if self.driver:
            self.driver.quit()


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Load fighter list
    try:
        combined_rounds_df = pd.read_csv("../data/combined_rounds.csv")
        fighters = combined_rounds_df["fighter"].unique().tolist()
        fighters = list(set(fighters))
    except Exception as e:
        logging.error(f"Error loading fighter data: {str(e)}")
        return

    # Load existing odds if available
    existing_odds_file = "../data/odds data/fight_odds.csv"
    existing_odds_df = None
    if os.path.exists(existing_odds_file):
        try:
            existing_odds_df = pd.read_csv(existing_odds_file)
        except Exception as e:
            logging.error(f"Error loading existing odds: {str(e)}")

    try:
        # Initialize and run scraper
        scraper = FightOddsScraper(
            fighters=fighters,
            existing_odds_df=existing_odds_df,
            headless=False,  # Set to True for headless mode
            retry_failed=True
        )

        try:
            # Scrape new odds
            new_odds_df = scraper.scrape_all()

            if new_odds_df.empty:
                logging.warning("No new odds data collected")
                return

            # Combine with existing data if available
            if existing_odds_df is not None and not existing_odds_df.empty:
                # Ensure date columns are compatible
                existing_odds_df['date'] = pd.to_datetime(existing_odds_df['date'])
                new_odds_df['date'] = pd.to_datetime(new_odds_df['date'])

                # Combine datasets
                combined_df = pd.concat([existing_odds_df, new_odds_df], ignore_index=True)

                # Remove duplicates based on event, matchup, and date
                combined_df = combined_df.drop_duplicates(
                    subset=['event', 'matchup', 'date'],
                    keep='last'
                )
            else:
                combined_df = new_odds_df

            # Save raw data
            combined_df.to_csv(existing_odds_file, index=False)
            logging.info(f"Saved raw odds data to {existing_odds_file}")

            # Clean and process the data
            cleaned_df = clean_fight_odds(combined_df)

            # Save cleaned data
            output_file = "../data/odds data/cleaned_fight_odds.csv"
            cleaned_df.to_csv(output_file, index=False)
            logging.info(f"Saved cleaned odds data to {output_file}")

            # Print summary statistics
            print("\nScraping Summary:")
            print(f"Total fighters processed: {len(scraper.processed_fighters)}")
            print(f"Failed fighters: {len(scraper.failed_fighters)}")
            print(f"Total matchups collected: {len(combined_df)}")
            print(f"Unique events: {combined_df['event'].nunique()}")

        except Exception as e:
            logging.error(f"Error during scraping process: {str(e)}")
            raise

        finally:
            scraper.close()


def clean_fight_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and process the fight odds data.

    Args:
        df: Raw fight odds DataFrame

    Returns:
        Cleaned DataFrame with processed odds data
    """
    # Create a copy to avoid modifying the original
    cleaned = df.copy()

    # Convert date to datetime if not already
    cleaned['date'] = pd.to_datetime(cleaned['date'], errors='coerce')

    # Normalize event names
    cleaned['event'] = cleaned['event'].str.strip()
    cleaned['event'] = cleaned['event'].fillna('')

    # Filter for UFC events
    cleaned = cleaned[
        cleaned['event'].str.contains('UFC', case=False, na=False)
    ]

    # Clean matchup names
    cleaned['matchup'] = cleaned['matchup'].str.strip()

    # Extract fighter names if not already present
    if 'fighter1' not in cleaned.columns or 'fighter2' not in cleaned.columns:
        fighter_splits = cleaned['matchup'].str.split(' vs ', expand=True)
        cleaned['fighter1'] = fighter_splits[0].str.strip()
        cleaned['fighter2'] = fighter_splits[1].str.strip()

    # Clean odds columns
    odds_columns = ['open_odds', 'closing_range_start', 'closing_range_end']
    for col in odds_columns:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].str.replace(r'[^\d+-]', '', regex=True)
            cleaned[col] = pd.to_numeric(cleaned[col], errors='coerce')

    # Calculate additional metrics
    cleaned['odds_movement'] = cleaned['closing_range_start'] - cleaned['open_odds']
    cleaned['odds_volatility'] = cleaned['closing_range_end'] - cleaned['closing_range_start']

    # Add timestamp for when the data was cleaned
    cleaned['processed_at'] = datetime.now()

    # Sort by date and reset index
    cleaned = cleaned.sort_values(['date', 'event', 'matchup']).reset_index(drop=True)

    return cleaned


if __name__ == "__main__":
    main()
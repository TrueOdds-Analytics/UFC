import os
import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium_stealth import stealth

# =====================================================
# 1. Load fighter names from the CSV file
# =====================================================
combined_rounds_df = pd.read_csv("../data/combined_rounds.csv")
fighters = list(set(combined_rounds_df["fighter"].unique().tolist()))
total_fighters = len(fighters)

# =====================================================
# 2. Check existing output CSV to skip already processed fighters
# =====================================================
output_csv = "../data/tapology_bouts_results.csv"
if os.path.exists(output_csv):
    existing_df = pd.read_csv(output_csv)
    processed_fighters = set(existing_df["fighter"].unique())
    print(f"Found existing output. Fighters already processed: {processed_fighters}")
else:
    processed_fighters = set()

# =====================================================
# 3. Set up Chrome with selenium-stealth and (optional) adblock extension
# =====================================================
chrome_options = Options()
chrome_options.add_argument("start-maximized")
# Uncomment the following line if you wish to run headless:
# chrome_options.add_argument("--headless")

chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option('useAutomationExtension', False)

# Optionally, add an adblock extension (Chrome .crx file) if desired.
# For example:
chrome_options.add_extension("ublock.crx")

# Initialize Chrome WebDriver (ensure chromedriver is in your PATH or specify executable_path)
driver = webdriver.Chrome(options=chrome_options)

# Apply selenium-stealth settings to help bypass bot detection
stealth(driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
       )

# =====================================================
# 4. Process each fighter
# =====================================================
for fighter in fighters:
    # Compute current status before processing
    scraped_count = len(processed_fighters)
    remaining = total_fighters - scraped_count
    percentage_scraped = (scraped_count / total_fighters) * 100
    print(f"\nStarting fighter '{fighter}'. {remaining} fighters left, {percentage_scraped:.2f}% scraped so far.")

    if fighter in processed_fighters:
        print(f"Skipping fighter '{fighter}' (already processed).")
        continue

    print(f"Processing fighter: {fighter}")
    new_bouts = []  # temporary list for this fighter's bout data

    # Navigate to the Tapology search page.
    driver.get("https://www.tapology.com/search")
    time.sleep(random.uniform(2, 4))

    # -------------------------------------
    # 4a. Click the "Bouts" subsection
    # -------------------------------------
    try:
        bouts_label = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "label[for='searchBouts']"))
        )
        bouts_label.click()
        time.sleep(random.uniform(1, 2))
    except Exception as e:
        print(f"Error clicking the Bouts label for fighter {fighter}: {e}")
        continue

    # -------------------------------------
    # 4b. Locate the search box, enter the fighter's name, and submit the search form.
    # -------------------------------------
    try:
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "siteSearch"))
        )
        search_box.clear()
        search_box.send_keys(fighter)
        time.sleep(random.uniform(1, 2))
        form = search_box.find_element(By.XPATH, "./ancestor::form")
        driver.execute_script("arguments[0].submit();", form)
        time.sleep(random.uniform(3, 5))
    except Exception as e:
        print(f"Error submitting the search form for fighter {fighter}: {e}")
        continue

    # -------------------------------------
    # 4c. Scrape bout data from the table for this fighter (with pagination)
    # -------------------------------------
    while True:
        try:
            table_body = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "tbody"))
            )
            time.sleep(random.uniform(1, 2))
            rows = table_body.find_elements(By.TAG_NAME, "tr")

            for row in rows:
                if row.find_elements(By.TAG_NAME, "th"):
                    continue  # Skip header rows

                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) < 7:
                    continue  # Not enough columns; skip this row.

                # Extract bout name and link from the first cell.
                try:
                    bout_anchor = cells[0].find_element(By.TAG_NAME, "a")
                    bout_name = bout_anchor.text.strip()
                    bout_link = bout_anchor.get_attribute("href").strip()
                except Exception:
                    bout_name = ""
                    bout_link = ""

                # Extract event (3rd cell), date (5th cell), and finish (7th cell).
                event = cells[2].text.strip() if len(cells) > 2 else ""
                date = cells[4].text.strip() if len(cells) > 4 else ""
                finish = cells[6].text.strip() if len(cells) > 6 else ""

                new_bouts.append({
                    "fighter": fighter,
                    "bout_name": bout_name,
                    "bout_link": bout_link,
                    "event": event,
                    "date": date,
                    "finish": finish
                })

            # -------------------------------------
            # 4d. Handle pagination: click "Next" if available.
            # -------------------------------------
            next_buttons = driver.find_elements(By.LINK_TEXT, "Next")
            if next_buttons:
                next_buttons[0].click()
                time.sleep(random.uniform(2, 4))
            else:
                break  # No "Next" button means we're done with this fighter.

        except TimeoutException:
            print(f"Timed out waiting for table data for fighter {fighter}; moving on.")
            break
        except Exception as e:
            print(f"Error processing bout table for fighter {fighter}: {e}")
            break

    # -------------------------------------
    # 4e. Append the new bouts for this fighter to the output CSV.
    # -------------------------------------
    if new_bouts:
        df_new = pd.DataFrame(new_bouts)
        if os.path.exists(output_csv):
            # Append new results without writing the header
            df_new.to_csv(output_csv, mode='a', header=False, index=False)
        else:
            # Write new file with header
            df_new.to_csv(output_csv, mode='w', header=True, index=False)
        print(f"Results for fighter '{fighter}' appended to {output_csv}")
        processed_fighters.add(fighter)  # update processed fighters
    else:
        print(f"No bout data found for fighter '{fighter}'.")

    # Additional pause between processing fighters.
    time.sleep(random.uniform(2, 4))

# =====================================================
# 5. Clean up
# =====================================================
driver.quit()
print("Scraping complete...")

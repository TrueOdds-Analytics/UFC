import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import TimeoutException

# =====================================================
# 1. Load fighter names from the CSV file
# =====================================================
combined_rounds_df = pd.read_csv("../data/combined_rounds.csv")
fighters = list(set(combined_rounds_df["fighter"].unique().tolist()))

# =====================================================
# 2. Set up Firefox with the adblock extension installed post-session start
# =====================================================
options = Options()
# Uncomment the following line if you wish to run headless:
# options.headless = True

driver = webdriver.Firefox(options=options)
driver.install_addon("uBlock0@raymondhill.net.xpi", temporary=True)

# This list will accumulate bout rows across all fighters.
all_bouts = []

# =====================================================
# 3. Process each fighter
# =====================================================
for fighter in fighters:
    print(f"Processing fighter: {fighter}")

    # Navigate to the Tapology search page.
    driver.get("https://www.tapology.com/search")
    time.sleep(2)  # Allow the page to fully load

    # -------------------------------------
    # 3a. Click the "Bouts" subsection
    # -------------------------------------
    try:
        bouts_label = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "label[for='searchBouts']"))
        )
        bouts_label.click()
    except Exception as e:
        print(f"Error clicking the Bouts label for fighter {fighter}: {e}")
        continue

    # -------------------------------------
    # 3b. Locate the search box, enter the fighter's name, and submit the search form.
    # -------------------------------------
    try:
        # Locate the search box (input with id "siteSearch")
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "siteSearch"))
        )
        search_box.clear()
        search_box.send_keys(fighter)
        time.sleep(1)  # Allow time for any dynamic suggestions to appear

        # Locate the parent form and submit it using JavaScript.
        form = search_box.find_element(By.XPATH, "./ancestor::form")
        driver.execute_script("arguments[0].submit();", form)
        time.sleep(2)  # Explicitly wait 2 seconds for the search results to load
    except Exception as e:
        print(f"Error submitting the search form for fighter {fighter}: {e}")
        continue

    # -------------------------------------
    # 3c. Scrape bout data from the table for this fighter (with pagination)
    # -------------------------------------
    while True:
        try:
            # Wait for the table body to load.
            table_body = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "tbody"))
            )
            # Get all rows within the table body.
            rows = table_body.find_elements(By.TAG_NAME, "tr")

            for row in rows:
                # Skip header rows (which contain <th> elements).
                if row.find_elements(By.TAG_NAME, "th"):
                    continue

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

                all_bouts.append({
                    "fighter": fighter,
                    "bout_name": bout_name,
                    "bout_link": bout_link,
                    "event": event,
                    "date": date,
                    "finish": finish
                })

            # -------------------------------------
            # 3d. Handle pagination: click "Next" if available.
            # -------------------------------------
            next_buttons = driver.find_elements(By.LINK_TEXT, "Next")
            if next_buttons:
                next_buttons[0].click()
                time.sleep(2)  # Wait 2 seconds for the next page to load
            else:
                break  # No "Next" button means we're done with this fighter.

        except TimeoutException:
            print(f"Timed out waiting for table data for fighter {fighter}; moving on.")
            break
        except Exception as e:
            print(f"Error processing bout table for fighter {fighter}: {e}")
            break

    # -------------------------------------------------
    # 3e. Save the current accumulated results to CSV.
    # -------------------------------------------------
    df = pd.DataFrame(all_bouts)
    df.to_csv("tapology_bouts_results.csv", index=False)
    print(f"Results for fighter '{fighter}' saved to tapology_bouts_results.csv")

# =====================================================
# 4. Clean up
# =====================================================
driver.quit()
print("Scraping complete.")

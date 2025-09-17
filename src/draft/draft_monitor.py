import os
import time
import multiprocessing
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# --- Configuration ---
# IMPORTANT: Replace this with the "Profile Path" you found by visiting chrome://version
# Ensure you use the full path.
CHROME_PROFILE_PATH = "/Users/josepherwin/Library/Application Support/Google/Chrome/Default"


class DraftMonitor:
    """
    A class to monitor a live ESPN fantasy football draft using Selenium.
    This class uses an existing Chrome profile to bypass automated login,
    with a fallback for manual login.
    """

    def __init__(self, league_id, season_id, team_id, member_id, shared_draft_log, opimizer_trigger, shutdown_event):
        self.draft_url = f"https://fantasy.espn.com/football/draft?leagueId={league_id}&seasonId={season_id}&teamId={team_id}&memberId={{{member_id}}}"
        self.shared_draft_log = shared_draft_log
        self.opimizer_trigger = opimizer_trigger
        self.shutdown_event = shutdown_event
        self.driver = None
        self.processed_picks = set()

    def _navigate_to_draft(self):
        """Navigates directly to the draft room URL and checks for success."""
        print(f"[Monitor] Attempting to navigate to draft room: {self.draft_url}")
        self.driver.get(self.draft_url)
        try:
            # Wait for a key element of the draft room to load to confirm we're in
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "tabs__list"))
            )
            print("[Monitor] Successfully entered draft room.")
            return True
        except TimeoutException:
            # This will be caught by the calling function to trigger the manual login flow
            return False

    def _handle_manual_login(self):
        """
        Pauses the script and waits for the user to manually log in and
        navigate to the correct draft page.
        """
        print("\n" + "=" * 60)
        print("ACTION REQUIRED: Automatic login failed or session expired.")
        print("Please log in to ESPN in the automated Chrome window.")
        print("The script will automatically continue once you have successfully")
        print("navigated to the draft room.")
        print("=" * 60 + "\n")

        while not self.shutdown_event.is_set():
            # Periodically check if the user has reached the draft room
            if self._navigate_to_draft():
                # Success! The user has logged in and navigated correctly.
                return
            # Wait before trying again
            print("[Monitor] Waiting for manual login... will check again in 10 seconds.")
            time.sleep(10)

    def _check_if_on_clock(self):
        """Checks if the 'You are on the clock!' message is displayed."""
        try:
            # Use a more specific XPath to avoid ambiguity
            on_clock_element = self.driver.find_element(By.XPATH, "//div[contains(@class, 'pickArea')]//h3")
            return "You are on the clock!" in on_clock_element.text
        except Exception as e:
            print(f"[Monitor] An error occurred while checking for on-clock status: {e}")
            return False

    def _scrape_draft_log(self, is_bulk_scrape=False):
        try:
            tab_buttons = self.driver.find_elements(By.CLASS_NAME, "tabs__link")
            for button in tab_buttons:
                if button.text == "Pick History":
                    parent_li = button.find_element(By.XPATH, "..")
                    if 'tabs__list__item--active' not in parent_li.get_attribute('class'):
                        button.click()
                        time.sleep(1)
                    break

            rows = self.driver.find_elements(By.CSS_SELECTOR,
                                             "div.pick-history-tables div.public_fixedDataTableRow_main")

            new_picks_found = False
            for row in rows:
                cells = row.find_elements(By.CSS_SELECTOR, "div.public_fixedDataTableCell_main")
                if len(cells) < 3: continue

                pick_num_text = cells[0].text.strip()
                try:
                    player_name = cells[1].find_element(By.CLASS_NAME, "playerinfo__playername").text.strip()
                    drafting_team = cells[2].text.strip()
                except NoSuchElementException:
                    continue

                if pick_num_text and player_name and pick_num_text not in self.processed_picks:
                    new_picks_found = True
                    pick_info = {
                        'type': 'new_pick',
                        'pick_num': pick_num_text,
                        # 'player_id': player_name,  # Placeholder
                        'player_name': player_name,
                        'team_name': drafting_team
                    }
                    self.shared_draft_log.append(pick_info)
                    self.processed_picks.add(pick_num_text)

            if new_picks_found and not is_bulk_scrape:
                print(f"[Monitor] New picks added to shared log.")

        except Exception as e:
            print(f"[Monitor] An error occurred during scraping: {e}")

    def run(self):
        options = webdriver.ChromeOptions()
        options.add_argument(f"user-data-dir={CHROME_PROFILE_PATH}")
        options.add_argument("profile-directory=Default")
        self.driver = webdriver.Chrome(options=options)

        try:
            if not self._navigate_to_draft():
                self._handle_manual_login()

            time.sleep(5)

            # --- NEW: Bulk scrape on startup ---
            print("[Monitor] Performing initial bulk scrape of draft log...")
            self._scrape_draft_log(is_bulk_scrape=True)

            # --- Continuous monitoring loop ---
            num_on_the_clock = 0
            while not self.shutdown_event.is_set():
                on_the_clock = self._check_if_on_clock()
                self.opimizer_trigger.set(1 if on_the_clock else 0)

                if on_the_clock:
                    num_on_the_clock += 1
                else:
                    num_on_the_clock = 0

                if num_on_the_clock <= 1:
                    self._scrape_draft_log()
                    time.sleep(3)  # Check for new picks every 5 seconds
                else:
                    time.sleep(0.5)

        except Exception as e:
            print(f"[Monitor] A critical error occurred in the main run loop: {e}")
        finally:
            print("[Monitor] Process shutting down.")
            if self.driver:
                self.driver.quit()


if __name__ == '__main__':
    TEST_LEAGUE_ID = 2110029406
    TEST_SEASON_ID = 2025
    TEST_TEAM_ID = 8

    q = multiprocessing.Queue()
    e = multiprocessing.Event()

    print("--- Running DraftMonitor in standalone test mode ---")
    print(
        "IMPORTANT: Make sure you are logged into ESPN.com in your regular Chrome browser for the profile loading to work.")

    monitor = DraftMonitor(
        league_id=TEST_LEAGUE_ID,
        season_id=TEST_SEASON_ID,
        team_id=TEST_TEAM_ID,
        member_id=os.environ.get('SWID'),
        draft_pick_queue=q,
        shutdown_event=e
    )

    try:
        monitor.run()
    except KeyboardInterrupt:
        print("\nTest stopped by user.")
        e.set()

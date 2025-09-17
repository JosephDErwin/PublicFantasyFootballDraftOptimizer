import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.paths import root_dir


def get_adp(name, season_adp):
    data = season_adp[['Player Name', 'AVG']]
    subset = data[season_adp['Player Name'] == name]

    if subset.shape[0] == 1:
        return subset['AVG'].values[0]

    for part in name.split(' '):
        subset = data[data['Player Name'].str.contains(part, na=False)]

        if subset.shape[0] == 1:
            return subset['AVG'].values[0]
        elif subset.shape[0] < 1:
            return None
    return None


def scrape_fantasypros_adp(url: str) -> pd.DataFrame:
    """
    Scrapes the overall ADP (Average Draft Position) data from a given FantasyPros URL.

    Args:
        url (str): The URL of the FantasyPros ADP page to scrape.

    Returns:
        pd.DataFrame: A DataFrame containing the ADP data with columns for Rank,
                      Player, Team, Position, Bye Week, and ADP. Returns an empty
                      DataFrame if the request fails or the table is not found.
    """
    print(f"Attempting to scrape data from: {url}")

    try:
        # Set a user-agent to mimic a browser, which can help avoid being blocked.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to retrieve the webpage. {e}")
        return pd.DataFrame()

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the main data table. FantasyPros often uses an id='data'.
    table = soup.find('table', id='data')

    if not table:
        print("Error: Could not find the data table on the page.")
        return pd.DataFrame()

    # Extract the table headers
    headers = [th.text.strip() for th in table.find_all('th')]

    # Extract the table rows
    rows = []
    for tr in table.find('tbody').find_all('tr'):
        cells = tr.find_all('td')
        if len(cells) > 0:
            rows.append([cell.text.strip() for cell in cells])

    # Create a pandas DataFrame
    if not rows:
        print("Warning: No data rows found in the table.")
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=headers)
    df['Player Name'] = df['Player Team (Bye)'].str.split(' ').map(lambda x: ' '.join(x[:-2]))
    df['AVG'] = df['AVG'].astype(float)

    return df


def download_adp_data(years, update=True):
    data_dir = root_dir / Path('data/historical_adp')
    data_dir.mkdir(exist_ok=True)

    adp_data = {}
    for year in years:
        if year == datetime.now().year and update:
            url = 'https://www.fantasypros.com/nfl/adp/half-point-ppr-overall.php'
            data = scrape_fantasypros_adp(url)
        elif not os.path.exists(data_dir / Path(f'adp_{year}.csv')):
            url = f"https://web.archive.org/web/{year}0901231401mp_/https://www.fantasypros.com/nfl/adp/half-point-ppr-overall.php"
            data = scrape_fantasypros_adp(url)
            data.to_csv(data_dir / Path(f'adp_{year}.csv'), index=False)
        else:
            data = pd.read_csv(data_dir / Path(f'adp_{year}.csv'))

        adp_data[year] = data

    return adp_data


if __name__ == '__main__':
    # The specific URL provided by the user
    download_adp_data(range(2022, 2025))


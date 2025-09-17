import json
from pathlib import Path

import cfbd
import joblib

from src.paths import root_dir

TOKEN = os.environ.get("NFL_TOKEN")


def get_api_profile():

    configuration = cfbd.Configuration(
        access_token=TOKEN
    )

    return configuration


def get_raw_nfl_draft_data(years):
    configuration = get_api_profile()

    output_dir = root_dir / Path('data/nfl_draft_data')
    output_dir.mkdir(exist_ok=True)

    # Enter a context with an instance of the API client
    yearly_drafts = {}
    with cfbd.ApiClient(configuration) as api_client:
        # Create an instance of the API class
        api_instance = cfbd.DraftApi(api_client)

        for year in years[::-1]:
            output_file = output_dir / Path(f'draft_{year}.sav')

            if output_file.exists():
                yearly_drafts[year] = joblib.load(open(output_file, 'rb'))
            else:
                try:
                    api_response = api_instance.get_draft_picks(year=year)
                    yearly_drafts[year] = api_response
                    joblib.dump(api_response, open(output_file, 'wb'))
                except Exception as e:
                    print("Exception when calling DraftApi->get_draft_picks: %s\n" % e)

    return yearly_drafts


def get_nfl_draft_data(years):
    api_response = get_raw_nfl_draft_data(years)

    all_data = sum(api_response.values(), [])

    player_data = {}
    for pick in all_data:
        player_data[pick.college_athlete_id] = pick

    return player_data


if __name__ == '__main__':
    data = get_nfl_draft_data()
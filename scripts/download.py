import os
import time
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

MCP = 'https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master'
ATP = 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master'
WTA = 'https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master'

URLS = {
    'charting-m-matches.csv': f'{MCP}/charting-m-matches.csv',
    'charting-m-points-2020s.csv': f'{MCP}/charting-m-points-2020s.csv',
    'charting-w-matches.csv': f'{MCP}/charting-w-matches.csv',
    'charting-w-points-2020s.csv': f'{MCP}/charting-w-points-2020s.csv',
    'atp_matches_2023.csv': f'{ATP}/atp_matches_2023.csv',
    'wta_matches_2023.csv': f'{WTA}/wta_matches_2023.csv',
}


def download(filename, url):
    dest = os.path.join(RAW_DIR, filename)
    print(f'  Downloading {filename}...')
    for attempt in range(3):
        try:
            response = requests.get(url, verify=False)
            response.raise_for_status()
            with open(dest, 'wb') as f:
                f.write(response.content)
            print(f'  -> Saved {filename}')
            return
        except Exception as e:
            print(f'  Attempt {attempt + 1} failed: {e}')
            time.sleep(2)
    print(f'  ERROR: Failed after 3 attempts.')


def main():
    print('=== download.py ===')
    print('Raw data is sacred - saving unmodified files to data/raw/')
    print()
    for filename, url in URLS.items():
        download(filename, url)
        time.sleep(1)
    print()
    print('=== Done ===')


if __name__ == '__main__':
    main()
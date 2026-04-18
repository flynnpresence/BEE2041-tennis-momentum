"""
download.py
Downloads raw data from Jeff Sackmann's GitHub repositories.
Saves files to data/raw/ without any modification.

Raw data is sacred — no filtering or cleaning occurs here.
All filtering and cleaning happens in clean.py.
"""

import os
import time

import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

# ── Source URLs ───────────────────────────────────────────────────────────────
MCP = 'https://raw.githubusercontent.com/JeffSackmann/tennis_MatchChartingProject/master'
ATP = 'https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master'
WTA = 'https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master'

HEADERS = {
    'User-Agent': 'BEE2041_Student_Project (Data Science in Economics)'
}

URLS = {
    'charting-m-matches.csv':      f'{MCP}/charting-m-matches.csv',
    'charting-m-points-2020s.csv': f'{MCP}/charting-m-points-2020s.csv',
    'charting-w-matches.csv':      f'{MCP}/charting-w-matches.csv',
    'charting-w-points-2020s.csv': f'{MCP}/charting-w-points-2020s.csv',
    'atp_matches_2023.csv':        f'{ATP}/atp_matches_2023.csv',
    'wta_matches_2023.csv':        f'{WTA}/wta_matches_2023.csv',
}


# ── Download helper ───────────────────────────────────────────────────────────
def download(filename: str, url: str) -> None:
    """
    Download a single CSV from url and save to data/raw/ unmodified.

    Uses requests.get with verify=False to bypass macOS SSL issues.
    Retries up to 3 times with a 2-second delay between attempts.
    Writes raw binary content directly — pandas is bypassed entirely
    to guarantee files are saved bit-for-bit identically to the source.

    Args:
        filename (str): Name to save the file as in data/raw/.
        url      (str): Remote CSV URL to fetch.
    """
    dest = os.path.join(RAW_DIR, filename)
    print(f'  Downloading {filename}...')

    for attempt in range(3):
        try:
            # verify=False bypasses macOS SSL certificate verification issues with GitHub
            # This is a known local environment workaround, not a security endorsement
            response = requests.get(url, headers=HEADERS, verify=False)
            response.raise_for_status()
            with open(dest, 'wb') as f:
                f.write(response.content)
            print(f'  -> Saved {filename}')
            return
        except Exception as e:
            print(f'  Attempt {attempt + 1} failed: {e}')
            time.sleep(2)

    print(f'  ERROR: Failed after 3 attempts.')


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """
    Download all six raw data files to data/raw/ without modification.

    Files downloaded:
        - charting-m-matches.csv      : ATP match metadata
        - charting-m-points-2020s.csv : ATP point-by-point data (2020s)
        - charting-w-matches.csv      : WTA match metadata
        - charting-w-points-2020s.csv : WTA point-by-point data (2020s)
        - atp_matches_2023.csv        : ATP 2023 results for rankings
        - wta_matches_2023.csv        : WTA 2023 results for rankings
    """
    print('=== download.py ===')
    print('Raw data is sacred - saving unmodified files to data/raw/')
    print()

    for filename, url in URLS.items():
        download(filename, url)
        time.sleep(1)

    print()
    print('=== Done ===')
    print(f'Location: {RAW_DIR}')


if __name__ == '__main__':
    main()
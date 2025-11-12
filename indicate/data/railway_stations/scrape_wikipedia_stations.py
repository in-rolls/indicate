#!/usr/bin/env python3
"""
Indian Railway Station Names from Wikipedia

Scrapes railway station names from Wikipedia's list and attempts to fetch
multilingual names from corresponding Wikipedia articles in regional languages.

This provides real, validated station names from public sources.

Usage:
    python scrape_wikipedia_stations.py
    python scrape_wikipedia_stations.py --limit 50 --output stations.tsv
"""

import argparse
import csv
import re
import time
from pathlib import Path
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Indicate-Research-Bot/1.0 (https://github.com/in-rolls/indicate; research purposes)'
}

# Language Wikipedia prefixes
LANG_WIKIS = {
    'hi': 'Hindi',
    'bn': 'Bengali',
    'ta': 'Tamil',
    'te': 'Telugu',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'pa': 'Punjabi',
    'or': 'Odia',
    'ur': 'Urdu'
}


def scrape_station_list() -> List[Dict]:
    """Scrape list of stations from English Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_railway_stations_in_India'

    print(f"Fetching station list from Wikipedia...")
    response = requests.get(url, headers=HEADERS, timeout=30)

    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    tables = soup.find_all('table', class_='wikitable')

    stations = []

    for table in tables:
        rows = table.find_all('tr')[1:]  # Skip header

        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 2:
                continue

            # Extract station name and code
            name_cell = cells[0]
            code_cell = cells[1]

            # Get the link to the station's article
            link = name_cell.find('a')
            if link and link.get('href'):
                station_name = name_cell.get_text(strip=True)
                station_code = code_cell.get_text(strip=True)
                wiki_path = link.get('href')

                stations.append({
                    'name_en': station_name,
                    'code': station_code,
                    'wiki_path': wiki_path
                })

    print(f"✓ Found {len(stations)} stations")
    return stations


def get_interwiki_names(wiki_path: str) -> Dict[str, str]:
    """Get multilingual names from Wikipedia interwiki links."""
    # Remove /wiki/ prefix
    if wiki_path.startswith('/wiki/'):
        page_title = wiki_path[6:]
    else:
        page_title = wiki_path

    # Use Wikipedia API to get interwiki links
    api_url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'titles': page_title,
        'prop': 'langlinks',
        'lllimit': 'max',
        'format': 'json'
    }

    try:
        response = requests.get(api_url, params=params, headers=HEADERS, timeout=10)
        data = response.json()

        pages = data.get('query', {}).get('pages', {})
        if not pages:
            return {}

        # Get first (and only) page
        page = list(pages.values())[0]
        langlinks = page.get('langlinks', [])

        # Extract names in target languages
        names = {}
        for link in langlinks:
            lang = link.get('lang')
            title = link.get('*')
            if lang in LANG_WIKIS and title:
                names[lang] = title

        return names
    except Exception as e:
        return {}


def extract_transliteration_pairs(stations: List[Dict], limit: Optional[int] = None) -> List[Dict]:
    """Extract transliteration pairs with multilingual names."""
    pairs = []
    count = 0

    for station in stations:
        if limit and count >= limit:
            break

        count += 1
        print(f"Processing {count}/{min(limit or len(stations), len(stations))}: {station['name_en']}...", end='\r')

        # Get multilingual names
        multilingual_names = get_interwiki_names(station['wiki_path'])

        if not multilingual_names:
            continue

        # Create pairs for each language
        for lang, native_name in multilingual_names.items():
            pairs.append({
                'native_script': native_name,
                'romanization': station['name_en'],
                'language': lang,
                'station_code': station['code'],
                'source': 'wikipedia'
            })

        # Rate limiting
        time.sleep(0.2)

    print()  # New line after progress
    return pairs


def save_to_tsv(pairs: List[Dict], output_file: Path):
    """Save pairs to TSV file."""
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['native_script', 'romanization', 'language', 'station_code', 'source'],
            delimiter='\t'
        )
        writer.writeheader()
        writer.writerows(pairs)


def print_statistics(pairs: List[Dict]):
    """Print statistics."""
    lang_counts = {}
    for pair in pairs:
        lang = pair['language']
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    print(f"\n{'='*60}")
    print(f"Total pairs: {len(pairs)}")
    print(f"\nBy language:")
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Scrape railway station names from Wikipedia')
    parser.add_argument('--limit', type=int, default=100, help='Limit number of stations to process')
    parser.add_argument('--output', type=str, default='railway_stations.tsv', help='Output file')
    args = parser.parse_args()

    print("="*60)
    print("Wikipedia Railway Station Scraper")
    print("="*60)

    # Scrape station list
    stations = scrape_station_list()

    if not stations:
        print("Error: No stations found")
        return

    # Extract transliteration pairs
    print(f"\nFetching multilingual names (limit: {args.limit})...")
    pairs = extract_transliteration_pairs(stations, args.limit)

    if not pairs:
        print("Warning: No transliteration pairs found")
        return

    # Save
    output_file = Path(args.output)
    save_to_tsv(pairs, output_file)
    print(f"\n✓ Saved {len(pairs)} pairs to {output_file}")

    # Stats
    print_statistics(pairs)

    # Sample
    print(f"\nSample pairs:")
    for pair in pairs[:10]:
        print(f"  {pair['native_script']} → {pair['romanization']} ({pair['language']})")


if __name__ == '__main__':
    main()

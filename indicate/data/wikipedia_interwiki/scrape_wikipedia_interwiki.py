#!/usr/bin/env python3
"""
Wikipedia Interwiki Link Scraper - Generalized

Extracts transliteration pairs from Wikipedia by mining interwiki links.
When the same article exists in multiple language editions, the titles
provide natural transliteration pairs for proper nouns.

This is a generalized version that works on ANY Wikipedia category,
not just railway stations.

Usage:
    # People (actors, politicians, etc.)
    python scrape_wikipedia_interwiki.py --category "Indian_actors" --limit 50

    # Places
    python scrape_wikipedia_interwiki.py --category "Cities_in_India" --limit 50

    # Films
    python scrape_wikipedia_interwiki.py --category "Indian_films" --limit 50

    # Random articles (any proper nouns)
    python scrape_wikipedia_interwiki.py --random --limit 100
"""

import argparse
import csv
import time
from pathlib import Path
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'Indicate-Research-Bot/1.0 (https://github.com/in-rolls/indicate; research purposes)'
}

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
    'ur': 'Urdu',
    'as': 'Assamese',
    'ne': 'Nepali',
    'si': 'Sinhala',
    'sa': 'Sanskrit'
}


def get_category_members(category: str, limit: int = 500) -> List[Dict]:
    """
    Get articles from a Wikipedia category.

    Args:
        category: Category name (without "Category:" prefix)
        limit: Maximum articles to fetch

    Returns:
        List of article dictionaries with title and page_id
    """
    api_url = 'https://en.wikipedia.org/w/api.php'

    articles = []
    continue_token = None

    print(f"Fetching articles from category: {category}")

    while len(articles) < limit:
        params = {
            'action': 'query',
            'list': 'categorymembers',
            'cmtitle': f'Category:{category}',
            'cmlimit': min(500, limit - len(articles)),
            'cmtype': 'page',  # Only pages, not subcategories
            'format': 'json'
        }

        if continue_token:
            params['cmcontinue'] = continue_token

        try:
            response = requests.get(api_url, params=params, headers=HEADERS, timeout=30)
            data = response.json()

            members = data.get('query', {}).get('categorymembers', [])

            for member in members:
                articles.append({
                    'title': member['title'],
                    'page_id': member['pageid']
                })

            # Check for continuation
            if 'continue' in data:
                continue_token = data['continue'].get('cmcontinue')
                time.sleep(0.5)  # Rate limiting
            else:
                break

        except Exception as e:
            print(f"Error fetching category members: {e}")
            break

    print(f"✓ Found {len(articles)} articles")
    return articles[:limit]


def get_random_articles(limit: int = 100) -> List[Dict]:
    """
    Get random articles from Wikipedia.

    Args:
        limit: Number of random articles

    Returns:
        List of article dictionaries
    """
    api_url = 'https://en.wikipedia.org/w/api.php'

    articles = []
    batch_size = 500  # Max allowed by API

    print(f"Fetching {limit} random articles...")

    for i in range(0, limit, batch_size):
        current_batch = min(batch_size, limit - i)

        params = {
            'action': 'query',
            'list': 'random',
            'rnnamespace': 0,  # Main namespace only
            'rnlimit': current_batch,
            'format': 'json'
        }

        try:
            response = requests.get(api_url, params=params, headers=HEADERS, timeout=30)
            data = response.json()

            randoms = data.get('query', {}).get('random', [])

            for item in randoms:
                articles.append({
                    'title': item['title'],
                    'page_id': item['id']
                })

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"Error fetching random articles: {e}")
            break

    print(f"✓ Found {len(articles)} articles")
    return articles


def get_interwiki_links(page_title: str) -> Dict[str, str]:
    """
    Get interwiki links for a Wikipedia page.

    Args:
        page_title: Wikipedia page title

    Returns:
        Dictionary mapping language codes to article titles
    """
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


def extract_transliteration_pairs(
    articles: List[Dict],
    source_type: str = 'wikipedia'
) -> List[Dict]:
    """
    Extract transliteration pairs from articles via interwiki links.

    Args:
        articles: List of article dictionaries
        source_type: Source identifier for output

    Returns:
        List of transliteration pair dictionaries
    """
    pairs = []
    total = len(articles)

    for idx, article in enumerate(articles, 1):
        print(f"Processing {idx}/{total}: {article['title']}...", end='\r')

        # Get multilingual names
        multilingual_names = get_interwiki_links(article['title'])

        if not multilingual_names:
            continue

        # Create pairs for each language
        for lang, native_title in multilingual_names.items():
            pairs.append({
                'native_script': native_title,
                'romanization': article['title'],
                'language': lang,
                'source': source_type
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
            fieldnames=['native_script', 'romanization', 'language', 'source'],
            delimiter='\t'
        )
        writer.writeheader()
        writer.writerows(pairs)


def print_statistics(pairs: List[Dict]):
    """Print statistics about collected data."""
    lang_counts = {}
    for pair in pairs:
        lang = pair['language']
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    print(f"\n{'='*60}")
    print(f"Total pairs: {len(pairs)}")
    print(f"\nBy language:")
    for lang, count in sorted(lang_counts.items()):
        lang_name = LANG_WIKIS.get(lang, lang)
        print(f"  {lang} ({lang_name}): {count}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract transliteration pairs from Wikipedia interwiki links',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Categories to try (proper nouns):

  People:
    Indian_actors
    Indian_politicians
    Indian_sportspeople
    Indian_writers
    Indian_singers
    Indian_film_directors

  Places:
    Cities_in_India
    Tourist_attractions_in_India
    Monuments_and_memorials_in_India
    Hindu_temples

  Culture:
    Indian_films
    Bollywood_films
    Tamil-language_films
    Telugu-language_films
    Indian_novels

  Organizations:
    Companies_of_India
    Universities_and_colleges_in_India
    Political_parties_in_India

Examples:
  # Get 50 Indian actors
  %(prog)s --category "Indian_actors" --limit 50

  # Get 100 Indian cities
  %(prog)s --category "Cities_in_India" --limit 100

  # Get 50 Bollywood films
  %(prog)s --category "Bollywood_films" --limit 50

  # Get 100 random articles
  %(prog)s --random --limit 100
        """
    )

    parser.add_argument(
        '--category',
        type=str,
        help='Wikipedia category name (without "Category:" prefix)'
    )

    parser.add_argument(
        '--random',
        action='store_true',
        help='Get random articles instead of from a category'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Maximum articles to process (default: 100)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='wikipedia_interwiki.tsv',
        help='Output TSV file (default: wikipedia_interwiki.tsv)'
    )

    args = parser.parse_args()

    if not args.category and not args.random:
        parser.error('Must specify either --category or --random')

    print("="*60)
    print("Wikipedia Interwiki Link Scraper")
    print("="*60)

    # Get articles
    if args.random:
        articles = get_random_articles(args.limit)
        source_type = 'wikipedia_random'
    else:
        articles = get_category_members(args.category, args.limit)
        source_type = f'wikipedia_{args.category}'

    if not articles:
        print("Error: No articles found")
        return

    # Extract transliteration pairs
    print(f"\nExtracting transliteration pairs...")
    pairs = extract_transliteration_pairs(articles, source_type)

    if not pairs:
        print("Warning: No transliteration pairs found")
        print("Try a different category with more proper nouns")
        return

    # Save
    output_file = Path(args.output)
    save_to_tsv(pairs, output_file)
    print(f"\n✓ Saved {len(pairs)} pairs to {output_file}")

    # Statistics
    print_statistics(pairs)

    # Sample
    print(f"\nSample pairs:")
    for pair in pairs[:10]:
        print(f"  {pair['native_script']} → {pair['romanization']} ({pair['language']})")


if __name__ == '__main__':
    main()

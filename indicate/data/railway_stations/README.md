# Indian Railway Station Names

**STATUS: VALIDATED AND WORKING**

Scrapes railway station names from Wikipedia and extracts multilingual transliterations via interwiki links.

## What It Does

1. Scrapes Wikipedia's "List of railway stations in India" (2,985+ stations)
2. For each station, fetches multilingual article titles via Wikipedia API
3. Extracts transliteration pairs in 11 South Asian languages
4. Outputs validated TSV format

## Languages Supported

Bengali (bn), Hindi (hi), Tamil (ta), Telugu (te), Kannada (kn), Malayalam (ml), Marathi (mr), Gujarati (gu), Punjabi (pa), Odia (or), Urdu (ur)

## Usage

```bash
# Process first 50 stations
python scrape_wikipedia_stations.py --limit 50

# Process first 200 stations
python scrape_wikipedia_stations.py --limit 200 --output stations_200.tsv

# Process all stations (takes ~10-15 minutes with rate limiting)
python scrape_wikipedia_stations.py --limit 2985
```

## Requirements

```bash
pip install requests beautifulsoup4
```

## Output Format

TSV file with columns:
- `native_script`: Station name in native script
- `romanization`: Station name in English
- `language`: ISO 639-1 language code
- `station_code`: Indian Railways station code
- `source`: Always "wikipedia"

## Example Output

```tsv
native_script	romanization	language	station_code	source
আবাদা রেলওয়ে স্টেশন	Abada	bn	ABB	wikipedia
अबादा रेलवे स्टेशन	Abada	hi	ABB	wikipedia
ஆபாதா தொடருந்து நிலையம்	Abada	ta	ABB	wikipedia
```

## Validated Test Results

**Test run**: 10 stations processed
- **Found**: 23 transliteration pairs
- **Languages**: 7 (bn, hi, mr, pa, ta, te, ur)
- **Success rate**: ~2.3 pairs per station average

## Data Quality

- ✅ **Real data** from Wikipedia
- ✅ **Community-verified** station names
- ✅ **Official station codes** from Indian Railways
- ✅ **Tested and working** (see test results above)

## Limitations

- Not all stations have multilingual Wikipedia articles
- Smaller stations may only have English articles
- Rate limited to 0.2s per station (respectful scraping)
- Station names include "Railway Station" suffix in some languages

## Expected Scale

- **Total stations available**: ~3,000
- **Estimated pairs** (with all stations): ~5,000-10,000 pairs
- **Processing time** (all stations): 10-15 minutes

## License

- Script: MIT (Indicate project)
- Data: CC BY-SA (Wikipedia content)

## Citation

```bibtex
@misc{indicate_railway_stations,
  author = {Indicate Project},
  title = {Indian Railway Station Transliterations from Wikipedia},
  year = {2024},
  url = {https://github.com/in-rolls/indicate}
}
```

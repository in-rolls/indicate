import json
import datetime
import requests
from espncricinfo.match import Match

hindi_matches = []
players = []

with open("all-matches.json", "r") as matches_file:
    matches = json.load(matches_file)
    possible_matches = [x for x in matches if x > 1230000]
    print(len(possible_matches))
    for match_id in possible_matches:
        try:
            match = Match(match_id)
        except Exception:
            continue
        y, m, d = [int(x) for x in match.date.split("-")]
        if datetime.datetime(y, m, d) > datetime.datetime(2021, 4, 8):
            print(match_id)
            r = requests.get(match.match_url)
            hindi_url = r.url.replace("/series/", "/hindi/series/").replace("live-cricket-score", "full-scorecard")
            r = requests.get(hindi_url)
            if r.status_code == 404:
                hindi_url = None
            if hindi_url:
                hindi_matches.append(match_id)

with open("hindi_matches.json", "w") as outfile:
    json.dump(hindi_matches, outfile)

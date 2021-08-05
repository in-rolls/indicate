import json
import datetime
import requests
from bs4 import BeautifulSoup
from espncricinfo.match import Match

players = []

with open('hindi_matches.json', 'r') as matches_file:
    hindi_matches = json.load(matches_file)
    for match in hindi_matches:
        print(match)
        m = Match(match)
#        for roster in m.rosters['teamPlayers']:
#            for player in roster['players']:
#                if not any(p['id'] == player['player']['id'] for p in players):
#                    players.append({'id': player['player']['id'], 'name': player['player']['name'], 'gender': player['player']['gender'], 'date_of_birth': player['player']['dateOfBirth'], 'country_id': player['player']['countryTeamId'], 'slug': player['player']['slug']})

        r = requests.get(m.match_url)
        hindi_url = r.url.replace('/series/','/hindi/series/').replace('live-cricket-score','full-scorecard')
        hindi_content = requests.get(hindi_url).text
        soup = BeautifulSoup(hindi_content, 'html.parser')
        text = soup.find('script', id='__NEXT_DATA__').string
        hindi_json = json.loads(text)
        if 'data' in hindi_json['props']['pageProps']:
            for team_players in hindi_json['props']['pageProps']['data']['pageData']['content']['matchPlayers']['teamPlayers']:
                for player in team_players['players']:
                    players.append({'id': player['player']['id'], 'hindi_name': player['player']['name'], 'hindi_long_name': player['player']['longName'], 'english_name': player['player']['indexName'], 'gender': player['player']['gender'], 'date_of_birth': player['player']['dateOfBirth'], 'country_id': player['player']['countryTeamId'], 'slug': player['player']['slug']})

with open('players_with_hindi_names.json', 'w') as outfile:
    json.dump(players, outfile)

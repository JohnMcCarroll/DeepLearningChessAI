import requests
import time

### This script interfaces with Chess.com's API to retrieve PGN files of all Gramdmaster games that have occured on the site

HEADERS = {
    'User-Agent': 'DeepLearningChessAI/1.0 (email@example.com)'
}

# receive request and parse JSON directly to get the list of GMs
response = requests.get("https://api.chess.com/pub/titled/GM", headers=HEADERS)
if response.status_code == 200:
    GMs = response.json().get('players', [])
else:
    print(f"Failed to fetch GMs: {response.status_code}")
    GMs = []

# iterate through GMs and retreive lists of their archive endpoints
for username in GMs:
    
    archives_resp = requests.get("https://api.chess.com/pub/player/" + username + "/games/archives", headers=HEADERS)
    print('archives status for ' + username + ': ' + str(archives_resp.status_code))
    
    if archives_resp.status_code == 200:
        archives = archives_resp.json().get('archives', [])
        
        if archives:
            for archive in archives:
                print('archive: ' + archive)
                games = requests.get(archive + "/pgn", headers=HEADERS)
                if games.status_code == 200:
                    with open("games.pgn", 'ab') as f:
                        f.write(games.content)
        else: 
            time.sleep(15)
            print('sleepytime')
    else:
        print("Skipping " + username + " due to error.")


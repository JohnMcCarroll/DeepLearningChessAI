import requests
import time

### This script interfaces with Chess.com's API to retrieve PGN files of all Gramdmaster games that have occured on the site

# receive request and remove leading and trailing characters so GMs
# is just a comma separated list of GMs
GMs = requests.get("https://api.chess.com/pub/titled/GM").text
GMs = GMs.split("[")[1]
GMs = GMs.split("]")[0]


# iterate through GMs and retreive lists of their archive endpoints
for username in GMs.split(","):

    username = username.strip("\"")
    
    archives = requests.get("https://api.chess.com/pub/player/" + username + "/games/archives")
    print('archives status: ' + str(archives.status_code))
    archives = archives.text
    archives = archives.split("[")[1]
    archives = archives.split("]")[0]
    
    if archives:
        for archive in archives.split(","):
            archive = archive.strip("\"")
            print('archive: ' + archive)
            games = requests.get(archive  + "/pgn")
            open("games.pgn", 'ab').write(games.content)
    else: 
        time.sleep(15)
        print('sleepytime')


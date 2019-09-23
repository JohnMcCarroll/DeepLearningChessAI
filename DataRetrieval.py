import requests
import time

# receive request and remove leading and trailing characters so GMs
# is just a comma separated list of GMs
GMs = requests.get("https://api.chess.com/pub/titled/GM").text
GMs = GMs.split("[")[1]
GMs = GMs.split("]")[0]

foundPlace = False

# iterate through GMs and retreive lists of their archive endpoints
for username in GMs.split(","):

    username = username.strip("\"")
    if username == "vorenus_lucius":
        foundPlace = True
    
    if foundPlace:
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
                open("D:\Machine Learning\DeepLearningChessAI\Chess Database\Chess.com GMs\GMsTest.pgn", 'ab').write(games.content)
        else: 
            time.sleep(15)
            print('sleepytime junction!')       # garykasparov, ___ got sleeped - download his specifically


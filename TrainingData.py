import os
import re

"""
    TrainingData
    This class is to handle the data. Will read PGN files and populate hash table(s) with valid games' board positions
    (represented in tensor form) paired with expected predictions. Will randomly order training examples and form
    training and cross validation (and test?) sets.
"""

class TrainingData:
    def __init__(self, dataLocation):
        # Populate list of filenames in PGN directory, can skip step later (straight to reading files)
        for filename in os.listdir(dataLocation):
            file = open(dataLocation + "\\" + filename, 'r', 1)
            result = ""
            for line in file:
                fields = line.split(" ")
                if fields[0] == "1.":
                    # filter games decided by time or disconnection
                    if re.match(fields[len(fields) - 4], "forfeits"):
                        print("invalid game")
                    else:
                        # store result of game (1 = white wins, 0 = white loses, 1/2 = draw)
                        result = fields[len(fields) - 1].split("-")[0]
                        print(result)

                        # parse game moves and pair with result
                        "get tensor package imported and set up initial rep of board state to begin translation of PGN"













# testing
TD = TrainingData("D:\\Machine Learning\\Chess Database\\2000+ Games")
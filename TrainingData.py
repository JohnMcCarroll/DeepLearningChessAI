import os

"""
    TrainingData
    This class is to handle the data. Will read PGN files and populate hash table(s) with valid games' board positions
    (represented in tensor form) paired with expected predictions. Will randomly order training examples and form
    training and cross validation (and test?) sets.
"""

class TrainingData:
    def __init__(self, dataLocation):
        # Populate list of filenames in PGN directory, can skip step later (straight to reading files)
        self.dataFiles = list()
        for filename in os.listdir(dataLocation):
            self.dataFiles.append(filename)

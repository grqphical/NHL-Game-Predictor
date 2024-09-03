"""Processes data from money puck into a dataframe we can use for our machine learning model"""

import pandas as pd

hockey_data = pd.read_csv("data/VAN.csv")

hockey_data = hockey_data[
    ["team", "season", "name", "opposingTeam", "home_or_away", "gameDate"]
]

print(hockey_data.head())

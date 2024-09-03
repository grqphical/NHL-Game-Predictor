"""Processes data from money puck into a dataframe we can use for our machine learning model"""

import pandas as pd
import numpy as np

hockey_data = pd.read_csv("data/VAN.csv")

# Remove 4on4, 5on4, etc
hockey_data = hockey_data[hockey_data["situation"] == "all"]

hockey_data = hockey_data[
    [
        "team",
        "season",
        "name",
        "opposingTeam",
        "home_or_away",
        "gameDate",
        "xGoalsFor",
        "xGoalsAgainst",
        "goalsAgainst",
        "goalsFor",
        "shotAttemptsFor",
        "shotAttemptsAgainst",
        "penaltiesFor",
        "penaltiesAgainst",
    ]
]

# Generate an outcome column
hockey_data["outcome"] = np.where(
    hockey_data["goalsFor"] > hockey_data["goalsAgainst"], 1, 0
)

print(hockey_data.head())

"""Processes data from money puck into a dataframe we can use for our machine learning model"""

import pandas as pd
import numpy as np


def load_and_process_data(file: str) -> pd.DataFrame:
    hockey_data = pd.read_csv(file)

    # Remove 4on4, 5on4, etc
    hockey_data = hockey_data[hockey_data["situation"] == "5on5"]

    hockey_data = hockey_data[
        [
            "team",
            "opposingTeam",
            "home_or_away",
            "gameDate",
            "goalsAgainst",
            "goalsFor",
            "shotAttemptsFor",
            "penaltiesFor",
            "missedShotsFor",
        ]
    ]

    # Generate an outcome column
    hockey_data["outcome"] = np.where(
        hockey_data["goalsFor"] > hockey_data["goalsAgainst"], 1, 0
    )

    # Convert home or away to either 1 for home or 0 for away
    hockey_data["home_or_away"] = np.where(hockey_data["home_or_away"] == "HOME", 1, 0)

    # Convert date to datetime object
    hockey_data["gameDate"] = pd.to_datetime(hockey_data["gameDate"], format="%Y%m%d")

    # Add a time delta representing how many days have passed since that teams last game
    hockey_data["daysSinceLastGame"] = (
        hockey_data.groupby("team")["gameDate"].diff().dt.days.fillna(0)
    )

    # Get the current day of the week
    hockey_data["dayOfWeek"] = hockey_data["gameDate"].dt.day_of_week

    return hockey_data

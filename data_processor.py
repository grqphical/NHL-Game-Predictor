"""Processes data from money puck into a dataframe we can use for our machine learning model"""

import pandas as pd
import numpy as np


def add_rolling_averages(
    group: pd.DataFrame, cols: list[str], new_cols: list[str]
) -> pd.DataFrame:
    group = group.sort_values("gameDate")
    rolling_stats = group[cols].rolling(5, closed="left").mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group


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
            "faceOffsWonFor",
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

    # add the rolling averages for stats from the past three games

    cols = [
        "goalsAgainst",
        "goalsFor",
        "shotAttemptsFor",
        "penaltiesFor",
        "missedShotsFor",
        "faceOffsWonFor",
    ]
    new_cols = [f"{c}_rolling" for c in cols]

    hockey_data_with_rolling = hockey_data.groupby("team").apply(
        lambda x: add_rolling_averages(x, cols, new_cols)
    )
    hockey_data = hockey_data_with_rolling.droplevel("team")

    # reset the index
    hockey_data.index = range(hockey_data.shape[0])

    return hockey_data

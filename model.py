"""Random forest classification model"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_processor import load_and_process_data
from sklearn.metrics import precision_score
import pandas as pd


def train_model(data: pd.DataFrame):
    training_data = data[data["gameDate"].dt.year < 2023]
    testing_data = data[data["gameDate"].dt.year == 2023]

    predictors = [
        "home_or_away",
        "dayOfWeek",
        "daysSinceLastGame",
        "goalsAgainst_rolling",
        "goalsFor_rolling",
        "shotAttemptsFor_rolling",
        "penaltiesFor_rolling",
        "missedShotsFor_rolling",
        "faceOffsWonFor_rolling",
    ]

    model = RandomForestClassifier(random_state=44)
    model.fit(training_data[predictors], training_data["outcome"])

    preds = model.predict(testing_data[predictors])

    print(accuracy_score(testing_data["outcome"], preds))

    combined = pd.DataFrame(dict(actual=testing_data["outcome"], predicted=preds))
    confusion_matrix = pd.crosstab(
        index=combined["actual"], columns=combined["predicted"]
    )
    print(confusion_matrix)


if __name__ == "__main__":
    data = load_and_process_data("data/all_teams.csv")
    train_model(data)

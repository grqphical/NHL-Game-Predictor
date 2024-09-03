"""Random forest classification model"""

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from data_processor import load_and_process_data
import pandas as pd
from sklearn.metrics import classification_report


def train_gradient_boost_model(data: pd.DataFrame):
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
        "corsiPercentage_rolling",
        "homeAdvantage",
    ]

    model = HistGradientBoostingClassifier(
        random_state=44,
        class_weight="balanced",
        max_depth=8,
        max_iter=500,
        learning_rate=0.5,
        l2_regularization=0.5,
    )

    model.fit(training_data[predictors], training_data["outcome"])

    preds = model.predict(testing_data[predictors])

    print(accuracy_score(testing_data["outcome"], preds))

    print(classification_report(testing_data["outcome"], preds))


def train_random_forest_model(data: pd.DataFrame):
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
        "corsiPercentage_rolling",
        "homeAdvantage",
    ]

    model = RandomForestClassifier(
        random_state=44,
        n_estimators=250,
        n_jobs=10,
    )
    model.fit(training_data[predictors], training_data["outcome"])

    preds = model.predict(testing_data[predictors])

    print(accuracy_score(testing_data["outcome"], preds))

    print(classification_report(testing_data["outcome"], preds))


if __name__ == "__main__":
    data = load_and_process_data("data/all_teams.csv")
    train_gradient_boost_model(data)

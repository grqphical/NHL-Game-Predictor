"""Random forest classification model"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_processor import load_and_process_data
import pandas as pd


def train_model(data: pd.DataFrame):
    training_data = data[data["gameDate"].dt.year < 2023]
    testing_data = data[data["gameDate"].dt.year == 2023]

    predictors = ["home_or_away", "dayOfWeek", "daysSinceLastGame"]

    model = RandomForestClassifier()
    model.fit(training_data[predictors], training_data["outcome"])

    preds = model.predict(testing_data[predictors])

    print(accuracy_score(testing_data["outcome"], preds))


if __name__ == "__main__":
    data = load_and_process_data("data/VAN.csv")
    train_model(data)

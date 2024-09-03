"""Random forest classification model"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from data_processor import load_and_process_data
import pandas as pd
from sklearn.metrics import classification_report
import argparse
import pickle


def train_random_forest_model(data: pd.DataFrame):
    training_data = data[data["gameDate"].dt.year < 2020]
    testing_data = data[data["gameDate"].dt.year >= 2020]

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
        random_state=44, n_estimators=250, n_jobs=10, class_weight="balanced"
    )
    model.fit(training_data[predictors], training_data["outcome"])

    preds = model.predict(testing_data[predictors])

    print(accuracy_score(testing_data["outcome"], preds))

    print(classification_report(testing_data["outcome"], preds))

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NHL Game Predictor")
    subparsers = parser.add_subparsers(dest="command")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train the random forest model")
    train_parser.add_argument("data_file", type=str, help="Path to the data file")
    train_parser.add_argument(
        "model_file", type=str, help="Path to where you want to save the model file"
    )

    args = parser.parse_args()

    if args.command == "train":
        data = load_and_process_data(args.data_file)
        model = train_random_forest_model(data)

        with open(args.model_file, "wb") as f:
            pickle.dump(model, f)

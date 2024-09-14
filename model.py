"""Random forest classification model"""

from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils import class_weight
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from data_processor import load_and_process_data
import pandas as pd
from sklearn.metrics import classification_report
import argparse
import pickle
import math


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

    value_counts = training_data.value_counts("outcome").array.tolist()

    X_train = training_data[predictors]
    y_train = training_data["outcome"]

    X_test = testing_data[predictors]
    y_test = testing_data["outcome"]

    estimators = [
        (
            "rf",
            RandomForestClassifier(
                random_state=44,
                n_estimators=250,
                n_jobs=10,
                class_weight={0: 0.677, 1: 1},
            ),
        ),
        (
            "histgb",
            XGBClassifier(
                random_state=44,
                learning_rate=0.01,
                max_depth=12,
                scale_pos_weight=math.sqrt(value_counts[0] / value_counts[1]),
                n_jobs=10,
            ),
        ),
        ("svc", LinearSVC(random_state=44, class_weight={0: 0.677, 1: 1})),
    ]

    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(
            random_state=44, class_weight={0: 0.677, 1: 1}
        ),
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print(accuracy_score(y_test, preds))

    print(classification_report(y_test, preds))

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

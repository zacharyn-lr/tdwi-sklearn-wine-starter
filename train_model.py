"""Train a wine cultivar classifier — implement per SPEC.md."""

from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path("models/wine_classifier.joblib")


def train_and_save_model(X_train, X_test, y_train, y_test, random_state=42):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


def main():
    from wine_data import get_train_test_split

    X_train, X_test, y_train, y_test = get_train_test_split()
    train_and_save_model(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()

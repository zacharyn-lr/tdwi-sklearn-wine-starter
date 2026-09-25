"""Acceptance tests for the wine cultivar classifier."""

from train_model import train_and_save_model
from wine_data import get_train_test_split


def test_holdout_accuracy():
    X_train, X_test, y_train, y_test = get_train_test_split()
    model = train_and_save_model(X_train, X_test, y_train, y_test)
    score = model.score(X_test, y_test)
    assert score >= 0.95, f"Holdout accuracy should be at least 0.95 (got {score:.3f})"

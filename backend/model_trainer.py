"""
model_trainer.py
----------------
Trains a Random Forest classifier on historical OpenSky flight data
to predict whether a departure will be on time.

Usage:
    python model_trainer.py --callsign SQ285
"""

import json
import joblib
import argparse
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "model"
MODEL_DIR.mkdir(exist_ok=True)


def load_data(callsign: str) -> list[dict]:
    path = DATA_DIR / f"{callsign.upper()}_history.json"
    if not path.exists():
        raise FileNotFoundError(f"No data file found at {path}. Run data_fetcher.py first.")
    with open(path) as f:
        return json.load(f)


def build_features(flights: list[dict]):
    airport_encoder = LabelEncoder()
    airports = [f.get("departure_airport") or "UNKNOWN" for f in flights]
    airport_encoder.fit(airports)

    X, y = [], []
    for f in flights:
        if f.get("on_time") is None:
            continue
        X.append([
            f.get("day_of_week", 0),
            f.get("departure_hour", 0),
            f.get("departure_minute", 0),
            f.get("month", 1),
            airport_encoder.transform([f.get("departure_airport") or "UNKNOWN"])[0],
        ])
        y.append(f["on_time"])

    return np.array(X), np.array(y), airport_encoder


def train(callsign: str):
    print(f"Loading data for {callsign.upper()} ...")
    flights = load_data(callsign)

    if len(flights) < 10:
        print(f"Only {len(flights)} records — model may not be reliable.")

    X, y, encoder = build_features(flights)
    print(f"Built feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"On-time rate in training data: {y.mean():.1%}")

    if len(set(y)) < 2:
        print("All flights have the same label — can't train a meaningful classifier.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Performance (test set):")
    print(f"   Accuracy: {acc:.1%}")
    print(classification_report(y_test, y_pred, target_names=["Delayed", "On Time"]))

    model_path = MODEL_DIR / f"{callsign.upper()}_model.pkl"
    encoder_path = MODEL_DIR / f"{callsign.upper()}_encoder.pkl"
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    print(f"Model saved to {model_path}")
    print(f"Encoder saved to {encoder_path}")

    return model, encoder


def predict_single(callsign: str, day_of_week: int, hour: int,
                   minute: int, month: int, airport: str) -> dict:
    model_path = MODEL_DIR / f"{callsign.upper()}_model.pkl"
    encoder_path = MODEL_DIR / f"{callsign.upper()}_encoder.pkl"

    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    if airport not in encoder.classes_:
        airport = encoder.classes_[0]

    features = np.array([[
        day_of_week,
        hour,
        minute,
        month,
        encoder.transform([airport])[0]
    ]])

    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    return {
        "on_time_prediction": bool(prediction),
        "on_time_probability": round(float(proba[1]), 3),
        "delayed_probability": round(float(proba[0]), 3),
        "confidence": round(float(max(proba)), 3),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--callsign", required=True)
    args = parser.parse_args()
    train(args.callsign)

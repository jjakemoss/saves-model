import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Directory where the team schedules are stored
TEAM_SCHEDULES_DIR = "team_schedules"
MODEL_FILENAME = "goalie_saves_model.pkl"

def load_data():
    """Load all team schedule data from the specified directory."""
    all_data = []
    for file_name in os.listdir(TEAM_SCHEDULES_DIR):
        if file_name.endswith("_schedule.csv"):
            file_path = os.path.join(TEAM_SCHEDULES_DIR, file_name)
            team_data = pd.read_csv(file_path)
            all_data.append(team_data)
    return pd.concat(all_data, ignore_index=True)

def preprocess_data(data):
    """Preprocess the data for training."""
    # Ensure necessary columns exist
    required_columns = ["goalieSaves", "opponent_saves", "backToBack"]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Drop rows with missing values
    data = data.dropna(subset=required_columns)

    # Extract features and labels
    X = data[["opponent_saves", "backToBack"]]
    y = data["goalieSaves"]

    return X, y

def train_model():
    """Train the Random Forest model and save it to a file."""
    print("Loading data...")
    data = load_data()

    print("Preprocessing data...")
    X, y = preprocess_data(data)

    print("Training model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    print(f"Saving model to {MODEL_FILENAME}...")
    with open(MODEL_FILENAME, "wb") as file:
        pickle.dump(model, file)

    print("Model training complete.")

def predict():
    """Make predictions for a future matchup."""
    if not os.path.exists(MODEL_FILENAME):
        print("Model not found. Training a new model...")
        train_model()

    with open(MODEL_FILENAME, "rb") as file:
        model = pickle.load(file)

    print("Enter the details for the future matchup:")
    home_team = input("Home team: ")
    away_team = input("Away team: ")
    home_goalie_saves = int(input(f"Baseline saves for {home_team}: "))
    away_goalie_saves = int(input(f"Baseline saves for {away_team}: "))

    # Example prediction data structure (user-provided values)
    prediction_data = pd.DataFrame({
        "opponent_saves": [away_goalie_saves, home_goalie_saves],
        "backToBack": [0, 0]  # Assuming no back-to-back for simplicity; adjust as needed
    })

    probabilities = model.predict_proba(prediction_data)

    print("Prediction Results:")
    print(f"Chance {home_team} makes more than {home_goalie_saves} saves: {probabilities[0][1] * 100:.2f}%")
    print(f"Chance {away_team} makes more than {away_goalie_saves} saves: {probabilities[1][1] * 100:.2f}%")

if __name__ == "__main__":
    action = input("Enter 'train' to train the model or 'predict' to make a prediction: ").strip().lower()
    if action == "train":
        train_model()
    elif action == "predict":
        predict()
    else:
        print("Invalid action. Please enter 'train' or 'predict'.")

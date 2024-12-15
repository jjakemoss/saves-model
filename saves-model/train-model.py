import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = "S:/Documents/GitHub/saves-model/goalie_saves_dataset.csv"
data = pd.read_csv(file_path)

# Drop rows with missing target variable `goalieSaves`
data = data.dropna(subset=["goalieSaves"])

# Select relevant features and target variable
features = [
    "team_id", "shots_against_wma", "shots_against_home", "shots_against_away",
    "shots_against_b2b", "opponent_shots_for_wma", "isHome", "backToBack"
]
target = "goalieSaves"

# Separate numeric and categorical columns
numeric_features = data[features].select_dtypes(include=['number']).columns
categorical_features = data[features].select_dtypes(exclude=['number']).columns

# Fill missing values for numeric columns with mean
data[numeric_features] = data[numeric_features].fillna(data[numeric_features].mean())

# Fill missing values for categorical columns with the most frequent value
data[categorical_features] = data[categorical_features].fillna(data[categorical_features].mode().iloc[0])


# One-hot encode categorical features (if necessary)
data["isHome"] = data["isHome"].astype(int)
data["backToBack"] = data["backToBack"].astype(int)

le = LabelEncoder()
data['team_id'] = le.fit_transform(data['team_id'])

# Split the dataset into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Gradient Boosting Regressor
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation results
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# Save the trained model
model_filename = "goalie_saves_model.pkl"
joblib.dump(model, model_filename)

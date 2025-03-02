import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the combined dataset
combined_df = pd.read_csv("S:/Documents/GitHub/saves-model/combined_simplified.csv")

# Sort by game date
combined_df_sorted = combined_df.sort_values(by='gameDate')

# Define features and target
X_home_columns = ['isHome', 'teamSaves_last', 'opponentSaves_last', 'teamSaves_rolling', 'opponentSaves_rolling', 'teamSaves_rolling_3', 'opponentSaves_rolling_3', 'teamSaves_rolling_10', 'opponentSaves_rolling_10', 'teamSaves_rolling_15', 'opponentSaves_rolling_15', 'opponentTeamSaves_last', 'opponentOpponentSaves_last', 'opponentTeamSaves_rolling', 'opponentOpponentSaves_rolling', 'opponentTeamSaves_rolling_3', 'opponentOpponentSaves_rolling_3', 'opponentTeamSaves_rolling_10', 'opponentOpponentSaves_rolling_10', 'opponentTeamSaves_rolling_15', 'opponentOpponentSaves_rolling_15', 'teamCorsi_last', 'opponentCorsi_last', 'teamCorsi_rolling', 'opponentCorsi_rolling', 'teamCorsi_rolling_3', 'opponentCorsi_rolling_3', 'teamCorsi_rolling_10', 'opponentCorsi_rolling_10', 'teamCorsi_rolling_15', 'opponentCorsi_rolling_15', 'opponentTeamCorsi_last', 'opponentOpponentCorsi_last', 'opponentTeamCorsi_rolling', 'opponentOpponentCorsi_rolling', 'opponentTeamCorsi_rolling_3', 'opponentOpponentCorsi_rolling_3', 'opponentTeamCorsi_rolling_10', 'opponentOpponentCorsi_rolling_10', 'opponentTeamCorsi_rolling_15', 'opponentOpponentCorsi_rolling_15', 'teamFenwick_last', 'opponentFenwick_last', 'teamFenwick_rolling', 'opponentFenwick_rolling', 'teamFenwick_rolling_3', 'opponentFenwick_rolling_3', 'teamFenwick_rolling_10', 'opponentFenwick_rolling_10', 'teamFenwick_rolling_15', 'opponentFenwick_rolling_15', 'opponentTeamFenwick_last', 'opponentOpponentFenwick_last', 'opponentTeamFenwick_rolling', 'opponentOpponentFenwick_rolling', 'opponentTeamFenwick_rolling_3', 'opponentOpponentFenwick_rolling_3', 'opponentTeamFenwick_rolling_10', 'opponentOpponentFenwick_rolling_10', 'opponentTeamFenwick_rolling_15', 'opponentOpponentFenwick_rolling_15', 'backToBack', 'team', 'opponent']
target_column = "teamSaves"

# Split into training and test sets (time-based)
train_size = int(len(combined_df_sorted) * 0.8)
train_df = combined_df_sorted[:train_size]
test_df = combined_df_sorted[train_size:]

X_train = train_df[X_home_columns].copy()
y_train = train_df[target_column].copy()
X_test = test_df[X_home_columns].copy()
y_test = test_df[target_column].copy()

# Scale the data (excluding categorical features)
numeric_columns = ['teamSaves_last', 'opponentSaves_last', 'teamSaves_rolling', 'opponentSaves_rolling', 'teamSaves_rolling_3', 'opponentSaves_rolling_3', 'teamSaves_rolling_10', 'opponentSaves_rolling_10', 'teamSaves_rolling_15', 'opponentSaves_rolling_15', 'opponentTeamSaves_last', 'opponentOpponentSaves_last', 'opponentTeamSaves_rolling', 'opponentOpponentSaves_rolling', 'opponentTeamSaves_rolling_3', 'opponentOpponentSaves_rolling_3', 'opponentTeamSaves_rolling_10', 'opponentOpponentSaves_rolling_10', 'opponentTeamSaves_rolling_15', 'opponentOpponentSaves_rolling_15', 'teamCorsi_last', 'opponentCorsi_last', 'teamCorsi_rolling', 'opponentCorsi_rolling', 'teamCorsi_rolling_3', 'opponentCorsi_rolling_3', 'teamCorsi_rolling_10', 'opponentCorsi_rolling_10', 'teamCorsi_rolling_15', 'opponentCorsi_rolling_15', 'opponentTeamCorsi_last', 'opponentOpponentCorsi_last', 'opponentTeamCorsi_rolling', 'opponentOpponentCorsi_rolling', 'opponentTeamCorsi_rolling_3', 'opponentOpponentCorsi_rolling_3', 'opponentTeamCorsi_rolling_10', 'opponentOpponentCorsi_rolling_10', 'opponentTeamCorsi_rolling_15', 'opponentOpponentCorsi_rolling_15', 'teamFenwick_last', 'opponentFenwick_last', 'teamFenwick_rolling', 'opponentFenwick_rolling', 'teamFenwick_rolling_3', 'opponentFenwick_rolling_3', 'teamFenwick_rolling_10', 'opponentFenwick_rolling_10', 'teamFenwick_rolling_15', 'opponentFenwick_rolling_15']

scaler = MinMaxScaler()
X_train.loc[:,numeric_columns] = scaler.fit_transform(X_train.loc[:,numeric_columns])
X_test.loc[:,numeric_columns] = scaler.transform(X_test.loc[:,numeric_columns])

# Initialize CatBoostRegressor with categorical features specified (using defaults)
cat_features = ['team', 'opponent']
catboost_model = CatBoostRegressor(random_state=42, verbose=0, cat_features=cat_features)

# Train the model
catboost_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = catboost_model.predict(X_test)

# Calculate error metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print error metrics
print(f"CatBoost - MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.kernel_ridge import KernelRidge

# Define the features and target
X_home_columns = ['isHome',
                  'teamSaves_last', 'opponentSaves_last',
                  'teamSaves_rolling_5', 'opponentSaves_rolling_5',
                  'teamSaves_rolling_3', 'opponentSaves_rolling_3',
                  'teamSaves_rolling_10', 'opponentSaves_rolling_10',
                  'teamSaves_rolling_15', 'opponentSaves_rolling_15',

                  'opponentTeamSaves_last', 'opponentOpponentSaves_last',
                  'opponentTeamSaves_rolling_5', 'opponentOpponentSaves_rolling_5',
                  'opponentTeamSaves_rolling_3', 'opponentOpponentSaves_rolling_3',
                  'opponentTeamSaves_rolling_10', 'opponentOpponentSaves_rolling_10',
                  'opponentTeamSaves_rolling_15', 'opponentOpponentSaves_rolling_15',

                  'teamCorsi_last', 'opponentCorsi_last',
                  'teamCorsi_rolling_5', 'opponentCorsi_rolling_5',
                  'teamCorsi_rolling_3', 'opponentCorsi_rolling_3',
                  'teamCorsi_rolling_10', 'opponentCorsi_rolling_10',
                  'teamCorsi_rolling_15', 'opponentCorsi_rolling_15',

                  'opponentTeamCorsi_last', 'opponentOpponentCorsi_last',
                  'opponentTeamCorsi_rolling_5', 'opponentOpponentCorsi_rolling_5',
                  'opponentTeamCorsi_rolling_3', 'opponentOpponentCorsi_rolling_3',
                  'opponentTeamCorsi_rolling_10', 'opponentOpponentCorsi_rolling_10',
                  'opponentTeamCorsi_rolling_15', 'opponentOpponentCorsi_rolling_15',

                  'teamFenwick_last', 'opponentFenwick_last',
                  'teamFenwick_rolling_5', 'opponentFenwick_rolling_5',
                  'teamFenwick_rolling_3', 'opponentFenwick_rolling_3',
                  'teamFenwick_rolling_10', 'opponentFenwick_rolling_10',
                  'teamFenwick_rolling_15', 'opponentFenwick_rolling_15',

                  'opponentTeamFenwick_last', 'opponentOpponentFenwick_last',
                  'opponentTeamFenwick_rolling_5', 'opponentOpponentFenwick_rolling_5',
                  'opponentTeamFenwick_rolling_3', 'opponentOpponentFenwick_rolling_3',
                  'opponentTeamFenwick_rolling_10', 'opponentOpponentFenwick_rolling_10',
                  'opponentTeamFenwick_rolling_15', 'opponentOpponentFenwick_rolling_15',

                  'backToBack']
y_home_column = 'teamSaves'

df = pd.read_csv('S:/Documents/GitHub/saves-model/combined_simplified.csv')

# Drop rows with NA values
df_cleaned = df.dropna(subset=X_home_columns + [y_home_column])

# Split data into features (X) and target (y)
X = df_cleaned[X_home_columns]
y = df_cleaned[y_home_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = [
    LinearRegression(),
    SVR(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    KNeighborsRegressor(),
    MLPRegressor(),
    XGBRegressor(),
    LGBMRegressor(),
    Ridge(),
    Lasso(),
    ElasticNet(),
    KernelRidge()
]

# Train and evaluate models
for model in models:
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Print the evaluation metrics
    print(f"Model: {model.__class__.__name__}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-squared (R2): {r2}")
    print("-" * 20)
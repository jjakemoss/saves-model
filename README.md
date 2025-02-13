# NHL Goalie Save Prediction Model
Uses neural networks and the NHL API to predict the probability that an NHL team's goalie will have more saves than a user-provided threshold.

## To Run
In order to run the model, all that needs to be done is to call the [saves-model/run-all-scripts.py](https://github.com/jjakemoss/saves-model/blob/3aec1f9889cea0f7ac06d96501e29336445ee562/saves-model/run-all-scripts.py) script. This script runs all of the [get-team-stats-simplified.py](https://github.com/jjakemoss/saves-model/blob/3aec1f9889cea0f7ac06d96501e29336445ee562/saves-model/get-team-stats-simplified.py), [combine-individual-csvs.py](https://github.com/jjakemoss/saves-model/blob/3aec1f9889cea0f7ac06d96501e29336445ee562/saves-model/combine-individual-csvs.py), and [current-best-model.py](https://github.com/jjakemoss/saves-model/blob/3aec1f9889cea0f7ac06d96501e29336445ee562/saves-model/current-best-model.py) scripts, which collect all necessary data before training the model and then allowing for user-input team names and thresholds (ex. ANA 24 BOS 25.5).

## Current Error Statistics
MAE: 1.50, MSE: 9.08, RMSE: 3.01, R²: 0.78

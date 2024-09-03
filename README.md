# NHL Game Predictor

I built this using data from MoneyPuck.com to predict the outcome of NHL games based on data such as the date, the rolling average of past game stats, whether or not it's a home game and more.

The model is a RandomForestClassifier model with tuned parameters. The data is split based on the year it was from so training data is from 2008-2019 and testing data
is from 2020-2023

In testing it has about a ~60% accuracy rating however I have yet to test it based on real world games

This project is still a WIP, I plan to add a web interface to allow you to customize team matchups and predict upcoming NHL games using the NHL API

## Where to get Dataset

Here: https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv

## License

MIT License

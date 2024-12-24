# Boston Housing Price Prediction

This Jupyter Notebook explores and predicts Boston housing prices using various machine learning models.

## Data and Preprocessing

The notebook begins by loading the housing dataset ("data.csv").  It handles missing numerical values using the mean imputation strategy.  Exploratory data analysis (EDA) is performed using visualizations:

- Pair plots to show relationships between features.
- Distribution plots for target variable ('MEDV').
- Correlation heatmaps to identify correlations among features.
- Joint plots to examine the relationship between specific features.
- Box plots and Violin plots for visualizing the distribution of 'MEDV' across different categories.


Data preprocessing steps include:

1. Stratified train-test splitting to maintain the distribution of 'CHAS' in both sets.
2. Separating features (X) from the target variable (y).
3. Building a pipeline for data transformation, including median imputation and standardization using `StandardScaler`.

## Model Training and Evaluation

Multiple regression models (Linear Regression, Decision Tree, and Random Forest) are trained and evaluated using 10-fold cross-validation with negative mean squared error as the scoring metric.  The results (RMSE scores, mean, and standard deviation) for each model are displayed.

Hyperparameter tuning is performed on the RandomForestRegressor using GridSearchCV to optimize model performance.


## Model Deployment and Prediction

The best-performing model (determined by GridSearchCV) is saved using `joblib`.  The notebook demonstrates how to load the saved model and make predictions on new data. A sample prediction is included.

## Requirements

The notebook requires the following libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

To install these libraries, you can use pip.

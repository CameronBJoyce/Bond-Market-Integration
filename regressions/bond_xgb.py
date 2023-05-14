import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

class BondMarketXGBoost:
    def __init__(self, target_variable):
        self.target_variable = target_variable
        self.model = xgb.XGBRegressor()
        
    def train(self, dataset):
        X = dataset.drop(self.target_variable, axis=1)
        y = dataset[self.target_variable]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        train_predictions = self.model.predict(X_train)
        train_mse = mean_squared_error(y_train, train_predictions)
        train_mae = mean_absolute_error(y_train, train_predictions)
        print("Training set performance:")
        print("Mean Squared Error:", train_mse)
        print("Mean Absolute Error:", train_mae)
        
        test_predictions = self.model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        print("\nTesting set performance:")
        print("Mean Squared Error:", test_mse)
        print("Mean Absolute Error:", test_mae)
        
    def predict(self, data):
        predictions = self.model.predict(data)
        return predictions

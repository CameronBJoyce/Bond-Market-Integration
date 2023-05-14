import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

class BondMarketFeatureAnalyzer:
    def __init__(self, target_variable):
        self.target_variable = target_variable
        
    def analyze_features(self, dataset):
        # Separate features and target variable
        X = dataset.drop(self.target_variable, axis=1)
        y = dataset[self.target_variable]
        
        # Perform feature importance ranking
        feature_importances = self.rank_features(X, y)
        
        # Perform correlation analysis
        correlation_matrix = self.compute_correlation_matrix(dataset)
        
        # Visualize results
        self.visualize_results(feature_importances, correlation_matrix)
        
        # Select top features based on analysis
        selected_features = self.select_features(feature_importances, correlation_matrix)
        
        return selected_features
    
    def rank_features(self, X, y):
        # Train a Random Forest model to rank feature importances
        model = RandomForestRegressor()
        model.fit(X, y)
        
        # Retrieve feature importances
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
        feature_importances = feature_importances.sort_values(ascending=False)
        
        return feature_importances
    
    def compute_correlation_matrix(self, dataset):
        # Compute the correlation matrix
        correlation_matrix = dataset.corr()
        
        return correlation_matrix
    
    def visualize_results(self, feature_importances, correlation_matrix):
        # Visualize feature importances
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importances.values, y=feature_importances.index)
        plt.title('Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.show()
        
        # Visualize correlation matrix
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
    
    def select_features(self, feature_importances, correlation_matrix):
        # Select top features based on a combination of feature importance and correlation analysis
        top_features = feature_importances[feature_importances > 0.05].index.tolist()
        correlated_features = self.get_correlated_features(correlation_matrix, threshold=0.7)
        
        selected_features = list(set(top_features + correlated_features))
        
        return selected_features
    
    def get_correlated_features(self, correlation_matrix, threshold):
        # Find highly correlated features
        correlated_features = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    correlated_features.add(correlation_matrix.columns[i])
                    correlated_features.add(correlation_matrix.columns[j])
        
        return list(correlated_features)

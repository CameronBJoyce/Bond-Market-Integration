import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

class BondMarketDataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.label_encoder = LabelEncoder()
        
    def preprocess_data(self, dataset):
        # Drop irrelevant columns
        dataset = dataset.drop(['Date'], axis=1)
        
        # Handle missing values
        dataset = self.handle_missing_values(dataset)
        
        # Normalize or scale features
        dataset = self.normalize_features(dataset)
        
        # Transform categorical variables
        dataset = self.transform_categorical_variables(dataset)
        
        return dataset
    
    def handle_missing_values(self, dataset):
        # Identify columns with missing values
        columns_with_missing = dataset.columns[dataset.isnull().any()].tolist()
        
        # Impute missing values with mean
        dataset[columns_with_missing] = self.imputer.fit_transform(dataset[columns_with_missing])
        
        return dataset
    
    def normalize_features(self, dataset):
        # Select columns to normalize
        columns_to_normalize = ['BondYield', 'BondSpread', 'InflationRate', 'GDPGrowth', 'ExchangeRate', 'UnemploymentRate',
                                'ForeignInvestment', 'BondDuration', 'GovernmentDebt', 'TradeBalance', 'StockMarketIndex',
                                'BondMarketSize']
        
        # Normalize selected columns
        dataset[columns_to_normalize] = self.scaler.fit_transform(dataset[columns_to_normalize])
        
        return dataset
    
    def transform_categorical_variables(self, dataset):
        # Encode CreditRating column
        dataset['CreditRating'] = self.label_encoder.fit_transform(dataset['CreditRating'])
        
        # Encode Country column using one-hot encoding
        dataset = pd.get_dummies(dataset, columns=['Country'], drop_first=True)
        
        return dataset

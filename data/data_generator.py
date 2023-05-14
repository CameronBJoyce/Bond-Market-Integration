import pandas as pd
import numpy as np

class BondMarketDatasetGenerator:
    def __init__(self, num_samples):
        self.num_samples = num_samples
        
    def generate_dataset(self):
        # Define the columns of the dataset
        columns = ['Date', 'Country', 'BondYield', 'BondSpread', 'InflationRate', 'GDPGrowth', 'ExchangeRate', 'UnemploymentRate',
                   'CreditRating', 'ForeignInvestment', 'BondDuration', 'GovernmentDebt', 'TradeBalance', 'StockMarketIndex',
                   'BondMarketSize']
        
        # Generate random data for the dataset
        data = {
            'Date': pd.date_range(start='01-01-2010', periods=self.num_samples, freq='M'),
            'Country': np.random.choice(['USA', 'Germany', 'Japan', 'UK', 'China'], size=self.num_samples,  p=[0.3, 0.2, 0.2, 0.15, 0.15]),
            'BondYield': self.generate_time_series(self.num_samples, mean=2.0, std=0.5),
            'BondSpread': self.generate_time_series(self.num_samples, mean=1.0, std=0.3),
            'InflationRate': self.generate_time_series(self.num_samples, mean=1.5, std=0.2),
            'GDPGrowth': self.generate_time_series(self.num_samples, mean=3.0, std=0.8),
            'ExchangeRate': self.generate_time_series(self.num_samples, mean=1.0, std=0.2),
            'UnemploymentRate': self.generate_time_series(self.num_samples, mean=5.0, std=1.0),
            'CreditRating': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB', 'B'], size=self.num_samples,  p=[0.2, 0.2, 0.2, 0.2, 0.15, 0.05]),
            'ForeignInvestment': self.generate_time_series(self.num_samples, mean=50.0, std=10.0),
            'BondDuration': self.generate_time_series(self.num_samples, mean=10.0, std=3.0),
            'GovernmentDebt': self.generate_time_series(self.num_samples, mean=70.0, std=15.0),
            'TradeBalance': self.generate_time_series(self.num_samples, mean=0.0, std=10.0),
            'StockMarketIndex': self.generate_time_series(self.num_samples, mean=20000, std=3000),
            'BondMarketSize': self.generate_time_series(self.num_samples, mean=500000, std=100000)
        }
        
        # Create a DataFrame from the generated data
        df = pd.DataFrame(data, columns=columns)
        
        return df
    
    def generate_time_series(self, length, mean, std):
        return np.random.normal(loc=mean, scale=std, size=length).cumsum()

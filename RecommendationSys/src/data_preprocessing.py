import pandas as pd

def load_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Inspect the first few rows of the data
    print(data.head())
    
    # Check for missing values
    print("Missing values:\n", data.isnull().sum())
    
    return data

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    data = pd.read_csv('data/raw/churn-bigml-20.csv')
    
    print(data.head())

    print("\nMissing values:")
    print(data.isnull().sum())

    data = pd.get_dummies(data, columns=['State', 'Area code'], drop_first=True)

    label_encoder = LabelEncoder()
    data['International plan'] = label_encoder.fit_transform(data['International plan'])
    data['Voice mail plan'] = label_encoder.fit_transform(data['Voice mail plan'])

    X = data.drop('Churn', axis=1)
    y = data['Churn']  
    return X, y

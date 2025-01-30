from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from sklearn.model_selection import train_test_split

X, y = preprocess_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = train_model(X_train, y_train)

evaluate_model(model, X_test, y_test)

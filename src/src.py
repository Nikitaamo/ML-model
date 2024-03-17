from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

def load_data():
    """Load the California housing dataset."""
    data = fetch_california_housing()
    X, y = data.data, data.target
    return X, y

def preprocess_data(X, y):
    """Split the dataset into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train the model using RandomForestRegressor and GridSearchCV for hyperparameter tuning."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    parameters = {'n_estimators': [100, 200], 'max_depth': [10, 20, 30]}
    clf = GridSearchCV(model, parameters, cv=5)
    clf.fit(X_train, y_train)
    print(f"Best parameters: {clf.best_params_}")
    return clf.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")

def visualize_predictions(y_test, predictions):
    """Visualize the actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.5)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted Housing Prices")
    plt.show()

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    evaluate_model(model, X_test, y_test)
    visualize_predictions(y_test, predictions)

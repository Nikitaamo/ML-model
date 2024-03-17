import ssl
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Skip SSL certificate verification (use with caution)
ssl._create_default_https_context = ssl._create_unverified_context

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
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    parameters = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
    clf = GridSearchCV(model, parameters, cv=3, n_jobs=-1)
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
    plt.scatter(y_test, predictions, alpha=0.5, c='r', label='Predicted Prices')
    plt.scatter(y_test, y_test, alpha=0.5, c='b', label='Actual Prices')
    plt.xlabel("Actual Prices (100k USD)")
    plt.ylabel("Predicted Prices (100k USD)")
    plt.title("Actual vs Predicted Housing Prices")
    plt.legend()
    plt.show()

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    evaluate_model(model, X_test, y_test)
    visualize_predictions(y_test, predictions)

if __name__ == "__main__":
    main()

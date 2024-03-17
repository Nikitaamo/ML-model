import ssl
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
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

def train_model(X_train, y_train, use_pca=False):
    """Train the model using RandomForestRegressor within a pipeline that optionally includes PCA."""
    if use_pca:
        pca = PCA(n_components=8)
        model = Pipeline(steps=[('pca', pca),
                                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))])
        parameters = {'regressor__n_estimators': [50, 100], 'regressor__max_depth': [10, 20]}
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        parameters = {'n_estimators': [50, 100], 'max_depth': [10, 20]}

    clf = GridSearchCV(model, parameters, cv=3, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(f"{'With PCA' if use_pca else 'Without PCA'} - Best parameters: {clf.best_params_}")
    return clf.best_estimator_

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse}")
    return predictions

def visualize_predictions(y_test, predictions_without_pca, predictions_with_pca):
    """Visualize the actual vs predicted values."""
    plt.figure(figsize=(20, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, predictions_without_pca, alpha=0.5, c='r', label='Predicted Prices (No PCA)')
    plt.scatter(y_test, y_test, alpha=0.5, c='b', label='Actual Prices')
    plt.xlabel("Actual Prices (100k USD)")
    plt.ylabel("Predicted Prices (100k USD)")
    plt.title("Actual vs Predicted Housing Prices (No PCA)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, predictions_with_pca, alpha=0.5, c='g', label='Predicted Prices (With PCA)')
    plt.scatter(y_test, y_test, alpha=0.5, c='b', label='Actual Prices')
    plt.xlabel("Actual Prices (100k USD)")
    plt.ylabel("Predicted Prices (100k USD)")
    plt.title("Actual vs Predicted Housing Prices (With PCA)")
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Train and evaluate the model without PCA
    model_without_pca = train_model(X_train, y_train, use_pca=False)
    predictions_without_pca = evaluate_model(model_without_pca, X_test, y_test)

    # Train and evaluate the model with PCA
    model_with_pca = train_model(X_train, y_train, use_pca=True)
    predictions_with_pca = evaluate_model(model_with_pca, X_test, y_test)

    # Visualize the predictions of both models
    visualize_predictions(y_test, predictions_without_pca, predictions_with_pca)

if __name__ == "__main__":
    main()

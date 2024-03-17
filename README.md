"""
# ML Model - California Housing Price Prediction

## Introduction
This machine learning project predicts housing prices using the RandomForestRegressor on the California Housing dataset. The model predicts prices based on features such as median income, housing age, average rooms, population, and more. The project includes the application of PCA for dimensionality reduction and evaluates its impact on performance.

## Model Details
Optimal parameters were found for the RandomForestRegressor: {'max_depth': 20, 'n_estimators': 100}. These parameters were chosen to capture the complex, non-linear relationships within the housing data, achieving a balance between the model's ability to learn and its generalization to new data. Notably, the model's performance was robust, even when dimensionality was reduced using PCA, though a slight performance drop was observed.

## Why PCA May or May Not Be Useful
- **Useful**: PCA can be effective if the features are highly correlated and the dataset has high dimensionality. By reducing dimensionality, PCA can mitigate the curse of dimensionality and overfitting.
- **Not Useful**: If dimensionality reduction leads to a significant loss of information, it can deteriorate model performance. This may happen if the variance in the data is essential and spread across many dimensions.

## Installation and Setup
Ensure Python 3 and the necessary libraries are installed. Install the required packages with the following command:
pip install -r requirements.txt

Run the model by executing the src.py script:
python src.py

The script will output logs to log.txt, including the model training process, parameter tuning details, and performance metrics like MSE, RMSE, MAE, and R² Score. Additionally, the script saves a visualization image img.png which compares the actual vs. predicted housing prices for models trained with and without PCA.

## Logging
The log.txt will include:
Details of the hyperparameter tuning process.
Reasons for choosing specific parameters.
Performance metrics calculated during model evaluation.

## Visualization
The script will generate an image img.png displaying a scatter plot that compares the actual housing prices with the predictions made by the model, illustrating the effectiveness of the model's predictions.

This content provides a full overview of your project, including an introduction, details about the model and PCA, installation instructions, logging, and visualization details. Remember to adapt any paths or commands to match the specific structure and requirements of your project.
"""

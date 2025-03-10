# Step 1: Import the required python packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Step 2: Load the dataset
file_path = "A:/rohit data/ml-projects/machine-learning-practice/salary_data.csv"  # Update your path
data = pd.read_csv(file_path)

# Step 3: Handling Categorical Variables
encoder = ColumnTransformer(transformers=[
    ('encoder', OneHotEncoder(drop='first'), ['Education', 'Location'])], remainder='passthrough')
data_encoded = encoder.fit_transform(data)

# Step 4: Extracting Dependent and Independent Variables
X = data_encoded[:, :-1]  # Independent variables
y = data_encoded[:, -1]   # Dependent variable (Salary)

# Step 5: Split data into Train/Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Applying the Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 7: Predict the result
y_pred = regressor.predict(X_test)

# Print model results
print("\nPredicted Salary Values:", y_pred)

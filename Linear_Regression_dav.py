# Step 1: Import the required Python packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load the dataset (Generating Data Instead of Using a CSV File)
np.random.seed(42)  # For reproducibility
experience = np.random.randint(0, 11, 30)  # Random Years of Experience (0 to 10)
salary = experience * 5000 + np.random.randint(-2000, 2000, 30)  # Salary with noise

# Convert to DataFrame
data = pd.DataFrame({'Experience': experience, 'Salary': salary})

# Step 3: Data Analysis (Print dataset info and first few rows)
print("Dataset Preview:")
print(data.head())  # Show first 5 rows
print("\nDataset Description:")
print(data.describe())  # Show statistical summary

# Step 4: Split dataset into dependent/independent variables
X = data[['Experience']]  # Independent variable
y = data['Salary']  # Dependent variable

# Step 5: Split data into Train/Test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Applying the Model (Train the model using training data)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step 7: Predict the result (Predict salary for X_test)
y_pred = regressor.predict(X_test)

# Step 7 (Continued): Plot the training and test results

# Plot training set data vs predictions
plt.scatter(X_train, y_train, color='blue', label="Actual Data")
plt.plot(X_train, regressor.predict(X_train), color='red', label="Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression (Training Data)")
plt.legend()
plt.show()

# Plot test set data vs predictions
plt.scatter(X_test, y_test, color='green', label="Actual Data")
plt.plot(X_train, regressor.predict(X_train), color='red', label="Regression Line")  # Same line as before
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Simple Linear Regression (Test Data)")
plt.legend()
plt.show()

# Display the linear equation y = mx + c
print(f"\nEquation of the regression line: y = {regressor.coef_[0]:.2f}x + {regressor.intercept_:.2f}")

# Step 8: Accuracy Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")  # Closer to 1 means better fit

:

🔥 Project: Electric Motor Temperature Prediction using Machine Learning
🔹 Step 1: Import Libraries
python
Copy
Edit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pandas: To read CSV file and handle data

numpy: For mathematical operations

matplotlib & seaborn: For creating graphs/plots

🔹 Step 2: Load the Dataset
python
Copy
Edit
data = pd.read_csv("measures_v2.csv")
This loads the motor sensor data from a CSV file.

The dataset contains readings like speed, torque, voltage, coolant temp, and motor temp.

🔹 Step 3: Explore the Data
python
Copy
Edit
print(data.head())
print(data.info())
print(data.describe())
head(): Shows the first 5 rows

info(): Gives data types and null value info

describe(): Gives stats like mean, std, etc.

🔹 Step 4: Check for Missing Values
python
Copy
Edit
data.isnull().sum()
Checks how many missing values are present in each column.

🔹 Step 5: Feature Selection
python
Copy
Edit
X = data[['motor_speed', 'torque', 'voltage', 'coolant']]
y = data['motor_temp']
X: The input features (sensor data)

y: The target variable (motor temperature)

🔹 Step 6: Split the Data
python
Copy
Edit
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Splits the data into 80% training and 20% testing.

🔹 Step 7: Train the Model
python
Copy
Edit
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
Trains a Linear Regression model on the training data.

🔹 Step 8: Make Predictions & Evaluate
python
Copy
Edit
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mean_squared_error: Measures average error

r2_score: Accuracy of the model (closer to 1 = better)

🔹 Step 9: Plot Actual vs Predicted
python
Copy
Edit
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Actual vs Predicted Motor Temperature")
plt.show()
A scatter plot to visualize how close the predictions are to actual values.

✅ Project Benefits
Predicts motor temperature in real time

Helps in preventing overheating

Useful for predictive maintenance

Practical application in Industrial IoT


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Step 1: Load the Dataset
data_path = r'C:\projects\measures_v2.csv\measures_v2.csv'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at: {data_path}")

data = pd.read_csv(data_path)
print("✅ Dataset Loaded Successfully")
print("Shape of data:", data.shape)
print("Column names:", data.columns.tolist())
print("Missing values:\n", data.isnull().sum())

# Step 2: Preprocessing
data.dropna(inplace=True)
target_col = 'coolant'
X = data.drop(columns=[target_col, 'profile_id'], errors='ignore')
y = data[target_col]

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Model Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 6: Model Evaluation
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📉 Mean Squared Error (MSE): {mse:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# Step 7: Save Model and Scaler
joblib.dump(model, 'motor_temp_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model and Scaler saved successfully.")

# Step 8: Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


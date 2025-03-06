import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Load dataset
file_path = "./DATA/charlottesville_weather.csv"
df = pd.read_csv(file_path)

# Convert 'time' to datetime and set as index
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)

# Select target variable (e.g., 'Avg Temp (°F)')
df['Avg Temp (°F)'] = df['Avg Temp (°F)'].interpolate()  # Fill missing values

# Function to perform ADF test
def adf_test(series):
    result = adfuller(series.dropna())  
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("Data is stationary.")
    else:
        print("Data is not stationary.")

# Check stationarity before differencing
print("Original Data ADF Test:")
adf_test(df['Avg Temp (°F)'])

# Differencing (First Order)
df['Avg Temp Diff'] = df['Avg Temp (°F)'].diff()

# Check stationarity after differencing
print("\nAfter First Differencing ADF Test:")
adf_test(df['Avg Temp Diff'])

# Plot original and differenced series
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
axes[0].plot(df['Avg Temp (°F)'], label='Original Avg Temp')
axes[0].set_title('Original Avg Temp (°F)')
axes[0].legend()

axes[1].plot(df['Avg Temp Diff'], label='First Order Differenced Temp', color='orange')
axes[1].set_title('First Order Differenced Avg Temp (°F)')
axes[1].legend()

plt.tight_layout()
plt.show()

# Fit ARIMA Model (Adjust p, d, q values based on results)
model = ARIMA(df['Avg Temp (°F)'], order=(1,1,1))
model_fit = model.fit()

# Print summary
print(model_fit.summary())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 📌 1. Load and Preprocess Data
file_path = "./DATA/charlottesville_weather.csv"  # Update path if needed
df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')

# Keep only relevant columns
df = df[['Avg Temp (°F)']]

# Fill missing values using interpolation
df['Avg Temp (°F)'] = df['Avg Temp (°F)'].interpolate()

# Extract Year and Month for better understanding
df['Year'] = df.index.year
df['Month'] = df.index.month

# Extract March 20 temperatures from historical data
march_20_temps = df[(df['Month'] == 3) & (df.index.day == 20)].copy()

# Compute average winter temperature (Dec–March)
df_winter = df[df['Month'].isin([12, 1, 2, 3])].copy()
df_winter['Season'] = df_winter['Year']
df_winter.loc[df_winter['Month'] == 12, 'Season'] += 1  # Shift Dec into the next winter

avg_winter_temp = df_winter.groupby('Season')['Avg Temp (°F)'].mean()

# Merge with March 20 temperatures
merged_data = march_20_temps.merge(avg_winter_temp, left_on='Year', right_index=True, suffixes=('_March20', '_WinterAvg'))

# 📌 2. Define SARIMA Model for Forecasting
p, d, q = 1, 1, 1  # ARIMA terms
P, D, Q, s = 1, 1, 1, 12  # Seasonal terms (s=12 for yearly seasonality)

sarima_model = SARIMAX(df['Avg Temp (°F)'], 
                       order=(p, d, q), 
                       seasonal_order=(P, D, Q, s), 
                       enforce_stationarity=False, 
                       enforce_invertibility=False)

sarima_result = sarima_model.fit()

# 📌 3. Forecast March 20 Temperature for Next Year
last_year = df.index[-1].year
forecast_date = pd.Timestamp(year=last_year + 1, month=3, day=20)
forecast_steps = (forecast_date - df.index[-1]).days

forecast = sarima_result.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

# Extract forecasted values and confidence intervals
forecast_mean = pd.Series(forecast.predicted_mean.values, index=forecast_index)
forecast_ci = forecast.conf_int()
forecast_ci.index = forecast_index

# Extract forecast for March 20
forecast_march20 = forecast_mean.loc[forecast_date]

# 📌 4. Visualizations

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1️⃣ Winter Average Temperature vs. March 20 Temperature Scatter Plot with Equal Axis Scaling
plt.figure(figsize=(8, 5))
sns.scatterplot(data=merged_data, x='Avg Temp (°F)_WinterAvg', y='Avg Temp (°F)_March20', color='blue')
plt.axhline(y=60, color='r', linestyle='--', label='60°F Threshold')
plt.xlabel("Average Winter Temperature (Dec–Mar)")
plt.ylabel("March 20 Temperature (°F)")
plt.title("Winter Temperature vs. March 20 Temperature")
plt.legend()
plt.grid(True)

# Ensuring the x and y axes have the same scale
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
axis_min = min(x_min, y_min)
axis_max = max(x_max, y_max)
plt.xlim(axis_min, axis_max)
plt.ylim(axis_min, axis_max)

plt.show()

# 2️⃣ Historical Winter Temperature Trends (Ordered Months: Dec, Jan, Feb, Mar)
df_winter['Month Name'] = df_winter['Month'].replace({12: 'Dec', 1: 'Jan', 2: 'Feb', 3: 'Mar'})
df_winter['Month Name'] = pd.Categorical(df_winter['Month Name'], categories=['Dec', 'Jan', 'Feb', 'Mar'], ordered=True)

plt.figure(figsize=(12, 5))
sns.lineplot(data=df_winter, x='Month Name', y='Avg Temp (°F)', hue='Season', marker='o', palette='coolwarm', ci=None)
sns.set_style("whitegrid")
plt.axhline(y=60, color='r', linestyle='--', label='60°F Threshold')
plt.title("Winter Temperature Trends (Dec–Mar)")
plt.xlabel("Month")
plt.ylabel("Temperature (°F)")
plt.legend(title='Winter Season', bbox_to_anchor=(1, 1))
plt.grid(True)
plt.show()

# 3️⃣ Forecasting Only March 2025 Temperatures
forecast_dates = pd.date_range(start='2025-03-01', end='2025-03-31', freq='D')
forecast_steps = len(forecast_dates)

# Generate forecast for March 2025
forecast = sarima_result.get_forecast(steps=forecast_steps)
forecast_mean = pd.Series(forecast.predicted_mean.values, index=forecast_dates)
forecast_ci = forecast.conf_int()
forecast_ci.index = forecast_dates

plt.figure(figsize=(8, 5))
forecast_mean.plot(label="Forecasted Temperature", color='blue')
plt.fill_between(forecast_dates, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='blue', alpha=0.2)
plt.axhline(y=60, color='r', linestyle='--', label='60°F Threshold')
plt.xlabel("Date")
plt.ylabel("Temperature (°F)")
plt.title("Forecast for March 2025 Temperatures")
plt.legend()
plt.grid(True)
plt.show()
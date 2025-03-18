import pandas as pd
import numpy as np
from datetime import datetime
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Load and sort the CSV file
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "..", "DATA", "charlottesville_weather.csv")
df = pd.read_csv(file_path, parse_dates=['time'])
df.sort_values('time', inplace=True)

# Handle missing values with linear interpolation then backfill
numeric_cols = ['Precipitation (mm)', 'Wind Direction (°)', 'Wind Speed (km/h)',
                'Pressure (hPa)', 'Avg Temp (°F)', 'Min Temp (°F)', 'Max Temp (°F)']
df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
df.bfill(inplace=True)

df.to_csv("cleaned_charlottesville_weather.csv", index=False)
print("Merged data saved to cleaned_charlottesville_weather.csv")

# Create Season_Year: December stays in the current year; Jan, Feb, Mar belong to the previous year.
df['Month'] = df['time'].dt.month
df['Season_Year'] = df['time'].dt.year
df.loc[df['Month'].isin([1, 2, 3]), 'Season_Year'] -= 1

# Mark dates on or before March 20
df['Is_Before_Mar20'] = df['time'].apply(lambda d: (d.month < 3) or (d.month == 3 and d.day <= 20))

# Define target: For each season, label as 1 if any day on or before March 20 has Max Temp (°F) >= 60°F, else 0.
def check_spring_arrival(group):
    group_before = group[group['Is_Before_Mar20']]
    if group_before.empty:
        return 0
    # Use the record for March 20 if available; otherwise, use the last day before March 21.
    march20 = group_before[group_before['time'].dt.day == 20]
    if not march20.empty:
        temp = march20.iloc[0]['Max Temp (°F)']
    else:
        temp = group_before.sort_values('time').iloc[-1]['Max Temp (°F)']
    return int(temp >= 60)

spring_arrivals = df.groupby('Season_Year').apply(
    lambda group: check_spring_arrival(group[['time', 'Max Temp (°F)', 'Is_Before_Mar20']])
).reset_index()
spring_arrivals.columns = ['Season_Year', 'Spring_Arrival']

# Aggregate season-level features (Dec–Mar)
season_agg = df.groupby('Season_Year').agg({
    'Avg Temp (°F)': ['mean', 'min', 'max', 'std'],
    'Precipitation (mm)': ['sum', 'mean'],
    'Pressure (hPa)': ['mean'],
    'Wind Speed (km/h)': ['mean']
}).reset_index()

# Flatten the aggregated columns
season_agg.columns = [
    'Season_Year',
    'AvgTemp_Mean', 'AvgTemp_Min', 'AvgTemp_Max', 'AvgTemp_Std',
    'Prcp_Sum', 'Prcp_Mean',
    'Pressure_Mean',
    'WindSpeed_Mean'
]

# Merge aggregated features with the target variable
merged_df = pd.merge(season_agg, spring_arrivals, on='Season_Year', how='inner')

print("Spring_Arrival distribution:")
print(merged_df['Spring_Arrival'].value_counts())
if merged_df['Spring_Arrival'].nunique() < 2:
    print("ERROR: Only one class found in the target variable. Adjust your threshold or data range.")
    exit()

# Save merged data to CSV
merged_df.to_csv("logReg_charlottesville_weather.csv", index=False)
print("Merged data saved to logReg_charlottesville_weather.csv")

# Prepare features and target for logistic regression
feature_cols = ['AvgTemp_Mean', 'AvgTemp_Min', 'AvgTemp_Max', 
                'AvgTemp_Std', 'Prcp_Sum', 'Prcp_Mean', 
                'Pressure_Mean', 'WindSpeed_Mean']
X = merged_df[feature_cols]
y = merged_df['Spring_Arrival']

# Scale features and split data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train logistic regression model 
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train, y_train)
print("Logistic Regression model trained successfully.")

# Coefficients Bar Chart
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': clf.coef_[0]
})
plt.figure(figsize=(8,6))
sns.barplot(data=coef_df, x='Coefficient', y='Feature')
plt.title("Logistic Regression Coefficients")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
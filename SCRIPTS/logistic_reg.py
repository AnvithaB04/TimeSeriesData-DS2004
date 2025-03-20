"""
--------------------------------------------------------------------------------
logistic_reg.py
--------------------------------------------------------------------------------
This script loads cleaned winter weather data for Charlottesville, aggregates
it by season (Dec–Mar), defines a binary target (Spring_Arrival), and trains
a logistic regression model to classify whether spring arrives by March 20
(based on a temperature threshold).
--------------------------------------------------------------------------------
"""

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
from imblearn.over_sampling import SMOTE  # Import SMOTE

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
merged_df.to_csv("merged_charlottesville_weather.csv", index=False)
print("Merged data saved to merged_charlottesville_weather.csv")

# Prepare features and target for logistic regression
feature_cols = ['AvgTemp_Mean', 'AvgTemp_Min', 'AvgTemp_Max', 
                'AvgTemp_Std', 'Prcp_Sum', 'Prcp_Mean', 
                'Pressure_Mean', 'WindSpeed_Mean']
X = merged_df[feature_cols]
y = merged_df['Spring_Arrival']

# Scale features and split data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert X_scaled back to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, index=y.index, columns=feature_cols)

# Ensure at least one `1` in the test set
positive_indices = y[y == 1].index.tolist()
negative_indices = y[y == 0].index.tolist()

if len(positive_indices) < 2:
    print("Not enough positive examples to guarantee a balanced test set!")
else:
    # Reserve one `1` for the test set
    test_positive_idx = [positive_indices[0]]
    remaining_positive_idx = positive_indices[1:]  # Keep at least one `1` in training

    # Remove the reserved `1` from the remaining dataset
    X_remaining = X_scaled_df.drop(index=test_positive_idx)
    y_remaining = y.drop(index=test_positive_idx)

    # Check if `y_remaining` has enough `1s` for stratified splitting
    if sum(y_remaining == 1) >= 2:
        stratify_option = y_remaining
    else:
        stratify_option = None  # Avoid stratification if too few `1s`

    # Perform train-test split on the remaining dataset
    X_train, X_test_temp, y_train, y_test_temp = train_test_split(
        X_remaining, y_remaining, test_size=0.2, random_state=42, stratify=stratify_option
    )

    # Manually add the `1` we reserved earlier to the test set
    X_test = np.vstack([X_test_temp, X_scaled_df.loc[test_positive_idx].values])
    y_test = np.append(y_test_temp, 1)

    # Print test set distribution
    print("Test Set Class Distribution After Fix:")
    print(pd.Series(y_test).value_counts())

print("Training Set Class Distribution After SMOTE:")
print(pd.Series(y_train).value_counts())

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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Predict on the test set
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Get probability estimates

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print performance metrics
print(f"Model Performance:\n"
      f"Accuracy: {accuracy:.2f}\n"
      f"Precision: {precision:.2f}\n"
      f"Recall: {recall:.2f}\n"
      f"F1 Score: {f1:.2f}\n"
      f"ROC AUC: {roc_auc:.2f}\n")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Define labels
labels = ["No Early Spring (0)", "Early Spring (1)"]

# Create confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot heatmap
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)

# Add labels
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")

# Show plot
plt.show()

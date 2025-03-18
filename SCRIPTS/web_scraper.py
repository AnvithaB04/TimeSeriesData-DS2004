import pandas as pd
from meteostat import Daily, Stations
from datetime import datetime

# Define location 
LOCATION_NAME = "Charlottesville"
START_YEAR = 2000
END_YEAR = 2025

# Get the closest weather station
stations = Stations()
stations = stations.nearby(38.0293, -78.4767)  # Charlottesville coordinates
station = stations.fetch(1)  # Get the nearest station
station_id = station.index[0] if not station.empty else None

if not station_id:
    print("No nearby weather station found. Try another data source.")
    exit()

# Storage for weather data
weather_data = []

# Fetch data for each winter season (December - February)
for year in range(START_YEAR, END_YEAR):
    start_date = datetime(year, 12, 1)
    end_date = datetime(year + 1, 3, 31)

    # Fetch historical daily data
    data = Daily(station_id, start=start_date, end=end_date)
    data = data.fetch()

    # Process data
    if not data.empty:
        data.reset_index(inplace=True)
        data.rename(columns={
            "tmax": "Max Temp (°C)",
            "tmin": "Min Temp (°C)",
            "tavg": "Avg Temp (°C)",
            "prcp": "Precipitation (mm)",
            "snow": "Snow Depth (cm)",
            "wdir": "Wind Direction (°)", 
            "wspd": "Wind Speed (km/h)",
            "wpgt": "Wind Gust (km/h)",
            "pres": "Pressure (hPa)",
            "tsun": "Sunshine Duration (minutes)"
        }, inplace=True)
        weather_data.append(data)

# Combine all years into a single DataFrame
if weather_data:
    df = pd.concat(weather_data, ignore_index=True)

    # Convert temperature columns from Celsius to Fahrenheit
    df["Avg Temp (°F)"] = ((df["Avg Temp (°C)"] * 9/5) + 32).round(2)
    df["Min Temp (°F)"] = ((df["Min Temp (°C)"] * 9/5) + 32).round(2)
    df["Max Temp (°F)"] = ((df["Max Temp (°C)"] * 9/5) + 32).round(2)

    # Drop old Celsius columns
    df.drop(columns=["Avg Temp (°C)", "Min Temp (°C)", "Max Temp (°C)"], inplace=True)

    # Drop empty columns and the "Year" column
    df.dropna(axis=1, how="all", inplace=True)  # Remove completely empty columns
    df.drop(columns=["Year"], errors="ignore", inplace=True)  # Remove Year if present

    # Convert the 'time' column to datetime and sort
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values('time', inplace=True)

    # Save the modified data to a CSV
    csv_filename = "charlottesville_weather.csv"
    df.to_csv(csv_filename, index=False)

    print(f"Cleaned weather data saved to {csv_filename} with temperatures in Fahrenheit")
else:
    print("No weather data available for the requested period.")

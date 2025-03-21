<!-- README for TimeSeriesDataProject4002 -->

# TimeSeriesDataProject4002

<!-- Project Overview -->
## 📌 Project Overview

This is a **Spring Arrival Weather Data** project. It leverages **Charlotesville weather data from December 1st to March 31st** to predict if **spring will arrive before March 20th**. The project involves **data collection, SARIMA modeling, machine learning modeling, and evaluation**.  

### Key Features:
- **Web Scraping** Charlotesville weather
- **SARIMA** for predicting march 2025 weather
- **Feature Engineering** for predictive modeling
- **Machine Learning** Logistical Regression modeling
- **Performance Evaluation** with RMSE, MAE, and R²

<!-- Section 1: Software & Platform -->
## Section 1: Software & Platform

This project was developed using the following software and tools:

- **Programming Language:** Python 3.x  
- **Development Environment:** VS Code  
- **Libraries & Packages:**
  - **pandas** (data manipulation)
  - **matplotlib**, **seaborn** (data visualization)
  - **scikit-learn** (machine learning models)
  - **meteostat** (API to extract weather data)
  - **statsmodels** (For SARIMA modeling)
- **Platform:** Developed and tested on Windows 10 and macOS

To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```
<!-- Section 2: Project Folder Structure -->
## Section 2: A Map of your documentation

TIMESERIESDATA-DS4002/  
├── DATA/  
│   ├── DataAppendix.pdf  
│   ├── charlottesville_weather.csv  
│   ├── cleaned_charlottesville_weather.csv  
│   └── merged_charlottesville_weather.csv  
├── OUTPUT/  
│   ├── DatasetGraphics/  
│   │    ├── daily_temperature_metrics.png  
│   │    └── weather-variables_heatmap.png  
│   └── Results/   
│   │   ├── average_temperatures_over_time.png  
│   │   ├── average_temperatures.png  
│   │   ├── confusion_matrix.png  
│   │   ├── log_reg_coefficients.png  
│   │   └── predicted_march_2025_values.png  
├── SCRIPTS/    
│   ├── arima_script.py  
│   ├── logistic_reg.py  
│   └── web_scraper.py  
├── LICENSE  
├── README.md  
└── requirements.txt  

<!-- Section 3: Instructions for Reproducing Results -->
## Section 3: Instructions for Reproducing Results
Follow these steps to reproduce our results:

1. **Set Up the Environment**  
   Ensure you have Python 3.x installed as well as the requirements.txt file
   Then run:
   ```bash
   pip install -r requirements.txt
   ```
2. **Collect & Prepare Data**
   - Navigate to the web_scraper.py python script and run it from the IDE. Alternativaly you can run the script by
   ```bash
   python ./SCRIPTS/web_scraper.py
   ```
4. **Train Models & Evaluate Performance**
   - Execute the arima_script.py script to train arima model and evaluate their performance.
   - Execute the logistic_reg.py script to train the logistic aggresion model and evaluate their performance.
5. **View Results**
   - Check the OUTPUT folder for performance tracking results and saved models.




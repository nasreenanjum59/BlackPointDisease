import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# File paths
input_file = r"C:\Users\s2118881\Downloads\Wheatdatasetnewexcel.xlsx"
output_file = r"C:\Users\s2118881\Downloads\Wheatdataset_cleaned_final.xlsx"

# Step 1: Load the dataset
data = pd.read_excel(input_file, sheet_name="Wheatdatasetnew")

# Step 2: Handle Missing Values
# Fill numerical missing values with column mean
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[num_cols] = data[num_cols].fillna(data[num_cols].mean())

# Fill categorical missing values with column mode
cat_cols = data.select_dtypes(include=['object']).columns
data[cat_cols] = data[cat_cols].fillna(data[cat_cols].mode().iloc[0])

# Step 3: Clean Weather Parameters
# Define valid ranges
temp_min, temp_max = 10, 50  # Degrees Celsius
precip_min, precip_max = 0, 1000  # mm for precipitation
humidity_min, humidity_max = 0, 100  # Percent for humidity

# Clip values to valid ranges
data['Average_Temperature'] = data['Average_Temperature'].clip(temp_min, temp_max)
data['Total_Precipitation_mm'] = data['Total_Precipitation_mm'].clip(precip_min, precip_max)
data['Humidity_%'] = data['Humidity_%'].clip(humidity_min, humidity_max)

# Step 4: Remove Outliers using IQR
def remove_outliers_iqr(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return column.clip(lower_bound, upper_bound)

# Apply IQR to weather features
weather_features = ['Average_Temperature', 'Total_Precipitation_mm', 'Humidity_%']
for feature in weather_features:
    data[feature] = remove_outliers_iqr(data[feature])

# Step 5: Encode Categorical Variables
label_encoder = LabelEncoder()
data["Pesticides_Encoded"] = label_encoder.fit_transform(data["Pesticides _Name"])

# Step 6: Normalize Weather Parameters using Z-Score
scaler = StandardScaler()
data[weather_features] = scaler.fit_transform(data[weather_features])

# Step 7: Save the Cleaned Dataset
data.to_excel(output_file, index=False)
print(f"Cleaned dataset saved successfully to: {output_file}")
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap

# Step 1: Load the Cleaned Dataset
input_file = r"C:\Users\s2118881\Downloads\Wheatdataset_cleaned_final.xlsx"
data = pd.read_excel(input_file)

# Step 2: Define Features and Target
X = data[['Average_Temperature', 'Humidity_%', 'Total_Precipitation_mm', 'Pesticides_Encoded']]
y = data['Black_Point_Incidence(%)']

# Map Pesticides_Encoded back to their names
pesticide_mapping = dict(zip(data['Pesticides_Encoded'], data['Pesticides _Name']))

# Initialize variables for Monte Carlo Simulation
n_simulations = 1000  # Updated to 1000 simulations
all_temperature = []
all_humidity = []
all_precipitation = []
all_interaction_values = []
all_pesticide_names = []

# Monte Carlo Simulations
print("Starting Monte Carlo Simulations...")
for i in range(n_simulations):
    print(f"Simulation {i + 1}/{n_simulations}")

    # Step 3: Split the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Step 4: Train XGBoost Model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
    model.fit(X_train, y_train)

    # Step 5: Initialize SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_interaction_values = explainer.shap_interaction_values(X_test)

    # Step 6: Collect Triple SHAP Interaction Values
    temp_idx = X.columns.get_loc("Average_Temperature")
    humid_idx = X.columns.get_loc("Humidity_%")
    precip_idx = X.columns.get_loc("Total_Precipitation_mm")

    # Compute interaction of all three weather variables
    triple_interaction = shap_interaction_values[:, temp_idx, humid_idx] + \
                         shap_interaction_values[:, temp_idx, precip_idx] + \
                         shap_interaction_values[:, humid_idx, precip_idx]

    all_temperature.extend(X_test["Average_Temperature"].values)
    all_humidity.extend(X_test["Humidity_%"].values)
    all_precipitation.extend(X_test["Total_Precipitation_mm"].values)
    all_interaction_values.extend(triple_interaction)
    all_pesticide_names.extend(
        [pesticide_mapping[code] for code in X_test["Pesticides_Encoded"].values]
    )

# Step 7: Convert to Numpy Arrays
all_temperature = np.array(all_temperature)
all_humidity = np.array(all_humidity)
all_precipitation = np.array(all_precipitation)
all_interaction_values = np.array(all_interaction_values)
all_pesticide_names = np.array(all_pesticide_names)

# Step 8: Plot Triple Interaction Effects
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Generate a distinct colormap for pesticides
unique_pesticides = list(set(all_pesticide_names))
cmap = get_cmap('tab10', len(unique_pesticides))  # Use 'tab10' for 10 distinct colors
color_map = ListedColormap(cmap.colors)

# Plot for each pesticide with distinct colors
for idx, pesticide in enumerate(unique_pesticides):
    mask = all_pesticide_names == pesticide
    ax.scatter(
        all_temperature[mask],
        all_humidity[mask],
        all_precipitation[mask],
        c=[color_map(idx)] * sum(mask),  # Assign unique color
        label=pesticide,
        alpha=0.8,
        edgecolors="k"  # Optional: Add black edge for better visibility
    )

# Step 9: Customize the Plot
ax.set_title(
    "Combined Effect of Temperature, Humidity, and Precipitation\non Pesticides (Averaged over 1000 Simulations)",
    fontsize=14)
ax.set_xlabel("Temperature (°C)", fontsize=12)
ax.set_ylabel("Humidity (%)", fontsize=12)
ax.set_zlabel("Precipitation (mm)", fontsize=12)
ax.legend(title="Pesticide Type", loc="upper left", fontsize=10)
plt.tight_layout()

# Step 10: Save and Show the Plot
plt.savefig("distinct_triple_interaction_climate_pesticides_1000.png", dpi=300)
plt.show()

print("Monte Carlo Simulations Completed. Plot saved as 'distinct_triple_interaction_climate_pesticides_1000.png'.")

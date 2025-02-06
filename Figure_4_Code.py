import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

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
interaction_results = {
    "Temperature-Humidity": {"x": [], "y": [], "interaction": [], "pesticides": []},
    "Temperature-Precipitation": {"x": [], "y": [], "interaction": [], "pesticides": []},
    "Humidity-Precipitation": {"x": [], "y": [], "interaction": [], "pesticides": []},
}

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

    # Step 6: Collect SHAP Interaction Values for Each Pair
    pairs = [
        ("Average_Temperature", "Humidity_%", "Temperature-Humidity"),
        ("Average_Temperature", "Total_Precipitation_mm", "Temperature-Precipitation"),
        ("Humidity_%", "Total_Precipitation_mm", "Humidity-Precipitation"),
    ]

    for feature_x, feature_y, label in pairs:
        x_idx = X.columns.get_loc(feature_x)
        y_idx = X.columns.get_loc(feature_y)
        pesticide_idx = X.columns.get_loc("Pesticides_Encoded")

        interaction = shap_interaction_values[:, x_idx, y_idx]

        interaction_results[label]["x"].extend(X_test[feature_x].values)
        interaction_results[label]["y"].extend(X_test[feature_y].values)
        interaction_results[label]["interaction"].extend(interaction)
        interaction_results[label]["pesticides"].extend(
            [pesticide_mapping[code] for code in X_test["Pesticides_Encoded"].values]
        )

# Step 7: Generate Subplots for Pairwise Interactions
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

titles = [
    "Combined Effect of Temperature and Humidity",
    "Combined Effect of Temperature and Precipitation",
    "Combined Effect of Humidity and Precipitation",
]
labels = ["Temperature-Humidity", "Temperature-Precipitation", "Humidity-Precipitation"]

for ax, title, label in zip(axes, titles, labels):
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(label.split("-")[0] + " (%)", fontsize=12)
    ax.set_ylabel("SHAP Interaction Value (Climate Variables and Pesticides)", fontsize=12)

    data = interaction_results[label]
    unique_pesticides = set(data["pesticides"])

    for pesticide in unique_pesticides:
        mask = np.array(data["pesticides"]) == pesticide
        ax.scatter(
            np.array(data["x"])[mask],
            np.array(data["interaction"])[mask],
            label=pesticide,
            alpha=0.7,
        )

    ax.legend(title="Pesticide Type", loc="upper right", fontsize=10)
    ax.grid(True)

plt.tight_layout()
plt.savefig("climate_variable_interactions_pesticides_1000_simulations.png", dpi=300)
plt.show()

print("Monte Carlo Simulations Completed. Plot saved as 'climate_variable_interactions_pesticides_1000_simulations.png'.")

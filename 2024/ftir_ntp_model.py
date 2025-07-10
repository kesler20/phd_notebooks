from scipy.signal import savgol_filter
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, r2_score
import os
from matplotlib import pyplot as plt
import matplotlib.cm as cm


# Function to apply Standard Normal Variate (SNV)
def standard_normal_variate(X):
    """
    Apply Standard Normal Variate (SNV) transformation row-wise.
    """
    X_snv = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    return X_snv


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CONSIDER_REPLICATES = True
REMOVE_OUTLIERS = False
VISUALISE_PREPROCESSED_DATA = True
VISUALISE_RESULTS = False

# Get the data and drop the wavenumber column
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Get the data and drop the wavenumber column
data = pd.read_csv(
    os.path.join(
        CURRENT_DIR, "wiz-app", "data_sets", "240403 4 NTP combinations Graph Data.csv"
    )
)
wavenumber = data["Wavenumber (1/cm)"].values
data.drop(columns=["Wavenumber (1/cm)"], inplace=True)

# Create the X and Y arrays
X_list = []
Y_list = []
last_col_name = ""
for col in data.columns:
    if "Unnamed" not in col:
        X_list.append(data[col].values.tolist())
        Y_list.append([float(val) for val in col.replace("'", ".").strip().split(" ")])
        last_col_name = col
    else:
        if last_col_name and CONSIDER_REPLICATES:
            X_list.append(data[col].values.tolist())
            Y_list.append(
                [
                    float(val)
                    for val in last_col_name.replace("'", ".").strip().split(" ")
                ]
            )
X = np.array(X_list)
Y = np.array(Y_list)

# generate a dataframe with the clean X and Y data and save it
# pd.DataFrame(data=np.hstack((X, Y))).to_csv(
#     "cleaned_data.csv", index=False
# )


if VISUALISE_RESULTS:
    df = pd.DataFrame(data={f"sample {i}": X[i, :] for i in range(X.shape[0])})
    df.plot()
    plt.show()

# Apply Savitzky-Golay filter row-wise
X_savgol = savgol_filter(X, window_length=11, polyorder=3, axis=1)

# Apply Standard Normal Variate (SNV) row-wise
X_snv = standard_normal_variate(X_savgol)


X = X_snv

if REMOVE_OUTLIERS:
    for i in range(3):
        max_diff = [0, 0]
        for i in range(X.shape[0]):
            sample = X[i, :]
            diff = sample.max() - sample.min()
            if diff > max_diff[0]:
                max_diff[0] = diff
                max_diff[1] = i
        X = np.delete(X, max_diff[1], axis=0)
        Y = np.delete(Y, max_diff[1], axis=0)

if VISUALISE_PREPROCESSED_DATA:
    df = pd.DataFrame(data={f"sample {i}": X[i, :] for i in range(X.shape[0])})
    df.plot()
    plt.title("Raw Samples")
    plt.show()

    # Create a new figure
    fig, ax = plt.subplots()

    # Get a colormap
    cmap = cm.get_cmap("viridis")  # or any other colormap

    # Plot each sample with color based on sample number
    for i in range(X.shape[0]):
        color = cmap(i / X.shape[0])
        ax.plot(wavenumber, X[i, :], color=color)

    plt.title("Samples After Processing")
    plt.show()


# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

# Create a PLS model
pls = PLSRegression()

# Use GridSearchCV to find the best number of components using KFold cross-validation
param_grid = {"n_components": list(range(1, 21))}
kf = KFold(n_splits=5, shuffle=True, random_state=0)
grid_search = GridSearchCV(pls, param_grid, cv=kf, scoring="neg_mean_squared_error")
grid_search.fit(X_train, Y_train)

# Best number of components
best_n_components = grid_search.best_params_["n_components"]
print(f"Best number of components: {best_n_components}")

# Fit the PLS model with the best number of components
pls_best = PLSRegression(n_components=best_n_components)
pls_best.fit(X_train, Y_train)

# Predict on the test set
Y_pred = pls_best.predict(X_test)

# Inverse transform the predictions and the test targets to original scale
Y_pred_original = Y_pred
Y_test_original = Y_test

# Calculate mean squared error and root mean squared error for the predictions
mse = mean_squared_error(Y_test_original, Y_pred_original)
rmse = np.sqrt(mse)
# Calculate R-squared for the predictions
r2 = r2_score(Y_test_original, Y_pred_original)


# Print results
print(f"R-squared: {r2}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Visualise the results
if VISUALISE_RESULTS:
    number_of_samples = 4

    # Display the prediction results
    X_test_sample = X_test[:number_of_samples]
    Y_test_sample = Y_test_original[:number_of_samples]
    Y_pred_sample = Y_pred_original[:number_of_samples]

    print("\nSample Test Data (X):")
    print(X_test_sample)
    print("\nActual Concentrations (Y Test):")
    print(Y_test_sample)
    print("\nPredicted Concentrations (Y Pred):")
    print(Y_pred_sample)

    components = ["ATP", "UTP", "CTP", "GTP"]
    # x_labels = ["Actual", "Predicted"]
    colors = ["b", "g", "r", "c"]

    fig, axs = plt.subplots(4, 1, figsize=(14, 20))
    fig.suptitle("Actual vs Predicted Concentrations for 4 Samples", fontsize=16)

    for i in range(number_of_samples):
        bar_width = 0.4
        indices = np.arange(len(components))
        actual_bars = axs[i].bar(
            indices, Y_test_sample[i], bar_width, label="Actual", color=colors
        )
        predicted_bars = axs[i].bar(
            indices + bar_width,
            Y_pred_sample[i],
            bar_width,
            label="Predicted",
            color=colors,
            alpha=0.6,
        )

        axs[i].set_title(f"Sample {i}")
        axs[i].set_xticks(indices + bar_width / 2)
        axs[i].set_xticklabels(components)
        axs[i].legend()

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()

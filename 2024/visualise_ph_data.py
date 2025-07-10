from r3.experiments.ntp_prediction_from_ph_time import train_model
from r3.utils import load_data
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

training_data = load_data("step_response_data.xlsx", file_type="excel")
test_data = load_data("pH data set.xlsx", file_type="excel")

test_data["ntp"] = test_data["ntp"] / test_data["ntp"].max() * 0.5

visualise_training_data = training_data.drop(columns=["RNA"])
visualise_training_data.plot(
    x="Time",
    figsize=(10, 6),
    layout=(2, 1),
    subplots=True,
    title="Time vs pH, NTPs, (Training Data)",
)
visualise_test_data = test_data.drop(columns=["rna"])
visualise_test_data = visualise_test_data.loc[
    :, ~visualise_test_data.columns.str.contains("^Unnamed")
]
visualise_test_data.plot(
    x="Time",
    figsize=(10, 6),
    layout=(2, 1),
    subplots=True,
    title="Time vs pH, NTPs, (Test Data)",
)
plt.show()

gp_kernel = C(1.0, (1e-2, 1e2)) + RBF(1.0, (1e-2, 1e5))
gp = GaussianProcessRegressor(kernel=gp_kernel, n_restarts_optimizer=10, alpha=1e-2)

linear_model = LinearRegression()


gp.fit(training_data[["Time", "pH"]], training_data["NTPs"])
linear_model.fit(training_data[["Time", "pH"]], training_data["NTPs"])
gp_custom = train_model(
    training_data[["Time", "pH"]].values, training_data["NTPs"].values
)


plt.scatter(training_data["Time"], training_data["NTPs"], color="black", label="Data")
plt.plot(
    training_data["Time"],
    gp.predict(training_data[["Time", "pH"]]),
    color="red",
    label="RBF Gaussian Process (R^2={})".format(
        round(gp.score(test_data[["Time", "pH"]], test_data["ntp"]), 2)
    ),
)
plt.plot(
    training_data["Time"],
    gp_custom.predict(training_data[["Time", "pH"]]),
    color="green",
    label="Custom Gaussian Process (R^2={})".format(
        round(gp_custom.score(test_data[["Time", "pH"]], test_data["ntp"]), 2)
    ),
)
plt.plot(
    training_data["Time"],
    linear_model.predict(training_data[["Time", "pH"]]),
    color="blue",
    label="Linear Regression (R^2={})".format(
        round(linear_model.score(test_data[["Time", "pH"]], test_data["ntp"]), 2)
    ),
)
plt.xlabel("Time")
plt.ylabel("NTPs")
plt.title("NTPs vs Time")
plt.legend()
plt.show()

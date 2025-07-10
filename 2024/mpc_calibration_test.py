import typing
import numpy as np
import pandas as pd
from r3.scripts.ivt_mpc_simulation import mpc_controller

results : typing.Dict[str, typing.List[typing.Any]] = {
    "w1": [],
    "w2": [],
    "w3": [],
    "RNA_diff": [],
    "PPi_diff": [],
}

for i in np.linspace(0, 1, 5):
    for j in np.linspace(0, 1, 5):
        for k in np.linspace(0, 1, 5):
            set_point = [0.0002, 0.0001]
            control_horizon = 3
            initial_guess = [1.0]
            flow_sheet = mpc_controller(
                y_init=[0, 0],
                n_steps=4,
                control_horizon=control_horizon,
                y_set_point=set_point,
                time_interval=2,
                u_init=initial_guess,
                w=[i, j, k],
            )
            rna = flow_sheet["RNA"][-1]
            ppi = flow_sheet["PPi"][-1]
            # calculate the difference between the rna and set-point
            rna_diff = np.abs(rna - set_point[0])
            # calculate the difference between the ppi and set-point
            ppi_diff = np.abs(ppi - set_point[1])

            # append the results to the dictionary
            results["w1"].append(i)
            results["w2"].append(j)
            results["w3"].append(k)
            results["RNA_diff"].append(rna_diff)
            results["PPi_diff"].append(ppi_diff)


            # make a 4D plot of the results where the x-axis is w1, y-axis is w2, z-axis is w3, and the color is the RNA_diff and the size is the PPi_diff
            df = pd.DataFrame(results)
            df.to_csv("mpc_calibration_test.csv", index=False)

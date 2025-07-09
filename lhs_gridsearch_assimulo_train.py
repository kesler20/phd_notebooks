import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import r3.adapters as adapters
from r3.models.assimulo.assimulo_model_v2 import run_ivt_model
import r3.schema as schema
from concurrent.futures import ProcessPoolExecutor, as_completed


def evaluate_params(kapp, K1, K2, X_train: pd.DataFrame, y_train: pd.DataFrame):
    predictions = []
    for _, row in X_train.iterrows():
        result: pd.DataFrame = run_ivt_model(
            kapp=kapp,
            K1=K1,
            K2=K2,
            ATP_init=row[schema.IVTReactionSchema.NTP_M.value] / 4,
            CTP_init=row[schema.IVTReactionSchema.NTP_M.value] / 4,
            GTP_init=row[schema.IVTReactionSchema.NTP_M.value] / 4,
            UTP_init=row[schema.IVTReactionSchema.NTP_M.value] / 4,
            T7tot_init=row[schema.IVTReactionSchema.T7RNAP_u_uL.value],
            DNA_init=row[schema.IVTReactionSchema.DNA_ug_mL.value],
            SPDtot_init=row[schema.IVTReactionSchema.Spd_M.value],
            Mgtot_init=row[schema.IVTReactionSchema.Mg2_M.value],
            t_final=row[schema.IVTReactionSchema.TIME_min.value],
        )
        predictions.append(result[schema.IVTReactionSchema.RNA_g_L.value].values[-1])
    score = r2_score(
        y_train[schema.IVTReactionSchema.RNA_g_L.value],
        predictions,
    )
    return (kapp, K1, K2, score)


def main():
    X_train, y_train = adapters.DataPipelineAdapter("csp_lhs").get(
        X_columns=[
            schema.IVTReactionSchema.NTP_M.value,
            schema.IVTReactionSchema.T7RNAP_u_uL.value,
            schema.IVTReactionSchema.DNA_ug_mL.value,
            schema.IVTReactionSchema.Mg2_M.value,
            schema.IVTReactionSchema.Spd_M.value,
            schema.IVTReactionSchema.TIME_min.value,
        ],
        y_column=[schema.IVTReactionSchema.RNA_g_L.value],
    )

    param_grid = [
        (kapp, K1, K2)
        for kapp in np.linspace(1, 10_000, 100)
        for K1 in np.linspace(1, 10_000, 100)
        for K2 in np.linspace(1, 10_000, 100)
    ]

    best_params = (None, None, None)
    best_score = -np.inf

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(evaluate_params, kapp, K1, K2, X_train, y_train)
            for kapp, K1, K2 in param_grid
        ]
        for future in as_completed(futures):
            kapp, K1, K2, score = future.result()
            if score > best_score:
                best_score = score
                best_params = (kapp, K1, K2)

                # Save best parameters to a file
                with open("best_params.txt", "w") as f:
                    f.write(f"Kapp: {kapp}, K1: {K1}, K2: {K2}, R2 Score: {score}\n")

    print(f"Best K1: {best_params[0]}, Best K2: {best_params[1]}")
    print(f"Best Kapp: {best_params[2]}")
    print(f"Best R2 Score: {best_score}")


if __name__ == "__main__":
    main()

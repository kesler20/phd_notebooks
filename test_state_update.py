"""
Testing a simple hypothesis, if reupdating the parameters of the model should suffice
to approach the system dynamics, then the model should be able to predict
the RNA yield of the IVT reaction.
"""

import pandas as pd
import r3.models.assimulo.assimulo_model_v2 as ivt_model
import r3.schema as schema
from matplotlib import pyplot as plt
import numpy as np

true_system_dynamics = ivt_model.run_ivt_model()

true_system_dynamics = true_system_dynamics.drop(
    columns=[
        schema.IVTReactionSchema.RNA_M.value,
        # schema.IVTReactionSchema.PPi_M.value,
        # schema.IVTReactionSchema.Pi_M.value,
        schema.IVTReactionSchema.ATP_M.value,
        schema.IVTReactionSchema.CTP_M.value,
        schema.IVTReactionSchema.UTP_M.value,
        schema.IVTReactionSchema.GTP_M.value,
        # schema.IVTReactionSchema.Mg2_M.value,
        # schema.IVTReactionSchema.DNA_ug_mL.value,
        # schema.IVTReactionSchema.T7RNAP_u_uL.value,
        # schema.IVTReactionSchema.pH.value,
    ]
)


def predict_rna_yield(experimental_conditions: pd.DataFrame) -> pd.DataFrame:
    predictions = []
    state: dict[str, float] = dict()
    for _, row in experimental_conditions.iterrows():

        # replace zeros in the row with 1e-12
        row = row.replace(0, 1e-12).to_dict()

        print(row)

        result: pd.DataFrame = ivt_model.run_ivt_model(
            RNA_init=row.get(
                schema.IVTReactionSchema.RNA_M.value,
                state.get(schema.IVTReactionSchema.RNA_M.value, None),
            ),
            PPi_init=row.get(
                schema.IVTReactionSchema.PPi_M.value,
                state.get(schema.IVTReactionSchema.RNA_M.value, None),
            ),
            Pi_init=row.get(
                schema.IVTReactionSchema.Pi_M.value,
                state.get(schema.IVTReactionSchema.RNA_M.value, None),
            ),
            ATP_init=row.get(
                schema.IVTReactionSchema.ATP_M.value,
                state.get(schema.IVTReactionSchema.RNA_M.value, None),
            ),
            CTP_init=row.get(
                schema.IVTReactionSchema.CTP_M.value,
                state.get(schema.IVTReactionSchema.RNA_M.value, None),
            ),
            GTP_init=row.get(
                schema.IVTReactionSchema.GTP_M.value,
                state.get(schema.IVTReactionSchema.RNA_M.value, None),
            ),
            UTP_init=row.get(
                schema.IVTReactionSchema.UTP_M.value,
                state.get(schema.IVTReactionSchema.RNA_M.value, None),
            ),
            Mgtot_init=row.get(
                schema.IVTReactionSchema.Mg2_M.value,
                state.get(schema.IVTReactionSchema.RNA_M.value, None),
            ),
            DNA_init=row.get(
                schema.IVTReactionSchema.DNA_ug_mL.value,
                state.get(schema.IVTReactionSchema.RNA_M.value, None),
            ),
            T7tot_init=row.get(
                schema.IVTReactionSchema.T7RNAP_u_uL.value,
                state.get(schema.IVTReactionSchema.RNA_M.value, None),
            ),
            pH=row.get(
                schema.IVTReactionSchema.pH.value,
                state.get(schema.IVTReactionSchema.RNA_M.value, None),
            ),
            t_final=1,
        )

        for k in result.columns:
            state[k] = result[k].values[-1]

        predictions.append(result[schema.IVTReactionSchema.RNA_g_L.value].values[-1])

    return pd.DataFrame(
        predictions, columns=[schema.IVTReactionSchema.RNA_g_L.value + " (predicted)"]
    )


df = predict_rna_yield(true_system_dynamics)
df[schema.IVTReactionSchema.RNA_g_L.value + " (true)"] = true_system_dynamics[
    schema.IVTReactionSchema.RNA_g_L.value
]

df.plot()
plt.show()

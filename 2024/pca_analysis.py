import os
from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from r3.models.assimulo import assimulo_model_v1
from SALib.sample.latin import sample

data_set_folder = os.path.join(Path.home(), "protocol", "r3", "src", "r3", "data")
df = pd.read_excel(os.path.join(data_set_folder, "csp LHS.xlsx"))


def generate_lhd_samples(factors, bounds, model, num_initial_samples=10):
    num_factors = len(factors)

    # Step 1: Initial LHS design using SALib
    problem = {
        "num_vars": num_factors,
        "names": factors,
        "bounds": bounds,
    }
    lhs_design = sample(problem, num_initial_samples)
    return np.array([model(point) for point in lhs_design]), lhs_design


def get_pca_model(n_components=3):
    pca = PCA(n_components=n_components)
    principalComponents = pca.fit_transform(df)
    principalDf = pd.DataFrame(
        data=principalComponents,
        columns=list(
            map(lambda x: f"principal component {x}", range(1, n_components + 1))
        ),
    )
    return pca, principalDf


# score plot
def score_plot(response_variable, second_response_variable=None):
    principalDf = get_pca_model()[1]
    plt.figure()
    # Add a colour gradient based on the response variable
    scatter = plt.scatter(
        principalDf["principal component 1"],
        principalDf["principal component 2"],
        c=df[response_variable],
        cmap="viridis",
        s=df[second_response_variable] * 100 if second_response_variable else 50,
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(scatter, label=response_variable)
    plt.title("PCA (Simulated Data)")
    plt.show()


def loadings_plot():
    pca = get_pca_model()[0]
    # Extract loadings
    loadings = pca.components_.T
    # Plot loadings
    features = df.columns
    num_features = len(features)
    x = np.arange(num_features)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot for the first principal component
    ax.bar(x - 0.2, loadings[:, 0], width=0.4, label="PC1", color="b")

    # Plot for the second principal component
    ax.bar(x + 0.2, loadings[:, 1], width=0.4, label="PC2", color="r")

    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Features")
    ax.set_ylabel("Loadings")
    ax.set_title("PCA Loadings Plot (Simulated Data)")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()


def explained_variance():
    pca = get_pca_model()[0]
    # Extract explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Plotting
    components = np.arange(1, len(explained_variance) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plot for individual variance
    ax1.bar(
        components,
        explained_variance,
        alpha=0.6,
        label="Individual Variance",
        color="b",
    )
    # Line plot for cumulative variance
    ax2 = ax1.twinx()
    ax2.plot(
        components,
        cumulative_variance,
        marker="o",
        color="r",
        label="Cumulative Variance",
    )
    # Add 95% dotted line
    ax2.axhline(y=0.95, color="g", linestyle="--", label="95% Threshold")

    # Labels and title
    ax1.set_xlabel("Principal Components")
    ax1.set_ylabel("Individual Variance Explained", color="b")
    ax2.set_ylabel("Cumulative Variance Explained", color="r")
    ax1.set_title("PCA Variance Explained (Simulated Data)")

    # Legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.show()


def elbow_analysis():
    # Initialize variables
    inertia = []
    k_values = range(1, 11)

    # Iterate over a range of k values
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertia, marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Selecting k (Simulated Data)")
    plt.axvline(x=3, color="r", linestyle="--", label="Elbow Point")
    plt.legend()
    plt.show()


def simulation(point):
    NTP, pH, T7RNAP, Mg, DNA = point
    return assimulo_model_v1.run_ivt_model(
        NTP=NTP,
        pH=pH,
        T7RNAP=T7RNAP,
        Mg=Mg,
        DNA=DNA,
    )[
        "RNA"
    ].values[-1]


def analysis_of_variance(df, factors, response_variable):
    from statsmodels.formula.api import ols
    import plotly.io as pio
    from statsmodels.stats.anova import anova_lm
    import plotly.graph_objects as go

    anova_data = pd.DataFrame(
        data={
            "x1": df[factors[0]],
            "y": df[response_variable],
            "x2": df[factors[1]],
            "x3": df[factors[2]],
        }
    )

    lm = ols(
        formula="y ~ x1 + x2 + x3 \
    + x1:x2 + x1:x3 + x2:x3 \
    + x1:x2:x3",
        data=anova_data,
    ).fit()
    print(lm.summary())

    ANOVA = anova_lm(lm)
    A_f = ANOVA.iloc[:, 4]

    fig = go.Figure(
        data=[
            go.Bar(
                y=["x1", "x2", "x3", "x1:x2", "x1:x3", "x2:x3", "x1:x2:x3"],
                x=A_f,
                orientation="h",
                text=["x1", "x2", "x3", "x1:x2", "x1:x3", "x2:x3", "x1:x2:x3"],
            )
        ]
    )
    # Add annotations to explain what the variables mean
    annotations = [
        dict(
            x=0.5,
            y=1.1,
            xref="paper",
            yref="paper",
            text=f"x1: {factors[0]}",
            showarrow=False,
        ),
        dict(
            x=0.5,
            y=1.05,
            xref="paper",
            yref="paper",
            text=f"x2: {factors[1]}",
            showarrow=False,
        ),
        dict(
            x=0.5,
            y=1.0,
            xref="paper",
            yref="paper",
            text=f"x3: {factors[2]}",
            showarrow=False,
        ),
    ]

    fig.update_layout(annotations=annotations)
    fig.add_vline(0.05, line_width=3, line_dash="dash", line_color="red")
    fig.update_xaxes(title="x", visible=False, showticklabels=False, range=[0, 1])
    fig.update_yaxes(title="y", visible=False, showticklabels=False)
    fig.update_layout(barmode="stack", yaxis={"categoryorder": "total ascending"})
    fig.update_layout(title_text=response_variable + "(Simulated Data)")
    fig.show()


if __name__ == "__main__":
    columns = ["NTP (M)", "pH", "T7RNAP (unit uL-1)", "Mg (M)", "DNA (pM)"]
    initial_bounds = [
        [4 * 0.01, 60 * 0.01],
        [6, 8],
        [(100**0.6) * 1e-9, (400**0.6) * 1e-9],
        [5 * 0.01, 90 * 0.01],
        [80, 100],
    ]

    # score_plot("Yield [g/L]-gel")
    response, design = generate_lhd_samples(
        columns, initial_bounds, simulation, num_initial_samples=60
    )
    print(response)
    print(design)

    simulated_df = pd.DataFrame(design, columns=columns)
    simulated_df["Yield [g/L]-gel"] = response

    analysis_of_variance(
        simulated_df,
        ["NTP (M)", "T7RNAP (unit uL-1)", "Mg (M)"],
        "Yield [g/L]-gel",
    )

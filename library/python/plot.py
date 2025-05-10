import numpy as np
import matplotlib.pyplot as plt
import arviz as az


def plot_stan_data(data_dict, fig_size=(15, 10)):
    """
    Create visualizations of the synthetic data generated for Stan models

    Parameters:
    - data_dict: Dictionary containing N, K, x, y, and true_params
    - fig_size: Figure size tuple
    """
    N = data_dict["N"]
    K = data_dict["K"]
    x = data_dict["x"]
    y = data_dict["y"]
    true_params = data_dict["true_params"]

    # Create figure with subplots
    fig = plt.figure(figsize=fig_size)

    # Scatter plots of each covariate vs response with true relationship
    for i in range(min(K, 5)):  # Show at most 5 covariates to avoid too many plots
        ax = fig.add_subplot(2, 3, i + 1)

        # Scatter plot for this covariate
        ax.scatter(x[:, i], y, alpha=0.5, label="Data points")

        # Calculate and plot the true relationship for this covariate
        x_range = np.linspace(min(x[:, i]), max(x[:, i]), 100)
        y_pred = true_params["alpha"] + true_params["beta"][i] * x_range
        ax.plot(x_range, y_pred, "r-", label=f"β{i}={true_params['beta'][i]:.2f}")

        ax.set_xlabel(f"x{i}")
        ax.set_ylabel("y")
        ax.set_title(f"Covariate {i} vs Response")
        ax.legend()

    # Distribution of response variable
    ax_hist = fig.add_subplot(2, 3, 6)
    ax_hist.hist(y, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    ax_hist.axvline(
        np.mean(y), color="r", linestyle="--", label=f"Mean: {np.mean(y):.2f}"
    )
    ax_hist.set_xlabel("y")
    ax_hist.set_ylabel("Frequency")
    ax_hist.set_title("Response Variable Distribution")
    ax_hist.legend()

    # Add overall title
    plt.suptitle(
        f"Synthetic Data for Linear Regression (N={N}, K={K})\n"
        + f"True intercept={true_params['alpha']:.2f}, σ={true_params['sigma']:.2f}",
        fontsize=16,
        y=1.02,
    )

    plt.tight_layout()
    plt.show()

    # Create a correlation heatmap of covariates
    plt.figure(figsize=(10, 8))
    corr_matrix = np.corrcoef(x.T)
    plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.title("Covariate Correlation Matrix")

    # Add correlation values
    for i in range(K):
        for j in range(K):
            plt.text(
                j,
                i,
                f"{corr_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if abs(corr_matrix[i, j]) < 0.7 else "white",
            )

    plt.xticks(range(K), [f"x{i}" for i in range(K)])
    plt.yticks(range(K), [f"x{i}" for i in range(K)])
    plt.tight_layout()
    plt.show()


def plot_trace_plots(results_dict, var_names=None, fig_size=(12, 8)):
    """
    Plot trace plots for specified variables in all models.

    Parameters:
        results_dict: Dictionary of model names and ArviZ InferenceData objects
        var_names: Variables to plot (None for all variables)
        fig_size: Figure size as (width, height)
    """
    for model_name, idata in results_dict.items():
        if idata is None:
            continue

        plt.figure(figsize=fig_size)
        az.plot_trace(idata, var_names=var_names, compact=True)
        plt.suptitle(f"Trace Plot: {model_name}", fontsize=14)
        plt.tight_layout()
        plt.show()

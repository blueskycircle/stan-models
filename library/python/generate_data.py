import numpy as np


def generate_linear_regression_data(
    N=100, K=3, sigma_true=2.0, intercept_true=5.0, seed=123
):
    """Generates synthetic data for Gaussian linear regression."""
    np.random.seed(seed)

    # True coefficients
    beta_true = np.random.randn(K) * 2.5

    # Generate predictor matrix X
    x_matrix = np.random.randn(N, K)

    # Generate outcome vector y
    mu = intercept_true + x_matrix @ beta_true
    y_vector = mu + np.random.normal(0, sigma_true, N)

    print("--- True Parameters for Data Generation ---")
    print(f"True Intercept (alpha): {intercept_true:.2f}")
    print(f"True Coefficients (beta): {np.round(beta_true, 2)}")
    print(f"True Error Std Dev (sigma): {sigma_true:.2f}")
    print("----------------------------------------\n")

    return {
        "N": N,
        "K": K,
        "x": x_matrix,
        "y": y_vector,
        "true_params": {
            "alpha": intercept_true,
            "beta": beta_true,
            "sigma": sigma_true,
        },
    }

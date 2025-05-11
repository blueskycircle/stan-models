import numpy as np


def generate_linear_regression_data(
    N=100,
    K=3,
    sigma_true=2.0,
    intercept_true=5.0,
    seed=123,
    target_correlation=0.0,
):
    """
    Generates synthetic data for Gaussian linear regression.

    Args:
        N (int): Number of data points.
        K (int): Number of predictors.
        sigma_true (float): True standard deviation of the error term.
        intercept_true (float): True intercept value.
        seed (int): Random seed for reproducibility.
        target_correlation (float): The target pairwise correlation between predictors.
                                    Must be between -1/(K-1) and 1.0.
                                    If 0, predictors are uncorrelated.
    Returns:
        dict: A dictionary containing the generated data and true parameters.
    """
    np.random.seed(seed)

    # True coefficients
    beta_true = np.random.randn(K) * 2.5

    # Generate predictor matrix X
    if K == 1 or target_correlation == 0.0:
        # Uncorrelated predictors
        x_matrix = np.random.randn(N, K)
    else:
        # Correlated predictors
        # Check if target_correlation is valid for a positive semi-definite matrix
        lower_bound = -1.0 / (K - 1) if K > 1 else 0.0
        if not (lower_bound <= target_correlation <= 1.0):
            raise ValueError(
                f"target_correlation must be between {lower_bound:.3f} and 1.0 for K={K}. "
                f"Received: {target_correlation}"
            )

        mean = np.zeros(K)
        # Create the covariance matrix (here, a correlation matrix as diagonal is 1)
        cov_matrix = np.full((K, K), target_correlation)
        np.fill_diagonal(cov_matrix, 1.0)

        # Generate correlated predictor matrix X
        # Each row is a sample from the multivariate normal distribution
        x_matrix = np.random.multivariate_normal(mean, cov_matrix, N)

    # Generate outcome vector y
    mu = intercept_true + x_matrix @ beta_true
    y_vector = mu + np.random.normal(0, sigma_true, N)

    print("--- True Parameters for Data Generation ---")
    print(f"True Intercept (alpha): {intercept_true:.2f}")
    print(f"True Coefficients (beta): {np.round(beta_true, 2)}")
    print(f"True Error Std Dev (sigma): {sigma_true:.2f}")
    if K > 1 and target_correlation != 0.0:
        sample_corr_matrix = np.corrcoef(x_matrix, rowvar=False)
        if K > 2:
            off_diag_indices = np.triu_indices(K, k=1)
            avg_off_diag_corr = np.mean(sample_corr_matrix[off_diag_indices])
            print(f"Target Predictor Correlation: {target_correlation:.2f}")
            print(
                f"Average Sample Off-Diagonal Predictor Correlation: {avg_off_diag_corr:.2f}"
            )
        elif K == 2:
            print(f"Target Predictor Correlation: {target_correlation:.2f}")
            print(f"Sample Predictor Correlation: {sample_corr_matrix[0,1]:.2f}")

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

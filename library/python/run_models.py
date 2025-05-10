from cmdstanpy import CmdStanModel
import arviz as az


def run_stan_model_cmdstanpy(model_name, model_code_path, stan_data, seed=123):
    """Loads, compiles, runs a Stan model using CmdStanPy, and prints summary."""
    print(f"\n--- Running Model: {model_name} ---")

    try:
        # Compile the model (only compiles if needed)
        model = CmdStanModel(stan_file=model_code_path)
    except Exception as e:
        print(f"Error compiling {model_name}: {e}")
        return None

    print(f"Sampling for {model_name}...")
    try:
        # Sample from the model
        fit = model.sample(
            data=stan_data, seed=seed, chains=4, iter_sampling=1000, iter_warmup=500
        )
    except Exception as e:
        print(f"Error sampling for {model_name}: {e}")
        return None

    print("Sampling complete.")

    # Convert to ArviZ InferenceData for analysis
    try:
        idata = az.from_cmdstanpy(fit)
    except (TypeError, AttributeError):
        print("Warning: Using alternative conversion method for ArviZ")

    print(f"\nSummary for {model_name}:")

    var_names_to_summarize = ["alpha", "sigma"]

    # Get parameter names from fit
    try:
        if hasattr(fit, "stan_variables"):
            if callable(fit.stan_variables):
                stan_vars = fit.stan_variables().keys()
            else:
                stan_vars = fit.stan_variables.keys()
        elif hasattr(fit, "column_names"):
            stan_vars = [name.split(".")[0] for name in fit.column_names]
            stan_vars = list(set(stan_vars))
        else:
            stan_vars = list(idata.posterior.data_vars.keys())
    except Exception as e:
        print(f"Warning: Error accessing parameter names: {e}")
        stan_vars = []

    if "beta" in stan_vars:  # Basic model
        var_names_to_summarize.append("beta")
    elif "theta" in stan_vars and "beta" in stan_vars:  # QR model
        var_names_to_summarize.append("beta")

    summary = az.summary(idata, var_names=var_names_to_summarize)
    print(summary)

    return idata

# =============================================================================
# ===== MV Hondius Andes Virus Outbreak Model: Calibration ====================
# =============================================================================

# -----------------------------------------------------------------------------
# ----- 0. Initialisation -----------------------------------------------------
# -----------------------------------------------------------------------------

# ----- 0.1. Description ------------------------------------------------------

# The following provides a single job instance for the MV Hondius calibration
# model.

# ----- 0.2. Dependencies -----------------------------------------------------

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model_functions import (
    read_parameters,
    read_manifest,
    read_cases,
    create_valid_manifest_matrices,
    initialise_MV_model,
    run_MV_model,
    pseudo_likelihood
)


# -----------------------------------------------------------------------------
# ----- 1. User Settings ------------------------------------------------------
# -----------------------------------------------------------------------------

N_PARAMETER_SETS = 2
N_SIMULATIONS = 25000
N_PERMUTATIONS = 5000
OBSERVATION_CUTOFF_DAY = 54

# Beta priors. Beta(1, 1) is uniform on [0, 1].
P_TRANS_ALPHA = 1
P_TRANS_BETA = 1

E_CONF_ALPHA = 1
E_CONF_BETA = 1

MAX_CASE01_CONTACTS = 120
MAX_POPULATION_CONTACTS = 120

# -----------------------------------------------------------------------------
# ----- 2. Run One Parameter Set ----------------------------------------------
# -----------------------------------------------------------------------------

def run_parameter_set(pars, manifest_matrices, model_inputs, P_trans, C_Case01, e_conf,
                      max_population_contacts):

    exact_likelihoods = np.zeros(N_SIMULATIONS)
    minimum_likelihoods = np.zeros(N_SIMULATIONS)
    onward_infections = np.zeros(N_SIMULATIONS, dtype=np.int64)

    for simulation_i in range(N_SIMULATIONS):

        agents = run_MV_model(
            pars=pars,
            manifest_matrices=manifest_matrices,
            model_inputs=model_inputs,
            P_trans=P_trans,
            C_Case01=C_Case01,
            e_conf=e_conf,
            max_population_contacts=max_population_contacts
        )

        exact_likelihood, minimum_likelihood = pseudo_likelihood(
            agents,
            model_inputs,
            observation_cutoff_day=OBSERVATION_CUTOFF_DAY,
            n_permutations=N_PERMUTATIONS
        )

        exact_likelihoods[simulation_i] = exact_likelihood
        minimum_likelihoods[simulation_i] = minimum_likelihood
        onward_infections[simulation_i] = np.sum(agents[1, :] > -1) - 1

    output = {
        "P_trans": P_trans,
        "C_Case01": C_Case01,
        "e_conf": e_conf,
        "likelihood_exact": exact_likelihoods.mean(),
        "likelihood_minimum": minimum_likelihoods.mean(),
        "acceptance_rate_exact": np.mean(onward_infections == 11)
    }

    return output


# -----------------------------------------------------------------------------
# ----- 3. Main ---------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():

    if len(sys.argv) < 2:
        raise ValueError("Please provide a job number, e.g. python run_mv_job.py 001")

    job_number = sys.argv[1]

    if len(sys.argv) >= 3:
        data_path = Path(sys.argv[2])
    else:
        data_path = Path("data")

    if len(sys.argv) >= 4:
        output_path = Path(sys.argv[3])
    else:
        output_path = data_path / "job_outputs"

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Job number: {job_number}")
    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")

    pars = read_parameters(data_path / "parameters.yaml", "MV")
    manifest = read_manifest(data_path / "manifest.csv")
    case_01_dates, cases = read_cases(data_path / "case_dates.csv")

    success, manifest_matrices = create_valid_manifest_matrices(pars, manifest)

    if not success:
        raise RuntimeError("Could not create valid manifest matrices.")

    model_inputs = initialise_MV_model(
        pars,
        manifest,
        case_01_dates,
        cases
    )

    # Numba warm-up with one small call.
    warmup_agents = run_MV_model(
        pars=pars,
        manifest_matrices=manifest_matrices,
        model_inputs=model_inputs,
        P_trans=0.01,
        C_Case01=15,
        e_conf=0.5,
        max_population_contacts=MAX_POPULATION_CONTACTS
    )

    pseudo_likelihood(
        warmup_agents,
        model_inputs,
        observation_cutoff_day=OBSERVATION_CUTOFF_DAY,
        n_permutations=10
    )

    results = []

    start_all = time.perf_counter()

    for parameter_i in range(N_PARAMETER_SETS):

        P_trans = np.random.beta(P_TRANS_ALPHA, P_TRANS_BETA)
        e_conf = np.random.beta(E_CONF_ALPHA, E_CONF_BETA)
        C_Case01 = np.random.uniform(0, MAX_CASE01_CONTACTS)

        print()
        print(f"Parameter set {parameter_i + 1} / {N_PARAMETER_SETS}")
        print(f"P_trans={P_trans:.6f}, C_Case01={C_Case01:.6f}, e_conf={e_conf:.6f}")

        start_parameter = time.perf_counter()

        result = run_parameter_set(
            pars=pars,
            manifest_matrices=manifest_matrices,
            model_inputs=model_inputs,
            P_trans=P_trans,
            C_Case01=C_Case01,
            e_conf=e_conf,
            max_population_contacts=MAX_POPULATION_CONTACTS
        )

        results.append(result)

        end_parameter = time.perf_counter()

        print(f"Completed in {(end_parameter - start_parameter) / 60:.2f} minutes")

    end_all = time.perf_counter()

    results = pd.DataFrame(results)

    output_file = output_path / f"{job_number}_MV_parameter_results.csv"
    results.to_csv(output_file, index=False)

    print()
    print(f"Total runtime: {(end_all - start_all) / 60:.2f} minutes")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()
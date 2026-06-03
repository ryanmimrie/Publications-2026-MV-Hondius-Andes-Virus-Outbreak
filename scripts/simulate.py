# =============================================================================
# ===== MV Hondius Andes Virus Outbreak Model: Mainland Simulation ============
# =============================================================================

# -----------------------------------------------------------------------------
# ----- 0. Initialisation -----------------------------------------------------
# -----------------------------------------------------------------------------

# ----- 0.1. Description ------------------------------------------------------

# The following provides a single job instance for the ANDV mainland outbreak
# simulation.

# ----- 0.2. Dependencies -----------------------------------------------------

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from model_functions import (
    read_parameters,
    initialise_mainland_model,
    run_mainland_model
)


# -----------------------------------------------------------------------------
# ----- 1. User Settings ------------------------------------------------------
# -----------------------------------------------------------------------------

N_SIMULATIONS = 100000

# Index case age group:
# "Age_0_4", "Age_5_18", "Age_19_64", or "Age_65_plus"
INDEX_AGE_GROUP = "Age_19_64"

AGE_GROUPS = ["Age_0_4", "Age_5_18", "Age_19_64", "Age_65_plus"]

P_TRANS_ALPHA = 4.6423
P_TRANS_BETA = 603.6745

# -----------------------------------------------------------------------------
# ----- 2. Helper Functions ---------------------------------------------------
# -----------------------------------------------------------------------------

def make_start_vector(index_age_group):

    start = np.zeros(len(AGE_GROUPS), dtype=np.int64)
    start[AGE_GROUPS.index(index_age_group)] = 1

    return start


def summarise_mainland_result(result, initial_infections):

    exposed_total = result["E"].sum(axis=1)
    infectious_total = result["Pro"].sum(axis=1)
    hps_total = result["HPS"].sum(axis=1)

    prevalence = exposed_total + infectious_total + hps_total

    active_days = np.where(prevalence > 0)[0]

    if len(active_days) == 0:
        outbreak_duration = 0
    else:
        outbreak_duration = int(active_days.max())

    onward_unique_cases = int(result["S"][0].sum() - result["S"][-1].sum())
    total_unique_cases = initial_infections + onward_unique_cases

    output = {
        "total_unique_cases": total_unique_cases,
        "onward_unique_cases": onward_unique_cases,
        "initial_cases": initial_infections,
        "total_infections": total_unique_cases,
        "onward_infections": onward_unique_cases,
        "outbreak_duration": outbreak_duration,
        "peak_infectious": int(infectious_total.max()),
        "peak_prevalence": int(prevalence.max()),
        "cumulative_HPS_total": int(result["cumulative_HPS_total"])
    }

    cumulative_HPS_by_age = result["cumulative_HPS_by_age"].astype(int)

    for age_i, age_group in enumerate(AGE_GROUPS):
        output[f"cumulative_HPS_{age_group}"] = int(cumulative_HPS_by_age[age_i])
        output[f"final_S_{age_group}"] = int(result["S"][-1, age_i])
        output[f"final_R_{age_group}"] = int(result["R"][-1, age_i])

    return output


def extract_infectious_trajectory(result):

    infectious_total = result["E"].sum(axis=1).astype(int) + result["Pro"].sum(axis=1).astype(int) + result["HPS"].sum(axis=1).astype(int)

    output = {}

    for day, value in enumerate(infectious_total):
        output[f"{day:03d}"] = int(value)

    return output


# -----------------------------------------------------------------------------
# ----- 3. Main ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def main():

    if len(sys.argv) < 3:
        raise ValueError(
            "Usage: python scripts/run_mainland.py <job_number> <output_path> "
            "[data_path] [index_age_group] [n_simulations]"
        )

    job_number = sys.argv[1]
    output_path = Path(sys.argv[2])

    if len(sys.argv) >= 4:
        data_path = Path(sys.argv[3])
    else:
        data_path = Path("data")

    if len(sys.argv) >= 5:
        index_age_group = sys.argv[4]
    else:
        index_age_group = INDEX_AGE_GROUP

    if len(sys.argv) >= 6:
        n_simulations = int(sys.argv[5])
    else:
        n_simulations = N_SIMULATIONS

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Job number: {job_number}")
    print(f"Data path: {data_path}")
    print(f"Output path: {output_path}")
    print(f"Index age group: {index_age_group}")
    print(f"N simulations: {n_simulations}")

    pars = read_parameters(data_path / "parameters.yaml", "UK")
    start = make_start_vector(index_age_group)
    initial_infections = int(start.sum())

    summary_results = []
    trajectory_results = []

    start_all = time.perf_counter()

    for simulation_i in range(n_simulations):

        P_trans = np.random.beta(P_TRANS_ALPHA, P_TRANS_BETA)

        model = initialise_mainland_model(
            pars=pars,
            start=start,
            P_trans=P_trans
        )

        result = run_mainland_model(model, verbose=False)

        summary = summarise_mainland_result(
            result=result,
            initial_infections=initial_infections
        )

        summary.update({
            "job_number": job_number,
            "simulation": simulation_i,
            "index_age_group": index_age_group,
            "P_trans": P_trans,
            "P_trans_alpha": P_TRANS_ALPHA,
            "P_trans_beta": P_TRANS_BETA
        })

        trajectory = extract_infectious_trajectory(result)

        trajectory.update({
            "job_number": job_number,
            "simulation": simulation_i,
            "index_age_group": index_age_group,
            "P_trans": P_trans,
            "total_unique_cases": summary["total_unique_cases"],
            "onward_unique_cases": summary["onward_unique_cases"]
        })

        summary_results.append(summary)
        trajectory_results.append(trajectory)

        if (simulation_i + 1) % 1000 == 0:
            elapsed = time.perf_counter() - start_all
            print(
                f"Completed {simulation_i + 1} / {n_simulations} "
                f"simulations | elapsed={elapsed / 60:.2f} min"
            )

    summary_results = pd.DataFrame(summary_results)
    trajectory_results = pd.DataFrame(trajectory_results)

    summary_file = output_path / f"{job_number}_mainland_{index_age_group}_summary.csv"
    trajectory_file = output_path / f"{job_number}_mainland_{index_age_group}_infectious_trajectories.csv"

    summary_results.to_csv(summary_file, index=False)
    trajectory_results.to_csv(trajectory_file, index=False)

    end_all = time.perf_counter()

    print()
    print(f"Total runtime: {(end_all - start_all) / 60:.2f} minutes")
    print(f"Saved summary: {summary_file}")
    print(f"Saved trajectories: {trajectory_file}")


if __name__ == "__main__":
    main()
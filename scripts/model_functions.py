# =============================================================================
# ===== MV Hondius Andes Virus Outbreak Model: Model Functions ================
# =============================================================================

# -----------------------------------------------------------------------------
# ----- 0. Initialisation -----------------------------------------------------
# -----------------------------------------------------------------------------

# ----- 0.1. Description ------------------------------------------------------

# The below contains functions for running models used in the study: 
# (2026) Rapid modelling of the 2026 Andes virus outbreak predicts limited
# transmissibility and low mainland outbreak risk.

# ----- 0.2. Dependencies -----------------------------------------------------

import yaml
import math
import random
import numpy as np
import pandas as pd
from numba import njit


# -----------------------------------------------------------------------------
# ----- 1. Helper/Short Functions ---------------------------------------------
# -----------------------------------------------------------------------------

# ----- 1.1. Read Parameters --------------------------------------------------

def read_parameters(file, model):
    with open(file, "r") as file:
        all_parameters = yaml.safe_load(file)

    if model == "MV":
        keys_to_keep = ["model_MV", "cabins", "progression", "contacts_MV"]

    if model == "UK":
        keys_to_keep = ["model_UK", "progression", "contacts_UK", "UK_population"]

    parameters = {}

    for key in keys_to_keep:
        parameters[key] = all_parameters[key]

    return parameters

# ----- 1.2. Read Manifest ----------------------------------------------------

def read_manifest(file):
    manifest = pd.read_csv(file)
    manifest["Date"] = pd.to_datetime(manifest["Date"], format="%d/%m/%Y")
    manifest = manifest.drop(columns=["Notes"])

    manifest = manifest[
        (manifest["P_embark"] != 0) |
        (manifest["P_disembark"] != 0) |
        (manifest["C_embark"] != 0) |
        (manifest["C_disembark"] != 0)]
    
    manifest = manifest.iloc[:-1]

    return manifest

# ----- 1.3. Read Cases -------------------------------------------------------

def read_cases(file):
    cases = pd.read_csv(file)

    date_columns = ["Symptom_onset", "Hospitalisation", "Death", "Disembarked"]

    for column in date_columns:
        cases[column] = pd.to_datetime(cases[column], format="%d/%m/%Y")

    case_01 = cases[cases["Case"] == 1].iloc[0]

    case_01_dates = {"symptom_onset": case_01["Symptom_onset"].date(), "death": case_01["Death"].date()}

    remaining_cases = cases[cases["Case"] != 1].copy()

    remaining_cases["Removed"] = remaining_cases[["Hospitalisation", "Death", "Disembarked"]].min(axis=1)

    remaining_cases["Symptom_onset"] = remaining_cases["Symptom_onset"].dt.date
    remaining_cases["Removed"] = remaining_cases["Removed"].dt.date

    remaining_cases = remaining_cases[["Case", "Symptom_onset", "Removed"]]

    return case_01_dates, remaining_cases


# -----------------------------------------------------------------------------
# ----- 2. MV Hondius Contact Matrices ----------------------------------------
# -----------------------------------------------------------------------------

# ----- 2.1. Assign Cabins ----------------------------------------------------

def assign_cabins(pars, manifest):
        
    passenger_cabin_occupancies = ([2] * pars["cabins"]["passenger_cabins_twin"] +
                                   [3] * pars["cabins"]["passenger_cabins_triple"] +
                                   [4] * pars["cabins"]["passenger_cabins_quad"])
    
    crew_cabin_occupancies = ([1] * pars["cabins"]["crew_cabins_single"] +
                              [2] * pars["cabins"]["crew_cabins_twin"])
    
    unique_passengers = sum(manifest["P_embark"])
    unique_crew = sum(manifest["C_embark"])
    
    passenger_ids = [f"P{i:03d}" for i in range(3, unique_passengers + 1)]
    crew_ids = [f"C{i:03d}" for i in range(4, unique_crew + 1)]
    
    # Deterministic passenger cabin
    passenger_cabins = [["P001", "P002"]]
    passenger_cabin_occupancies = passenger_cabin_occupancies[1:]
    random.shuffle(passenger_cabin_occupancies)
    
    # Deterministic crew cabins
    crew_cabins = [["C001"], ["C002"], ["C003"]]
    crew_cabin_occupancies = crew_cabin_occupancies[3:]
    random.shuffle(crew_cabin_occupancies)

    passenger_berths = ([1] * len(passenger_ids) + [0] * (sum(passenger_cabin_occupancies) - len(passenger_ids)))
    crew_berths = ([1] * len(crew_ids) + [0] * (sum(crew_cabin_occupancies) - len(crew_ids)))

    random.shuffle(passenger_berths)
    random.shuffle(crew_berths)
    
    berth_start = 0
    id_start = 0
    
    for cabin_size in passenger_cabin_occupancies:
        berth_end = berth_start + cabin_size
        n_occupants = sum(passenger_berths[berth_start:berth_end])
    
        passenger_cabins.append(passenger_ids[id_start:id_start + n_occupants] + ["X"] * (cabin_size - n_occupants))
    
        berth_start = berth_end
        id_start += n_occupants
        
        
    berth_start = 0
    id_start = 0
    
    for cabin_size in crew_cabin_occupancies:
        berth_end = berth_start + cabin_size
        n_occupants = sum(crew_berths[berth_start:berth_end])
    
        crew_cabins.append(crew_ids[id_start:id_start + n_occupants] + ["X"] * (cabin_size - n_occupants))
    
        berth_start = berth_end
        id_start += n_occupants
    
    cabins = {"passenger" : passenger_cabins,
              "crew" : crew_cabins}
    
    return cabins

# ----- 2.2. Create Master Contact Matrix -------------------------------------

def create_master_contact_matrix(pars, cabins):

    all_cabins = cabins["passenger"] + cabins["crew"]

    individual_ids = []

    for cabin in all_cabins:
        for person_id in cabin:
            if person_id != "X":
                individual_ids.append(person_id)

    n_individuals = len(individual_ids)

    id_to_index = {}

    for i, person_id in enumerate(individual_ids):
        id_to_index[person_id] = i

    master_matrix = np.full(shape=(n_individuals, n_individuals), fill_value=2)

    # Case 01 has lower contact weight with others
    master_matrix[0, :] = 1
    master_matrix[:, 0] = 1

    # Within-cabin contacts have higher contact weight
    for cabin in all_cabins:
        occupants = []

        for person_id in cabin:
            if person_id != "X":
                occupants.append(person_id)

        occupant_indices = []

        for person_id in occupants:
            occupant_indices.append(id_to_index[person_id])

        master_matrix[np.ix_(occupant_indices, occupant_indices)] = 3

    # No self-contact
    np.fill_diagonal(master_matrix, 0)

    master_matrix = pd.DataFrame(master_matrix, index=individual_ids, columns=individual_ids)

    return master_matrix

# ----- 2.3. Attempt Manifest Contact Matrices --------------------------------

def attempt_manifest_contact_matrices(manifest, cabins, master_matrix):
        
    passenger_cabins = [[x for x in cabin if x != "X"] for cabin in cabins["passenger"]]
    passenger_cabins = [cabin for cabin in passenger_cabins if len(cabin) > 0]

    crew_list = [x for cabin in cabins["crew"] for x in cabin if x != "X"]
        
    do_not_disembark = [["P001", "P002"]]
    
    passengers_over_time = []
    crew_over_time = []
    
    current_passengers = []
    current_crew = []

    
    for index, row in manifest.iterrows():
    
        passengers_embarking = row["P_embark"]
        passengers_disembarking = row["P_disembark"]

        crew_embarking = row["C_embark"]
        crew_disembarking = row["C_disembark"]

        P = current_passengers
        C = current_crew
        
        while passengers_disembarking > 0:
            
            next_cabin = P.pop(0)
            
            if next_cabin in do_not_disembark:
                P.append(next_cabin)
            else:
                passengers_disembarking -= len(next_cabin)
            
        while crew_disembarking > 0:
            
            next_crew = C.pop(0)
            
            if next_crew in do_not_disembark:
                C.append(next_crew)
            else:
                crew_disembarking -= 1
        
        while passengers_embarking > 0:
                
            next_cabin = passenger_cabins.pop(0)
            passengers_embarking -= len(next_cabin)
            P.append(next_cabin)
                
        while crew_embarking > 0:
                
            next_crew = crew_list.pop(0)
            crew_embarking -= 1
            C.append(next_crew)
            
        if passengers_disembarking < 0 or crew_disembarking < 0 or passengers_embarking < 0 or crew_embarking < 0:
            return False, []
        else:
            passengers_over_time.append([cabin.copy() for cabin in P])
            crew_over_time.append(C.copy())
            random.shuffle(P)
            random.shuffle(C)
            current_passengers = P
            current_crew = C
        
    matrices = []

    for P, C in zip(passengers_over_time, crew_over_time):

        onboard_passengers = [x for cabin in P for x in cabin]
        onboard_people = onboard_passengers + C

        absent_people = [x for x in master_matrix.index if x not in onboard_people]

        matrix = master_matrix.copy()
        matrix.loc[absent_people, :] = 0
        matrix.loc[:, absent_people] = 0

        matrices.append(matrix)
        
    return True, matrices

# ----- 2.4. Create Valid Manifest Matrices -----------------------------------

def create_valid_manifest_matrices(pars, manifest, max_attempts=10000):

    for attempt in range(1, max_attempts + 1):

        cabins = assign_cabins(pars, manifest)
        master_matrix = create_master_contact_matrix(pars, cabins)

        try:
            success, manifest_matrices = attempt_manifest_contact_matrices(manifest, cabins, master_matrix)
        except (IndexError, KeyError, ValueError):
            success = False
            manifest_matrices = []

        if success:
            return True, manifest_matrices

    return False, {}

# ----- 2.5. Plot Matrix ------------------------------------------------------

def plot_matrix(matrix, figsize=(12, 12), title="Contact matrix"):

    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    arr = matrix.to_numpy() if hasattr(matrix, "to_numpy") else np.asarray(matrix)

    masked_arr = np.ma.masked_where(arr == 0, arr)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("black")

    heatmap = ax.imshow(masked_arr, cmap=cmap)

    colourbar = fig.colorbar(heatmap, ax=ax)
    colourbar.set_label("Daily Contact Events", fontsize=22)
    colourbar.ax.tick_params(labelsize=18)

    ax.set_title(title, fontsize=26, pad=20)

    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.show()

# ----- 2.6. Find symmetrical matrix ------------------------------------------

@njit
def find_symmetrical_matrix(eligibility_matrix, target_unique_contacts,
                                 max_iterations=1000, tolerance=1e-1,
                                 bisection_steps=10):

    n_individuals = eligibility_matrix.shape[0]
    scaling_factors = np.ones(n_individuals)

    targets = target_unique_contacts.copy()

    for i in range(n_individuals):

        possible_contacts = 0.0

        for j in range(n_individuals):
            possible_contacts += eligibility_matrix[i, j]

        if possible_contacts <= 0 or targets[i] <= 0:
            targets[i] = 0.0
            scaling_factors[i] = 0.0

        elif targets[i] >= possible_contacts:
            targets[i] = possible_contacts * (1 - 1e-12)

    probability_matrix = np.zeros((n_individuals, n_individuals))

    for iteration in range(max_iterations):

        for i in range(n_individuals):

            target = targets[i]

            if target == 0:
                scaling_factors[i] = 0.0

            else:
                lower = 0.0
                upper = 1.0

                row_total = 0.0

                for j in range(n_individuals):
                    partner_value = eligibility_matrix[i, j] * scaling_factors[j]
                    value = upper * partner_value
                    row_total += value / (1 + value)

                while row_total < target and upper < 1e12:

                    upper = upper * 2
                    row_total = 0.0

                    for j in range(n_individuals):
                        partner_value = eligibility_matrix[i, j] * scaling_factors[j]
                        value = upper * partner_value
                        row_total += value / (1 + value)

                for step in range(bisection_steps):

                    midpoint = (lower + upper) / 2
                    row_total = 0.0

                    for j in range(n_individuals):
                        partner_value = eligibility_matrix[i, j] * scaling_factors[j]
                        value = midpoint * partner_value
                        row_total += value / (1 + value)

                    if row_total < target:
                        lower = midpoint
                    else:
                        upper = midpoint

                scaling_factors[i] = (lower + upper) / 2

        largest_difference = 0.0

        for i in range(n_individuals):

            if targets[i] > 0:

                row_total = 0.0

                for j in range(n_individuals):
                    odds = (
                        scaling_factors[i] *
                        eligibility_matrix[i, j] *
                        scaling_factors[j]
                    )

                    row_total += odds / (1 + odds)

                difference = abs(row_total - targets[i])

                if difference > largest_difference:
                    largest_difference = difference

        if largest_difference < tolerance:
            break

    for i in range(n_individuals):
        for j in range(n_individuals):

            odds = (
                scaling_factors[i] *
                eligibility_matrix[i, j] *
                scaling_factors[j]
            )

            probability_matrix[i, j] = odds / (1 + odds)

    return probability_matrix

# ----- 2.7. Fill Matrices ----------------------------------------------------

def fill_manifest_contact_values(pars, manifest_matrices, case_01_nongroup_contact,
                                 population_contact_mode="lognormal",
                                 fixed_population_contact=15,
                                 max_population_contacts=120,
                                 max_iterations=1000, tolerance=1e-1,
                                 bisection_steps=10):

    meanlog = pars["contacts_MV"]["unique_nongroup_contacts_meanlog"]
    sdlog = pars["contacts_MV"]["unique_nongroup_contacts_sdlog"]
    group_contact = pars["contacts_MV"]["group_contacts"]

    individual_ids = list(manifest_matrices[0].index)
    n_individuals = len(individual_ids)

    if population_contact_mode == "lognormal":
        individual_contact_rates = np.random.lognormal(meanlog, sdlog, n_individuals)
        individual_contact_rates = np.minimum(individual_contact_rates, max_population_contacts)

    if population_contact_mode == "fixed":
        individual_contact_rates = np.full(n_individuals, fixed_population_contact, dtype=np.float64)

    individual_contact_rates[0] = case_01_nongroup_contact

    filled_matrices = []
    outside_contact_targets = np.zeros((len(manifest_matrices), n_individuals))

    for matrix_i, matrix in enumerate(manifest_matrices):

        placeholder_matrix = matrix.to_numpy()

        eligibility_matrix = (
            (placeholder_matrix == 1) |
            (placeholder_matrix == 2)
        ).astype(np.float64)

        filled_matrix = np.zeros(placeholder_matrix.shape, dtype=np.float64)
        filled_matrix[placeholder_matrix == 3] = group_contact

        n_cabin_mates = (placeholder_matrix == 3).sum(axis=1)
        n_possible_outside_contacts = eligibility_matrix.sum(axis=1)

        target_unique_contacts = individual_contact_rates - n_cabin_mates
        target_unique_contacts = np.maximum(0, target_unique_contacts)

        target_unique_contacts[0] = case_01_nongroup_contact
        target_unique_contacts[n_possible_outside_contacts == 0] = 0

        outside_contact_targets[matrix_i, :] = target_unique_contacts

        probability_matrix = find_symmetrical_matrix(
            eligibility_matrix,
            target_unique_contacts,
            max_iterations=max_iterations,
            tolerance=tolerance,
            bisection_steps=bisection_steps
        )

        probability_matrix = np.clip(probability_matrix, 0, 1 - 1e-12)
        lambda_matrix = -np.log1p(-probability_matrix)

        filled_matrix[eligibility_matrix > 0] = lambda_matrix[eligibility_matrix > 0]

        filled_matrices.append(filled_matrix)

    return filled_matrices, individual_contact_rates, outside_contact_targets

# ----- 2.8. Evaluate Filled Matrices -----------------------------------------

def check_filled_matrices(filled_matrices, outside_contact_targets, manifest_matrices,
                          difference_threshold=1e-6):

    checks = []

    for matrix_i in range(len(filled_matrices)):

        filled_matrix = filled_matrices[matrix_i]
        placeholder_matrix = manifest_matrices[matrix_i]

        matrix_targets = outside_contact_targets[
            outside_contact_targets["matrix"] == matrix_i
        ]

        for _, row in matrix_targets.iterrows():

            person_id = row["person_id"]

            non_cabin_cells = (
                (placeholder_matrix.loc[person_id, :] == 1) |
                (placeholder_matrix.loc[person_id, :] == 2)
            )

            pair_lambdas = filled_matrix.loc[person_id, non_cabin_cells]

            realised_unique_total = (1 - np.exp(-pair_lambdas)).sum()
            realised_contact_events = pair_lambdas.sum()

            expected_unique_total = row["outside_unique_contact_target"]

            checks.append({
                "matrix": matrix_i,
                "person_id": person_id,
                "expected_unique_total": expected_unique_total,
                "realised_unique_total": realised_unique_total,
                "realised_contact_events": realised_contact_events,
                "difference": realised_unique_total - expected_unique_total
            })

    checks = pd.DataFrame(checks)

    print("Difference summary:")
    print(checks["difference"].describe()[["min", "25%", "50%", "75%", "max"]])

    largest_difference = abs(checks["difference"]).max()
    passed_check = largest_difference < difference_threshold

    print()
    print("Largest absolute difference:", largest_difference)
    print("Passed threshold:", passed_check)

    return checks, passed_check


# -----------------------------------------------------------------------------
# ----- 3. MV Hondius Model ---------------------------------------------------
# -----------------------------------------------------------------------------

# ----- 3.1. Initialise Model Objects -----------------------------------------

def initialise_MV_model(pars, manifest, case_01_dates, cases):

    model_start = pars["model_MV"]["start_date"]
    model_end = pars["model_MV"]["end_date"]
    confinement_start = pars["contacts_MV"]["confinement_start_date"]

    n_days = (model_end - model_start).days + 1

    manifest_dates = []

    for date in manifest["Date"]:
        manifest_dates.append(date.date())

    matrix_start_days = np.zeros(len(manifest_dates), dtype=np.int64)

    for i, date in enumerate(manifest_dates):
        matrix_start_days[i] = (date - model_start).days

    matrix_for_day = np.zeros(n_days, dtype=np.int64)

    for matrix_i in range(len(matrix_start_days)):

        start_day = matrix_start_days[matrix_i]

        if matrix_i < len(matrix_start_days) - 1:
            end_day = matrix_start_days[matrix_i + 1]
        else:
            end_day = n_days

        matrix_for_day[start_day:end_day] = matrix_i

    case_01_prodromal_day = (case_01_dates["symptom_onset"] - model_start).days
    case_01_hps_day = (case_01_dates["death"] - model_start).days

    n_individuals = sum(manifest["P_embark"]) + sum(manifest["C_embark"])

    agents = np.full((4, n_individuals), -1, dtype=np.int64)

    agents[0, :] = 0
    agents[:, 0] = np.array([
        -1,
        0,
        case_01_prodromal_day,
        case_01_hps_day
    ])

    observed_symptom_onset_days = np.zeros(len(cases), dtype=np.int64)

    for i in range(len(cases)):
        observed_symptom_onset_days[i] = (
            cases["Symptom_onset"].iloc[i] - model_start
        ).days

    progression = pars["progression"]

    model_inputs = {
        "n_days": n_days,
        "matrix_for_day": matrix_for_day,
        "confinement_start_day": (confinement_start - model_start).days,
        "case_01_prodromal_day": case_01_prodromal_day,
        "case_01_hps_day": case_01_hps_day,
        "agents": agents,
        "observed_symptom_onset_days": observed_symptom_onset_days,
        "incubation_meanlog": progression["f_inc_meanlog"],
        "incubation_sdlog": progression["f_inc_sdlog"],
        "prodromal_shape": progression["f_pro_shape"],
        "prodromal_scale": progression["f_pro_scale"]
    }

    return model_inputs

# ----- 3.2. Run Model --------------------------------------------------------

def run_MV_model(pars, manifest_matrices, model_inputs, P_trans, C_Case01, e_conf,
                 population_contact_mode="lognormal",
                 fixed_population_contact=15,
                 max_population_contacts=120):

    S = 0
    E = 1
    PRO = 2
    HPS = 3

    n_days = model_inputs["n_days"]
    matrix_for_day = model_inputs["matrix_for_day"]
    confinement_start_day = model_inputs["confinement_start_day"]

    incubation_meanlog = model_inputs["incubation_meanlog"]
    incubation_sdlog = model_inputs["incubation_sdlog"]

    prodromal_shape = model_inputs["prodromal_shape"]
    prodromal_scale = model_inputs["prodromal_scale"]

    agents = model_inputs["agents"].copy()

    filled_matrices, individual_contact_rates, outside_contact_targets = fill_manifest_contact_values(
        pars,
        manifest_matrices,
        case_01_nongroup_contact=C_Case01,
        population_contact_mode=population_contact_mode,
        fixed_population_contact=fixed_population_contact,
        max_population_contacts=max_population_contacts
    )

    for day in range(n_days):

        contact_matrix = filled_matrices[matrix_for_day[day]]

        if day >= confinement_start_day:
            contact_matrix = contact_matrix * (1 - e_conf)

        infectious = np.where((agents[PRO, :] > -1) & (agents[PRO, :] <= day) & (agents[HPS, :] > day))[0]

        susceptible = np.where((agents[S, :] == 0) & (agents[E, :] == -1))[0]


        if len(infectious) == 0 or len(susceptible) == 0:
            continue

        for infector in infectious:

            susceptible = np.where((agents[S, :] == 0) & (agents[E, :] == -1))[0]

            if len(susceptible) == 0:
                break

            contact_lambdas = contact_matrix[infector, susceptible]

            contact_events = np.random.poisson(contact_lambdas)

            possible_transmissions = contact_events > 0

            if not np.any(possible_transmissions):
                continue

            contacted_susceptibles = susceptible[possible_transmissions]
            contacted_events = contact_events[possible_transmissions]

            successful_events = np.random.binomial( contacted_events, P_trans)

            newly_infected = contacted_susceptibles[successful_events > 0]

            if len(newly_infected) == 0:
                continue

            for infectee in newly_infected:

                incubation_period = np.random.lognormal(incubation_meanlog, incubation_sdlog)
                incubation_period = max(1, int(round(incubation_period)))

                prodromal_period = np.random.weibull(prodromal_shape)
                prodromal_period = prodromal_period * prodromal_scale
                prodromal_period = max(1, int(round(prodromal_period)))

                agents[E, infectee] = day
                agents[PRO, infectee] = day + incubation_period
                agents[HPS, infectee] = day + incubation_period + prodromal_period

    return agents

# -----------------------------------------------------------------------------
# ----- 4. Pseudo-likelihood --------------------------------------------------
# -----------------------------------------------------------------------------

# ----- 4.1. Pseudo-likelihood Core Logic -------------------------------------

@njit
def pseudo_likelihood_core(simulated_exposure_days, observed_symptom_onset_days,
                           incubation_meanlog, incubation_sdlog,
                           observation_cutoff_day, n_permutations):

    n_simulated = len(simulated_exposure_days)
    n_observed = len(observed_symptom_onset_days)

    if (n_simulated < n_observed) or (n_simulated > 24):
        return 0.0, 0.0

    sqrt_2 = math.sqrt(2.0)

    timing_probability_matrix = np.zeros((n_simulated, n_observed))

    for simulated_i in range(n_simulated):

        exposure_day = simulated_exposure_days[simulated_i]

        for observed_i in range(n_observed):

            symptom_onset_day = observed_symptom_onset_days[observed_i]

            lower = symptom_onset_day - exposure_day
            upper = symptom_onset_day + 1 - exposure_day

            if lower <= 0:
                lower_cdf = 0.0
            else:
                lower_z = (math.log(lower) - incubation_meanlog) / (incubation_sdlog * sqrt_2)
                lower_cdf = 0.5 * (1.0 + math.erf(lower_z))

            if upper <= 0:
                upper_cdf = 0.0
            else:
                upper_z = (math.log(upper) - incubation_meanlog) / (incubation_sdlog * sqrt_2)
                upper_cdf = 0.5 * (1.0 + math.erf(upper_z))

            timing_probability_matrix[simulated_i, observed_i] = upper_cdf - lower_cdf

    unobserved_probability = np.zeros(n_simulated)

    for simulated_i in range(n_simulated):

        exposure_day = simulated_exposure_days[simulated_i]
        cutoff_interval = observation_cutoff_day + 1 - exposure_day

        if cutoff_interval <= 0:
            observed_by_cutoff_cdf = 0.0
        else:
            cutoff_z = (math.log(cutoff_interval) - incubation_meanlog) / (incubation_sdlog * sqrt_2)
            observed_by_cutoff_cdf = 0.5 * (1.0 + math.erf(cutoff_z))

        unobserved_probability[simulated_i] = 1.0 - observed_by_cutoff_cdf

    likelihood_sum = 0.0
    permuted_indices = np.empty(n_simulated, dtype=np.int64)

    for permutation_i in range(n_permutations):

        for i in range(n_simulated):
            permuted_indices[i] = i

        for i in range(n_simulated - 1, 0, -1):
            j = np.random.randint(0, i + 1)

            temporary = permuted_indices[i]
            permuted_indices[i] = permuted_indices[j]
            permuted_indices[j] = temporary

        timing_likelihood = 1.0

        for observed_i in range(n_observed):
            simulated_i = permuted_indices[observed_i]
            timing_likelihood *= timing_probability_matrix[simulated_i, observed_i]

        unobserved_likelihood = 1.0

        for position_i in range(n_observed, n_simulated):
            simulated_i = permuted_indices[position_i]
            unobserved_likelihood *= unobserved_probability[simulated_i]

        likelihood_sum += timing_likelihood * unobserved_likelihood

    minimum_likelihood = likelihood_sum / n_permutations

    if n_simulated == n_observed:
        exact_likelihood = minimum_likelihood
    else:
        exact_likelihood = 0.0

    return exact_likelihood, minimum_likelihood

# ----- 4.2. Pseudo-likelihood Loop -------------------------------------------

def pseudo_likelihood(agents, model_inputs, observation_cutoff_day,
                      n_permutations=10000):

    E = 1

    observed_symptom_onset_days = model_inputs["observed_symptom_onset_days"].astype(np.int64)

    incubation_meanlog = model_inputs["incubation_meanlog"]
    incubation_sdlog = model_inputs["incubation_sdlog"]

    simulated_exposure_days = agents[E, 1:]
    simulated_exposure_days = simulated_exposure_days[simulated_exposure_days > -1]
    simulated_exposure_days = simulated_exposure_days.astype(np.int64)

    exact_likelihood, minimum_likelihood = pseudo_likelihood_core(
        simulated_exposure_days,
        observed_symptom_onset_days,
        incubation_meanlog,
        incubation_sdlog,
        observation_cutoff_day,
        n_permutations
    )

    return exact_likelihood, minimum_likelihood


# -----------------------------------------------------------------------------
# ----- 5. Mainland Model -----------------------------------------------------
# -----------------------------------------------------------------------------

# ----- 5.1. Initialise Model Object ------------------------------------------

def initialise_mainland_model(pars, start = [0,1,0,0], P_trans = 0.02):

    age_groups = ["Age_0_4", "Age_5_18", "Age_19_64", "Age_65_plus"]

    n_age_groups = len(age_groups)

    max_duration = pars["model_UK"]["max_duration"]

    population = np.zeros(n_age_groups)

    for age_i, age_group in enumerate(age_groups):
        population[age_i] = pars["UK_population"][age_group]

    contact_matrix = np.zeros((n_age_groups, n_age_groups), dtype=np.float64)

    for age_i, infectious_age_group in enumerate(age_groups):
        for age_j, contacted_age_group in enumerate(age_groups):
            contact_matrix[age_i, age_j] = pars["contacts_UK"][infectious_age_group][contacted_age_group]

    susceptible = np.zeros((max_duration + 1, n_age_groups), dtype=np.float64)
    exposed = susceptible.copy()
    prodromal = susceptible.copy()
    HPS = susceptible.copy()
    recovered = susceptible.copy()
    
    susceptible[0] = population - start
    exposed[0] = start

    agents = []    

    mainland_inputs = {
        "max_duration": max_duration,

        "population": population,
        "contact_matrix": contact_matrix,

        "S": susceptible,
        "E": exposed,
        "Pro": prodromal,
        "HPS": HPS,
        "R": recovered,

        "f_inc_meanlog": pars["progression"]["f_inc_meanlog"],
        "f_inc_sdlog": pars["progression"]["f_inc_sdlog"],

        "f_pro_shape": pars["progression"]["f_pro_shape"],
        "f_pro_scale": pars["progression"]["f_pro_scale"],

        "f_HPS_meanlog": pars["progression"]["f_HPS_meanlog"],
        "f_HPS_sdlog": pars["progression"]["f_HPS_sdlog"],
        
        "P_trans": P_trans,

        "agents": agents
    }

    return mainland_inputs

# ----- 5.2. Run Mainland Model -----------------------------------------------

def run_mainland_model(model, verbose=False):

    import numpy as np

    max_duration = model["max_duration"]
    n_age_groups = len(model["population"])

    population = model["population"]
    contact_matrix = model["contact_matrix"]
    P_trans = model["P_trans"]

    f_inc_meanlog = model["f_inc_meanlog"]
    f_inc_sdlog = model["f_inc_sdlog"]

    f_pro_shape = model["f_pro_shape"]
    f_pro_scale = model["f_pro_scale"]

    f_HPS_meanlog = model["f_HPS_meanlog"]
    f_HPS_sdlog = model["f_HPS_sdlog"]

    S = model["S"]
    E = model["E"]
    Pro = model["Pro"]
    HPS = model["HPS"]
    R = model["R"]

    current_S = S[0].copy()
    current_E = E[0].copy()
    current_Pro = Pro[0].copy()
    current_HPS = HPS[0].copy()
    current_R = R[0].copy()

    daily_new_HPS = np.zeros_like(HPS)
    cumulative_HPS = np.zeros(n_age_groups, dtype=np.float64)

    agents = []

    STATE_E = 1
    STATE_PRO = 2
    STATE_HPS = 3

    for age_i in range(n_age_groups):

        n_initial = int(current_E[age_i])

        for _ in range(n_initial):

            incubation_period = np.random.lognormal(f_inc_meanlog, f_inc_sdlog)
            incubation_period = max(1, int(round(incubation_period)))

            prodromal_period = np.random.weibull(f_pro_shape) * f_pro_scale
            prodromal_period = max(1, int(round(prodromal_period)))

            HPS_period = np.random.lognormal(f_HPS_meanlog, f_HPS_sdlog)
            HPS_period = max(1, int(round(HPS_period)))

            exposure_day = 0
            prodromal_day = exposure_day + incubation_period
            HPS_day = prodromal_day + prodromal_period
            recovered_day = HPS_day + HPS_period

            agents.append({
                "age_group": age_i,
                "state": STATE_E,
                "exposure_day": exposure_day,
                "prodromal_day": prodromal_day,
                "HPS_day": HPS_day,
                "recovered_day": recovered_day
            })

    if verbose:
        print("Day 0")

    for day in range(1, max_duration + 1):

        if verbose:
            print(f"Day {day}")

        active_agents = []

        for agent in agents:

            age_i = agent["age_group"]

            if agent["state"] == STATE_E:

                if agent["prodromal_day"] <= day:

                    agent["state"] = STATE_PRO

                    current_E[age_i] -= 1
                    current_Pro[age_i] += 1

            if agent["state"] == STATE_PRO:

                if agent["HPS_day"] <= day:

                    agent["state"] = STATE_HPS

                    current_Pro[age_i] -= 1
                    current_HPS[age_i] += 1

                    cumulative_HPS[age_i] += 1
                    daily_new_HPS[day, age_i] += 1

            if agent["state"] == STATE_HPS:

                if agent["recovered_day"] <= day:

                    current_HPS[age_i] -= 1
                    current_R[age_i] += 1

                    continue

            active_agents.append(agent)

        agents = active_agents

        new_exposures_by_age = np.zeros(n_age_groups, dtype=np.int64)

        for infectee_age_i in range(n_age_groups):

            if current_S[infectee_age_i] <= 0:
                continue

            force_of_infection = 0.0

            for infector_age_i in range(n_age_groups):

                force_of_infection += (
                    current_Pro[infector_age_i] *
                    contact_matrix[infector_age_i, infectee_age_i] *
                    P_trans /
                    population[infectee_age_i]
                )

            infection_probability = 1 - np.exp(-force_of_infection)

            if infection_probability <= 0:
                continue

            if infection_probability > 1:
                infection_probability = 1

            new_exposures = np.random.binomial(
                int(current_S[infectee_age_i]),
                infection_probability
            )

            if new_exposures > 0:
                new_exposures_by_age[infectee_age_i] = new_exposures

        for age_i in range(n_age_groups):

            n_new = int(new_exposures_by_age[age_i])

            if n_new == 0:
                continue

            current_S[age_i] -= n_new
            current_E[age_i] += n_new

            for _ in range(n_new):

                incubation_period = np.random.lognormal(f_inc_meanlog, f_inc_sdlog)
                incubation_period = max(1, int(round(incubation_period)))

                prodromal_period = np.random.weibull(f_pro_shape) * f_pro_scale
                prodromal_period = max(1, int(round(prodromal_period)))

                HPS_period = np.random.lognormal(f_HPS_meanlog, f_HPS_sdlog)
                HPS_period = max(1, int(round(HPS_period)))

                exposure_day = day
                prodromal_day = exposure_day + incubation_period
                HPS_day = prodromal_day + prodromal_period
                recovered_day = HPS_day + HPS_period

                agents.append({
                    "age_group": age_i,
                    "state": STATE_E,
                    "exposure_day": exposure_day,
                    "prodromal_day": prodromal_day,
                    "HPS_day": HPS_day,
                    "recovered_day": recovered_day
                })

        S[day] = current_S
        E[day] = current_E
        Pro[day] = current_Pro
        HPS[day] = current_HPS
        R[day] = current_R

        if current_E.sum() == 0 and current_Pro.sum() == 0 and current_HPS.sum() == 0:
            S[(day + 1):] = current_S
            E[(day + 1):] = current_E
            Pro[(day + 1):] = current_Pro
            HPS[(day + 1):] = current_HPS
            R[(day + 1):] = current_R
            break

    model["S"] = S
    model["E"] = E
    model["Pro"] = Pro
    model["HPS"] = HPS
    model["R"] = R

    model["daily_new_HPS"] = daily_new_HPS
    model["cumulative_HPS_by_age"] = cumulative_HPS
    model["cumulative_HPS_total"] = cumulative_HPS.sum()

    model["agents"] = agents

    return model



# -----------------------------------------------------------------------------
# ----- 6. MV Hondius Alternative Model (Within-individual heterogeneity) -----
# -----------------------------------------------------------------------------

def run_MV_model_daily_contacts(pars, manifest_matrices, model_inputs, P_trans,
                                C_Case01=None, e_conf=0,
                                population_contact_mode="lognormal",
                                fixed_population_contact=15,
                                max_population_contacts=120):

    S = 0
    E = 1
    PRO = 2
    HPS = 3

    n_days = model_inputs["n_days"]
    matrix_for_day = model_inputs["matrix_for_day"]
    confinement_start_day = model_inputs["confinement_start_day"]

    incubation_meanlog = model_inputs["incubation_meanlog"]
    incubation_sdlog = model_inputs["incubation_sdlog"]

    prodromal_shape = model_inputs["prodromal_shape"]
    prodromal_scale = model_inputs["prodromal_scale"]

    contact_meanlog = pars["contacts_MV"]["unique_nongroup_contacts_meanlog"]
    contact_sdlog = pars["contacts_MV"]["unique_nongroup_contacts_sdlog"]
    group_contact = pars["contacts_MV"]["group_contacts"]

    agents = model_inputs["agents"].copy()

    def infect_individual(infectee, day):

        incubation_period = np.random.lognormal(incubation_meanlog, incubation_sdlog)
        incubation_period = max(1, int(round(incubation_period)))

        prodromal_period = np.random.weibull(prodromal_shape)
        prodromal_period = prodromal_period * prodromal_scale
        prodromal_period = max(1, int(round(prodromal_period)))

        agents[E, infectee] = day
        agents[PRO, infectee] = day + incubation_period
        agents[HPS, infectee] = day + incubation_period + prodromal_period

    for day in range(n_days):

        placeholder_matrix = manifest_matrices[matrix_for_day[day]].to_numpy()

        infectious = np.where(
            (agents[PRO, :] > -1) &
            (agents[PRO, :] <= day) &
            (agents[HPS, :] > day)
        )[0]

        if len(infectious) == 0:
            continue

        for infector in infectious:

            cabin_contacts = np.where(placeholder_matrix[infector, :] == 3)[0]

            for infectee in cabin_contacts:

                if agents[S, infectee] != 0 or agents[E, infectee] > -1:
                    continue

                contact_events = np.random.poisson(group_contact)

                if contact_events == 0:
                    continue

                successful_events = np.random.binomial(contact_events, P_trans)

                if successful_events > 0:
                    infect_individual(infectee, day)

            outside_candidates = np.where(
                (placeholder_matrix[infector, :] == 1) |
                (placeholder_matrix[infector, :] == 2)
            )[0]

            if len(outside_candidates) == 0:
                continue

            if infector == 0:

                daily_unique_contacts = C_Case01
                daily_unique_contacts = min(daily_unique_contacts, max_population_contacts)

            else:

                if population_contact_mode == "lognormal":

                    daily_unique_contacts = np.random.lognormal(
                        contact_meanlog,
                        contact_sdlog
                    )

                    daily_unique_contacts = min(
                        daily_unique_contacts,
                        max_population_contacts
                    )

                elif population_contact_mode == "fixed":

                    daily_unique_contacts = fixed_population_contact

            n_cabin_mates = len(cabin_contacts)

            n_outside_contacts = int(round(daily_unique_contacts - n_cabin_mates))
            n_outside_contacts = max(0, n_outside_contacts)

            if day >= confinement_start_day:
                retention_probability = max(0, min(1, 1 - e_conf))
                n_outside_contacts = np.random.binomial(
                    n_outside_contacts,
                    retention_probability
                )

            n_outside_contacts = min(n_outside_contacts, len(outside_candidates))

            if n_outside_contacts == 0:
                continue

            contacted_individuals = np.random.choice(
                outside_candidates,
                size=n_outside_contacts,
                replace=False
            )

            for infectee in contacted_individuals:

                if agents[S, infectee] != 0 or agents[E, infectee] > -1:
                    continue

                successful_transmission = np.random.binomial(1, P_trans)

                if successful_transmission > 0:
                    infect_individual(infectee, day)

    return agents
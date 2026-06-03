# ==============================================================================
# ===== MV Hondius Andes Virus Outbreak Model: Parameterisation ================
# ==============================================================================

# ------------------------------------------------------------------------------
# ----- 0. Initialisation ------------------------------------------------------
# ------------------------------------------------------------------------------

# ----- 0.1. Description -------------------------------------------------------

# The following script is used to parameterise wait times and contact rates for
# the MV Hondius Andes virus outbreak and the mainland UK models.

# Referenced studies:

# Vial PA, Valdivieso F, Mertz G, Castillo C, Belmar E, Delgado I, Tapia M,
# Ferrés M. Incubation period of hantavirus cardiopulmonary syndrome. Emerging
# Infectious Diseases. 2006 Aug;12(8):1271.

# Valenzuela G, Barahona K, Rojas C, Barrera A, Henríquez C, Martínez-
# Valdebenito C, Potin M, Bedregal P, Ferrés M. Beyond ECMO Survival: Long-Term
# Symptom Burden and Quality-of-Life Impairment in Hantavirus Cardiopulmonary
# Syndrome Survivors. Viruses. 2025 Sep 15;17(9):1241.

# Pung R, Firth JA, Spurgin LG, Singapore CruiseSafe working group, Lee VJ,
# Kucharski AJ. Using high-resolution contact networks to evaluate SARS-CoV-2
# transmission and control in large-scale multi-day events. Nature communications.
# 2022 Apr 12;13(1):1956.

# Mossong J, Hens N, Jit M, Beutels P, Auranen K, Mikolajczyk R, Massari M,
# Salmaso S, Tomba GS, Wallinga J, Heijne J. Social contacts and mixing patterns
# relevant to the spread of infectious diseases. PLoS medicine. 2008 Mar 25;5(3):e74.

# ----- 0.2. Dependencies ------------------------------------------------------

library(tidyverse)
library(here)
library(MASS)
library(yaml)
library(socialmixr)

# ------------------------------------------------------------------------------
# ----- 1. E->P->HPS->R Parameters ---------------------------------------------
# ------------------------------------------------------------------------------

# ----- 1.1. Vial et. al. wait times -------------------------------------------

data_Vial <- read_csv(here("data", "parameterisation", "data_from_Vial_et_al.csv"))

data_Vial <- data_Vial %>% group_by(Patient) %>%
  mutate(Day = Day - Day[Stage == "P_end"]) %>%
  ungroup()

data_Vial_wide <- spread(data_Vial, key = Stage, value = Day)

ggplot(data_Vial_wide) +
  geom_segment(aes(x = E_start, xend = E_end, y = Patient, yend = Patient, colour = "Range of possible exposure"), linewidth = 2.5) +
  geom_segment(aes(x = E_end, xend = P_start, y = Patient, yend = Patient, colour = "Minimum incubation"), linewidth = 2.5) +
  geom_segment(aes(x = P_start, xend = P_end, y = Patient, yend = Patient, colour = "Prodromal"), linewidth = 2.5) +
  labs(x = "Day relative to P_end", y = "Patient", colour = "Stage") +
  theme_classic()

# ----- 1.2. Prodromal wait time distribution ----------------------------------

data_prodromal <- dplyr::select(data_Vial_wide, Patient, P_start, P_end)
data_prodromal$duration <- data_prodromal$P_end - data_prodromal$P_start

prodromal_gamma <- fitdistr(data_prodromal$duration, densfun = "gamma")
prodromal_lognormal <- fitdistr(data_prodromal$duration, densfun = "lognormal")
prodromal_weibull <- fitdistr(data_prodromal$duration, densfun = "weibull")

AIC(prodromal_gamma, prodromal_lognormal, prodromal_weibull)

gamma_shape <- unname(prodromal_gamma$estimate["shape"])
gamma_rate  <- unname(prodromal_gamma$estimate["rate"])

ln_meanlog <- unname(prodromal_lognormal$estimate["meanlog"])
ln_sdlog   <- unname(prodromal_lognormal$estimate["sdlog"])

weib_shape <- unname(prodromal_weibull$estimate["shape"])
weib_scale <- unname(prodromal_weibull$estimate["scale"])

ggplot(data_prodromal) +
  geom_histogram(aes(x = duration, y = after_stat(density)), binwidth = 0.5, colour = "white") +
  stat_function(aes(color = "gamma"), fun = dgamma, args = list(shape = gamma_shape, rate = gamma_rate)) +
  stat_function(aes(color = "lognormal"), fun = dlnorm, args = list(meanlog = ln_meanlog, sdlog = ln_sdlog)) +
  stat_function(aes(color = "weibull"), fun = dweibull, args = list(shape = weib_shape, scale = weib_scale)) +
  scale_x_continuous(limits = c(0, 10)) +
  theme_classic()

# ----- 1.3. Incubation wait time distribution ---------------------------------

data_incubation <- dplyr::select(data_Vial_wide, Patient, E_start, E_end, P_start)

data_incubation$E_day <- (data_incubation$E_start + data_incubation$E_end)/2
data_incubation$duration <- data_incubation$P_start - data_incubation$E_day

incubation_gamma <- fitdistr(data_incubation$duration, densfun = "gamma")
incubation_lognormal <- fitdistr(data_incubation$duration, densfun = "lognormal")
incubation_weibull <- fitdistr(data_incubation$duration, densfun = "weibull")

AIC(incubation_gamma, incubation_lognormal, incubation_weibull)

incubation_gamma_shape <- unname(incubation_gamma$estimate["shape"])
incubation_gamma_rate  <- unname(incubation_gamma$estimate["rate"])

incubation_ln_meanlog <- unname(incubation_lognormal$estimate["meanlog"])
incubation_ln_sdlog   <- unname(incubation_lognormal$estimate["sdlog"])

incubation_weib_shape <- unname(incubation_weibull$estimate["shape"])
incubation_weib_scale <- unname(incubation_weibull$estimate["scale"])

ggplot(data_incubation) +
  geom_histogram(aes(x = duration, y = after_stat(density)), binwidth = 0.5, colour = "white") +
  stat_function(aes(color = "gamma"), fun = dgamma, args = list(shape = incubation_gamma_shape, rate = incubation_gamma_rate)) +
  stat_function(aes(color = "lognormal"), fun = dlnorm, args = list(meanlog = incubation_ln_meanlog, sdlog = incubation_ln_sdlog)) +
  stat_function(aes(color = "weibull"), fun = dweibull, args = list(shape = incubation_weib_shape, scale = incubation_weib_scale)) +
  scale_x_continuous(limits = c(0, 40)) +
  theme_classic()

# ----- 1.3. Hospitalisation wait time distribution ----------------------------

data_Valenzuela <- read_csv(here("data", "parameterisation", "data_from_Valenzuela_et_al.csv"))

p <- c(0.25, 0.50, 0.75)
q <- c(data_Valenzuela$IQR_low[1], data_Valenzuela$Median[1], data_Valenzuela$IQR_high[1])

fit_quantiles <- function(start, qfun, par_transform, dist_name) {
  objective <- function(par) {
    theta <- par_transform(par)
    fitted_q <- do.call(qfun, c(list(p = p), theta))
    sum((fitted_q - q)^2)
  }
  
  fit <- optim(par = start, fn = objective)
  theta <- par_transform(fit$par)
  fitted_q <- do.call(qfun, c(list(p = p), theta))
  
  list(dist = dist_name, fit = fit, pars = theta, fitted_q = fitted_q, error = fit$value)
}

hospitalisation_gamma <- fit_quantiles(start = log(c(2, 2 / q[2])), qfun = qgamma, dist_name = "gamma",
                                       par_transform = function(par) list(shape = exp(par[1]), rate = exp(par[2])))

hospitalisation_lognormal <- fit_quantiles(start = c(log(q[2]), log(0.5)), qfun = qlnorm, dist_name = "lognormal",
                                           par_transform = function(par) list(meanlog = par[1], sdlog = exp(par[2])))

hospitalisation_weibull <- fit_quantiles(start = log(c(2, q[2])), qfun = qweibull, dist_name = "weibull",
                                         par_transform = function(par) list(shape = exp(par[1]), scale = exp(par[2])))

hospitalisation_fits <- list(hospitalisation_gamma, hospitalisation_lognormal, hospitalisation_weibull)

hospitalisation_fit_summary <- data.frame(distribution = sapply(hospitalisation_fits, \(x) x$dist),
                                          quantile_error = sapply(hospitalisation_fits, \(x) x$error))

hospitalisation_fit_summary$delta_error <- hospitalisation_fit_summary$quantile_error -
  min(hospitalisation_fit_summary$quantile_error)

hospitalisation_fit_summary[order(hospitalisation_fit_summary$quantile_error),]

hospitalisation_gamma_shape <- hospitalisation_gamma$pars$shape
hospitalisation_gamma_rate  <- hospitalisation_gamma$pars$rate

hospitalisation_ln_meanlog <- hospitalisation_lognormal$pars$meanlog
hospitalisation_ln_sdlog   <- hospitalisation_lognormal$pars$sdlog

hospitalisation_weib_shape <- hospitalisation_weibull$pars$shape
hospitalisation_weib_scale <- hospitalisation_weibull$pars$scale

hospitalisation_plotdata <- data.frame(day = seq(0.001, 60, length.out = 1000))

ggplot(hospitalisation_plotdata, aes(x = day)) +
  stat_function(aes(color = "gamma"), fun = dgamma, args = list(shape = hospitalisation_gamma_shape, rate = hospitalisation_gamma_rate)) +
  stat_function(aes(color = "lognormal"), fun = dlnorm, args = list(meanlog = hospitalisation_ln_meanlog, sdlog = hospitalisation_ln_sdlog)) +
  stat_function(aes(color = "weibull"), fun = dweibull, args = list(shape = hospitalisation_weib_shape, scale = hospitalisation_weib_scale)) +
  geom_vline(xintercept = q, linetype = 2) +
  scale_x_continuous(limits = c(0, 60)) +
  labs(x = "HPS duration / hospital stay", y = "density", color = "distribution") +
  theme_classic()

hospitalisation_quantiles <- data.frame(distribution = rep(c("reported", "gamma", "lognormal", "weibull"), each = 3),
                                        quantile = rep(c("Q1", "Median", "Q3"), times = 4),
                                        p = rep(p, times = 4),
                                        day = c(q, hospitalisation_gamma$fitted_q, hospitalisation_lognormal$fitted_q, hospitalisation_weibull$fitted_q))

ggplot(hospitalisation_quantiles, aes(x = quantile, y = day, colour = distribution, group = distribution)) +
  geom_point() +
  geom_line() +
  labs(x = "Quantile", y = "Days") +
  theme_classic()

# ------------------------------------------------------------------------------
# ----- 2. Contact Rates -------------------------------------------------------
# ------------------------------------------------------------------------------

# ----- 2.1. MV Hondius within-cabin -------------------------------------------

close_contact_definition <- 0.25

min_hours_spent_close <- 3

min_expected_cabin_contacts <- min_hours_spent_close / close_contact_definition

# ----- 2.2. MV Hondius contact rate distribution ------------------------------

contact_median <- 10^1.173
contact_low_95 <- 10^0.108
contact_high_95 <- 10^1.837

p <- c(0.025, 0.50, 0.975)
q <- c(contact_low_95, contact_median, contact_high_95)

fit_lognormal_quantiles <- function(start = c(log(q[2]), log(0.5))) {
  objective <- function(par) {
    meanlog <- par[1]
    sdlog <- exp(par[2])
    fitted_q <- qlnorm(p, meanlog = meanlog, sdlog = sdlog)
    sum((fitted_q - q)^2)
  }
  
  fit <- optim(par = start, fn = objective, control = list(maxit = 10000))
  
  list(fit = fit,
    meanlog = fit$par[1],
    sdlog = exp(fit$par[2]),
    fitted_q = qlnorm(p, meanlog = fit$par[1], sdlog = exp(fit$par[2])),
    quantile_error = fit$value)
}

contact_lognormal <- fit_lognormal_quantiles()

contact_ln_meanlog <- contact_lognormal$meanlog
contact_ln_sdlog <- contact_lognormal$sdlog

contact_lognormal$fitted_q
contact_lognormal$quantile_error

contact_plotdata <- data.frame(contact_rate = seq(0.001, contact_high_95 * 1.25, length.out = 1000))

ggplot(contact_plotdata, aes(x = contact_rate)) +
  stat_function(aes(color = "lognormal"), fun = dlnorm,
                args = list(meanlog = contact_ln_meanlog, sdlog = contact_ln_sdlog),
                linewidth = 1) +
  geom_vline(xintercept = q, linetype = 2) +
  labs(x = "Daily unique close contacts", y = "Density", color = "Distribution") +
  theme_classic()

# ----- 2.3. Mainland contact rates -------------------------------------------

data(polymod)

UK_population <- read_yaml(here("data", "parameters.yaml"))$UK


N_age <- c("0-4" = UK_population$Age_0_4,
           "5-18" = UK_population$Age_5_18,
           "19-64" = UK_population$Age_19_64,
           "65+" = UK_population$Age_65_plus)

age_limits <- c(0, 5, 19, 65)
age_labels <- c("0-4", "5-18", "19-64", "65+")

polymod_uk_15min <- polymod[country == "United Kingdom"]
polymod_uk_15min <- polymod_uk_15min[duration_multi >= 3]

contact_matrix <- assign_age_groups(polymod_uk_15min, age_limits = age_limits) %>%
  compute_matrix()

contact_matrix <- contact_matrix$matrix

rownames(contact_matrix) <- age_labels
colnames(contact_matrix) <- age_labels


contact_matrix_symmetric <- contact_matrix

for (i in seq_len(nrow(contact_matrix))) {
  for (j in seq_len(ncol(contact_matrix))) {
    contact_matrix_symmetric[i, j] <- (
      contact_matrix[i, j] * N_age[i] + contact_matrix[j, i] * N_age[j]
    ) / (2 * N_age[i])
  }
}

rownames(contact_matrix_symmetric) <- age_labels
colnames(contact_matrix_symmetric) <- age_labels


contact_matrix_plot <- as.data.frame(contact_matrix_symmetric) %>%
  rownames_to_column("contactor_age_group") %>%
  pivot_longer(cols = -contactor_age_group,
               names_to = "contactee_age_group",
               values_to = "contact_rate") %>%
  mutate(contactor_age_group = factor(contactor_age_group, levels = age_labels),
         contactee_age_group = factor(contactee_age_group, levels = age_labels))

ggplot(contact_matrix_plot, aes(x = contactee_age_group, y = contactor_age_group,
                                fill = contact_rate)) +
  geom_tile() +
  geom_text(aes(label = round(contact_rate, 2))) +
  scale_x_discrete(name = "Contactee age group", expand = c(0, 0)) +
  scale_y_discrete(name = "Contactor age group", expand = c(0, 0)) +
  scale_fill_viridis_c(name = "Expected\ndaily contacts") +
  theme_classic() +
  theme(aspect.ratio = 1)

# Note to self: Not expected to be symmetrical as expected daily contacts.
#               Will be symmetrical at the per-pair probability level:

pairwise_contact_matrix <- contact_matrix_symmetric

for (i in seq_len(nrow(contact_matrix_symmetric))) {
  for (j in seq_len(ncol(contact_matrix_symmetric))) {
    pairwise_contact_matrix[i, j] <- contact_matrix_symmetric[i, j] / N_age[j]
  }
}

round(pairwise_contact_matrix - t(pairwise_contact_matrix), 12)

pairwise_contact_matrix_plot <- as.data.frame(pairwise_contact_matrix) %>%
  rownames_to_column("contactor_age_group") %>%
  pivot_longer(cols = -contactor_age_group,
               names_to = "contactee_age_group",
               values_to = "pairwise_contact_rate") %>%
  mutate(contactor_age_group = factor(contactor_age_group, levels = age_labels),
         contactee_age_group = factor(contactee_age_group, levels = age_labels))

ggplot(pairwise_contact_matrix_plot, aes(x = contactee_age_group,
                                         y = contactor_age_group,
                                         fill = pairwise_contact_rate)) +
  geom_tile() +
  geom_text(aes(label = signif(pairwise_contact_rate, 3))) +
  scale_x_discrete(name = "Contactee age group", expand = c(0, 0)) +
  scale_y_discrete(name = "Contactor age group", expand = c(0, 0)) +
  scale_fill_viridis_c(name = "Per-pair\ncontact rate") +
  theme_classic() +
  theme(aspect.ratio = 1)

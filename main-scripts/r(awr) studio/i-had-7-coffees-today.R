# Load libraries
library(tidyverse)
library(brms)
library(tidybayes)
library(posterior)  # For epred & tidy posterior summaries
library(ggplot2)
library(lme4)
library(sjPlot)

# Load data
csrt_data <- read_csv("/Users/similovesyou/Desktop/qts/simian-behavior/5-csrt/derivatives/raw-modeling-data-elo-min-attempts.csv")

# Inspect structure
glimpse(csrt_data)
summary(csrt_data)

# Clean and normalize data
data_clean <- csrt_data %>%
  # Drop rows with missing key data
  filter(
    !is.na(result),
    !is.na(elo_score),
    !is.na(age),
    !is.na(gender),
    !is.na(species),
    !is.na(name)
  ) %>%
  # Ensure proper factor encoding
  mutate(
    gender = factor(gender),
    species = factor(species),
    result = factor(result, levels = c("success", "error", "prematured", "stepomission"))
  ) %>%
  # Scale age and elo within species
  group_by(species) %>%
  mutate(
    age_s = as.numeric(scale(age)),
    elo_s = as.numeric(scale(elo_score)),
    task_duration_s = as.numeric(scale(task_duration)),
    reaction_time_s = as.numeric(scale(reaction_time))
  ) %>%
  ungroup()

# Sanity checks
cat("\nFactor levels:\n")
print(levels(data_clean$result))
print(levels(data_clean$gender))
print(levels(data_clean$species))

cat("\nSummary statistics:\n")
summary(data_clean %>% select(age_s, elo_s, reaction_time_s, task_duration_s))

cat("\nDistribution of result:\n")
print(table(data_clean$result))

cat("\nNumber of attempts per subject:\n")
data_clean %>%
  group_by(name) %>%
  summarize(
    n_attempts = n(),
    mean_elo = mean(elo_s, na.rm = TRUE),
    mean_age = mean(age_s, na.rm = TRUE),
    success_rate = mean(result == "success")
  ) %>%
  summary()

# Visualization: Elo by result and gender
ggplot(data_clean, aes(x = elo_s, fill = result)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~gender) +
  theme_minimal() +
  labs(title = "Elo Distribution by Result and Gender", x = "Elo (scaled)", y = "Density")

# Visualization: Age by result and species
ggplot(data_clean, aes(x = age_s, fill = result)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~species) +
  theme_minimal() +
  labs(title = "Age Distribution by Result and Species", x = "Age (scaled)", y = "Density")

# Visualization: Reaction time by result and gender
ggplot(data_clean, aes(x = reaction_time_s, fill = result)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~gender) +
  theme_minimal() +
  labs(title = "Reaction time by Result and Gender", x = "Reaction time (scaled)", y = "Density")

# Visualization: Task duration by result and species
ggplot(data_clean, aes(x = task_duration_s, fill = result)) +
  geom_density(alpha = 0.5) +
  facet_wrap(~species) +
  theme_minimal() +
  labs(title = "Task Duration by Result and Species", x = "Age (scaled)", y = "Density")

fit_multinom <- brm(
  formula = result ~ 
    elo_s + I(elo_s^2) + 
    age_s + I(age_s^2) + 
    task_duration_s + reaction_time_s + 
    (1 | name) + (1 | species),
  data = data_clean,
  family = categorical(),
  prior = c(
    prior(normal(0, 1), class = "b", dpar = "muerror"),
    prior(normal(0, 1), class = "b", dpar = "muprematured"),
    prior(normal(0, 1), class = "b", dpar = "mustepomission")
  ),
  chains = 4,
  cores = 4,
  iter = 4000,
  warmup = 2000,
  seed = 123,
  backend = "cmdstanr",
  control = list(adapt_delta = 0.99, max_treedepth = 15),
  file = "fit_multinom_v2_model"
)


tab_model(fit_multinom)
summary(fit_multinom)

# Note to self -- try feeding the entire df and just add date as a predictor (maybe the model can account for measures being taken at different times)

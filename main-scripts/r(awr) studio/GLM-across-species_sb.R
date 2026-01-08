# Load libraries
library(readxl)
library(ggplot2)
library(lme4)
library(lmerTest)
library(MuMIn)

# Load Data
setwd ("D:/R/MALT_Ilaria")
dat <- read_excel("./demos-performance-combined.xlsx")

# Convert 'sex' to a factor
dat$sex <- factor(dat$sex, levels = c(1, 2), labels = c("M", "F"))

# Convert 'species' to a factor with capitalized levels
dat$species <- factor(dat$species, levels = c("Rhesus", "Tonkean"))

# GLM model for success
mdl_success <- lm(p_success ~ age * sex * species, data = dat)
summary(mdl_success)

options(na.action = "na.fail")
model_selection <- dredge(mdl_success, rank = BIC)

best_model <- get.models(model_selection, 1)[[1]]

summary(best_model)


# Plot data
ggplot(dat, aes(x = age, y = p_premature, shape = species)) +
  geom_point(size = 3) +
  geom_smooth(method = "lm", se = TRUE) +
  scale_color_manual(values = c("F" = "#FF69B4", "M" = "#4169E1")) +
  labs(
    title = "Performance by Species ",
    x = "Age",
    y = "Premature Performance",
    color = "Sex",
    shape = "Species"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    legend.position = "right"
  )



# GLM model for premature performance
mdl_prem <- lm(p_premature ~ age * sex * species, data = dat)
summary(mdl_prem)

options(na.action = "na.fail")
model_selection <- dredge(mdl_prem, rank = BIC)

best_model <- get.models(model_selection, 1)[[1]]

summary(best_model)

# Plot data
ggplot(dat, aes(x = age, y = p_success, shape = species)) +
  geom_point(size = 3) +
  geom_smooth(method = "lm", se = TRUE) +
  scale_color_manual(values = c("F" = "#FF69B4", "M" = "#4169E1")) +
  labs(
    title = "Performance by Species and Gender",
    x = "Age",
    y = "Successful Performance",
    color = "Sex",
    shape = "Species"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    legend.position = "right"
  )

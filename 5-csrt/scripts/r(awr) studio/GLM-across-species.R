# Load libraries
library(readxl)
library(ggplot2)
library(lme4)
library(lmerTest)

# Load Data
dat <- read_excel("/Users/similovesyou/Desktop/qts/simian-behavior/data/py/hierarchy/tonkean_task_demographics.xlsx")

# Convert 'sex' to a factor
dat$sex <- factor(dat$sex, levels = c(1, 2), labels = c("M", "F"))

# Convert 'species' to a factor with capitalized levels
dat$species <- factor(dat$species, levels = c("Rhesus", "Tonkean"))

# GLM model for success
mdl_success <- lm(p_success ~ age + sex + species, data = dat)
summary(mdl_success)

# GLM model for premature performance
mdl_prem <- lm(p_premature ~ age + sex + species, data = dat)
summary(mdl_prem)

# Plot data
ggplot(dat, aes(x = age, y = p_premature, col = sex, shape = species)) +
  geom_point(size = 3) +
  geom_smooth(method = "lm", se = TRUE) +
  scale_color_manual(values = c("F" = "#FF69B4", "M" = "#4169E1")) +
  labs(
    title = "Performance by Species and Gender",
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

# Plot data
ggplot(dat, aes(x = age, y = p_success, col = sex, shape = species)) +
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

library(tidyverse)
library(tidymodels)
library(themis)

df <- read_csv("diabetes_3class_clean.csv")

df <-  df %>%
  mutate(diabetes_class = factor(diabetes_class, 
                                 levels = c("No Diabetes", "Pre-Diabetes", "Type 2"))) %>%
  select(-c(glucose_fasting, glucose_postprandial, insulin_level,
            hba1c, diabetes_risk_score, diagnosed_diabetes, diabetes_stage))

set.seed(123)
data_split <- initial_split(df, prop = 0.80, strata= diabetes_class)
train_data <- training(data_split)
test_data <- testing(data_split)

diabetes_recipe <- recipe(diabetes_class ~ ., data = train_data) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(diabetes_class)


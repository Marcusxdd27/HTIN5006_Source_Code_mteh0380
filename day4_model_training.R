library(tidyverse)
library(tidymodels)
library(themis)
library(bonsai)

df <- read_csv("diabetes_3class_clean.csv")

#Converts diabetes_class to a factor with ordered levels & drops all leakage columns
df <- df %>%
  mutate(diabetes_class = factor(diabetes_class, 
                                 levels = c("No Diabetes", "Pre-Diabetes", "Type 2"))) %>%
  select(-c(glucose_fasting, glucose_postprandial, insulin_level,
            hba1c, diabetes_risk_score, diagnosed_diabetes, diabetes_stage))

set.seed(123) #Locks the randomness so the split is reproducible every time
data_split <- initial_split(df, prop = 0.80, strata = diabetes_class) #Splits data to 80/20 for training and testing, stratified by diabetes_class to preserve class proportions in both sets
train_data <- training(data_split) #Extracts the 80% training portion
test_data <- testing(data_split) #Extracts the 20% testing portion

diabetes_recipe <- recipe(diabetes_class ~ ., data = train_data) %>% #Defines the prediction formula and anchors preprocessing to training data only
  step_normalize(all_numeric_predictors()) %>% #Scales all numeric features to mean=0, SD=1
  step_dummy(all_nominal_predictors()) %>% #Converts categorical text columns to binary dummy variables
  step_smote(diabetes_class) #Synthetically oversamples minority classes to balance class distribution

#Define all 4 models
lr_model <- multinom_reg() %>%
  set_mode("classification") %>%
  set_engine("nnet")

rf_model <- rand_forest() %>%
  set_mode("classification") %>%
  set_engine("ranger")

xgb_model <- boost_tree() %>%
  set_mode("classification") %>%
  set_engine("xgboost")

lgbm_model <- boost_tree() %>%
  set_mode("classification") %>%
  set_engine("lightgbm")

#Wrap models into workflows (basically make a workflow to combine diabetes preprocessing recipes to models)
lr_workflow <- workflow() %>%
  add_recipe(diabetes_recipe) %>%
  add_model(lr_model)

rf_workflow <- workflow() %>%
  add_recipe(diabetes_recipe) %>%
  add_model(rf_model)

xgb_workflow <- workflow() %>%
  add_recipe(diabetes_recipe) %>%
  add_model(xgb_model)

lgbm_workflow <- workflow() %>%
  add_recipe(diabetes_recipe) %>%
  add_model(lgbm_model)

#Fit training data to models
lr_fit <- fit(lr_workflow, data = train_data)

rf_fit <- fit(rf_workflow, data = train_data)

xgb_fit <- fit(xgb_workflow, data = train_data)

lgbm_fit <- fit(lgbm_workflow, data = train_data)

#save all 4 models
saveRDS(lr_fit, "lr_fit.rds")
saveRDS(rf_fit, "rf_fit.rds")
saveRDS(xgb_fit, "xgb_fit.rds")
saveRDS(lgbm_fit, "lgbm_fit.rds")

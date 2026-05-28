library(tidyverse)
library(tidymodels)
library(themis)
library(bonsai)
library(doParallel)
library(scales)

# Log errors to file
options(error = quote(dump.frames("error_dump", TRUE)))

# Set up parallel processing (using 8 of 12 cores)
cl <- makePSOCKcluster(8)
registerDoParallel(cl)

df <- read_csv("diabetes_3class_clean.csv")

df <- df %>%
  mutate(diabetes_class = factor(diabetes_class, 
                                 levels = c("No Diabetes", "Pre-Diabetes", "Type 2"))) %>%
  select(-c(glucose_fasting, glucose_postprandial, insulin_level,
            hba1c, diabetes_risk_score, diagnosed_diabetes, diabetes_stage))

set.seed(123)
data_split <- initial_split(df, prop = 0.80, strata = diabetes_class)
train_data <- training(data_split)
test_data <- testing(data_split)

diabetes_recipe <- recipe(diabetes_class ~ ., data = train_data) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(diabetes_class)

#Load saved LR model fit
lr_fit <- readRDS("lr_fit.rds")

# Set up 5-fold cross-validation
set.seed(123)
cv_folds <- vfold_cv(train_data, v = 5, strata = diabetes_class)

#Redefine models with tunable hyperparameters
rf_model_tune <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 500
) %>%
  set_mode("classification") %>%
  set_engine("ranger")

xgb_model_tune <- boost_tree(
  tree_depth = tune(),
  learn_rate = tune(),
  trees = tune()
) %>%
  set_mode("classification") %>%
  set_engine("xgboost")

lgbm_model_tune <- boost_tree(
  tree_depth = tune(),
  learn_rate = tune(),
  trees = tune()
) %>%
  set_mode("classification") %>%
  set_engine("lightgbm")

#workflow update
rf_workflow_tune <- workflow() %>%
  add_recipe(diabetes_recipe) %>%
  add_model(rf_model_tune)

xgb_workflow_tune <- workflow() %>%
  add_recipe(diabetes_recipe) %>%
  add_model(xgb_model_tune)

lgbm_workflow_tune <- workflow() %>%
  add_recipe(diabetes_recipe) %>%
  add_model(lgbm_model_tune)

#Define tuning grids
rf_grid <- grid_random(
  mtry(range = c(2, 10)),
  min_n(range = c(5, 30)),
  size = 10
)

xgb_grid <- grid_regular(
  tree_depth(range = c(3, 8)),
  learn_rate(range = c(-2, -0.5), trans = log10_trans()),
  trees(range = c(100, 500)),
  levels = 3
)

lgbm_grid <- grid_regular(
  tree_depth(range = c(3, 8)),
  learn_rate(range = c(-2, -0.5), trans = log10_trans()),
  trees(range = c(100, 500)),
  levels = 3
)

#Tune Random Forest
rf_tune_results <- tune_grid(
  rf_workflow_tune,
  resamples = cv_folds,
  grid = rf_grid,
  metrics = metric_set(f_meas)
)

#Tune XGBoost
xgb_tune_results <- tune_grid(
  xgb_workflow_tune,
  resamples = cv_folds,
  grid = xgb_grid,
  metrics = metric_set(f_meas)
)

#Tune LightGBM
lgbm_tune_results <- tune_grid(
  lgbm_workflow_tune,
  resamples = cv_folds,
  grid = lgbm_grid,
  metrics = metric_set(f_meas)
)

# Select best hyperparameters
rf_best <- select_best(rf_tune_results, metric = "f_meas")
xgb_best <- select_best(xgb_tune_results, metric = "f_meas")
lgbm_best <- select_best(lgbm_tune_results, metric = "f_meas")

# Finalize workflows with best hyperparameters
rf_final_workflow <- finalize_workflow(rf_workflow_tune, rf_best)
xgb_final_workflow <- finalize_workflow(xgb_workflow_tune, xgb_best)
lgbm_final_workflow <- finalize_workflow(lgbm_workflow_tune, lgbm_best)

# Refit finalized models on full training data
rf_final_fit <- fit(rf_final_workflow, data = train_data)
xgb_final_fit <- fit(xgb_final_workflow, data = train_data)
lgbm_final_fit <- fit(lgbm_final_workflow, data = train_data)

# Save tuning results and final fits
saveRDS(rf_tune_results, "rf_tune_results.rds")
saveRDS(xgb_tune_results, "xgb_tune_results.rds")
saveRDS(lgbm_tune_results, "lgbm_tune_results.rds")
saveRDS(rf_final_fit, "rf_final_fit.rds")
saveRDS(xgb_final_fit, "xgb_final_fit.rds")
saveRDS(lgbm_final_fit, "lgbm_final_fit.rds")

# Stop parallel processing
stopCluster(cl)

#Success Message
cat("Day 5 tuning complete!\n")
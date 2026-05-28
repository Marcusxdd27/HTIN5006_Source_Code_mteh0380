library(tidyverse)
library(tidymodels)
library(bonsai)
library(themis)
library(shapviz)
library(rsample)

#load saved model fits
lr_fit <- readRDS("lr_fit.rds")
rf_final_fit <- readRDS("rf_final_fit.rds")
xgb_final_fit <- readRDS("xgb_final_fit.rds")
lgbm_final_fit <- readRDS("lgbm_final_fit.rds")

#data preprocessing
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

#Step 1: Generating predictions with all 4 models using test set
##Logistic regression 
lr_preds <- test_data %>%
  bind_cols(predict(lr_fit, test_data)) %>%
  bind_cols(predict(lr_fit, test_data, type = "prob"))

##Random Forest 
rf_preds <- test_data %>%
  bind_cols(predict(rf_final_fit, test_data)) %>%
  bind_cols(predict(rf_final_fit, test_data, type = "prob"))

##XGBoost
xgb_preds <- test_data %>%
  bind_cols(predict(xgb_final_fit, test_data)) %>%
  bind_cols(predict(xgb_final_fit, test_data, type = "prob"))

##LightGBM
lgbm_preds <- test_data %>%
  bind_cols(predict(lgbm_final_fit, test_data)) %>%
  bind_cols(predict(lgbm_final_fit, test_data, type = "prob"))

#Step 2: Evaluate metrics
## Define metric set
class_metrics <- metric_set(f_meas, precision, recall)

## Calculate metrics for each model
lr_metrics <- class_metrics(lr_preds, 
                            truth = diabetes_class, 
                            estimate = .pred_class,
                            estimator = "macro")

rf_metrics <- class_metrics(rf_preds,
                            truth = diabetes_class,
                            estimate = .pred_class,
                            estimator = "macro")

xgb_metrics <- class_metrics(xgb_preds,
                             truth = diabetes_class,
                             estimate = .pred_class,
                             estimator = "macro")

lgbm_metrics <- class_metrics(lgbm_preds,
                              truth = diabetes_class,
                              estimate = .pred_class,
                              estimator = "macro")

## Combine all metrics into one table (F1, Precision, Recall)
all_metrics <- bind_rows(
  lr_metrics %>% mutate(model = "Logistic Regression"),
  rf_metrics %>% mutate(model = "Random Forest"),
  xgb_metrics %>% mutate(model = "XGBoost"),
  lgbm_metrics %>% mutate(model = "LightGBM")
)

print(all_metrics)

## Confusion matrices
lr_cm <- conf_mat(lr_preds, truth = diabetes_class, estimate = .pred_class)

rf_cm <- conf_mat(rf_preds, truth = diabetes_class, estimate = .pred_class)

xgb_cm <- conf_mat(xgb_preds, truth = diabetes_class, estimate = .pred_class)

lgbm_cm <- conf_mat(lgbm_preds, truth = diabetes_class, estimate = .pred_class)

##Visualize the Confusion matrices
autoplot(lr_cm, type = "heatmap") + ggtitle("Logistic Regression")

autoplot(rf_cm, type = "heatmap") + ggtitle("Random Forest")

autoplot(xgb_cm, type = "heatmap") + ggtitle("XGBoost")

autoplot(lgbm_cm, type = "heatmap") + 
  ggtitle("LightGBM Confusion Matrix") +
  scale_fill_gradient(low = "white", high = "#2196F3") +
  theme_minimal()

##One-vs-Rest ROC-AUC
lr_roc <- roc_auc(lr_preds, truth = diabetes_class,
                  `.pred_No Diabetes`, `.pred_Pre-Diabetes`, `.pred_Type 2`,
                  estimator = "macro_weighted")

rf_roc <- roc_auc(rf_preds, truth = diabetes_class,
                  `.pred_No Diabetes`, `.pred_Pre-Diabetes`, `.pred_Type 2`,
                  estimator = "macro_weighted")

xgb_roc <- roc_auc(xgb_preds, truth = diabetes_class,
                  `.pred_No Diabetes`, `.pred_Pre-Diabetes`, `.pred_Type 2`,
                  estimator = "macro_weighted")

lgbm_roc <- roc_auc(lgbm_preds, truth = diabetes_class,
                  `.pred_No Diabetes`, `.pred_Pre-Diabetes`, `.pred_Type 2`,
                  estimator = "macro_weighted")

##Combine all One-vs-Rest ROC-AUC into a table
all_roc <- bind_rows(
  lr_roc %>% mutate(model = "Logistic Regression"),
  rf_roc %>% mutate(model = "Random Forest"),
  xgb_roc %>% mutate(model = "XGBoost"),
  lgbm_roc %>% mutate(model = "LightGBM")
)

print(all_roc)

#Step 3: Additional Analyses
##Subgroup performance analysis
### Subgroup performance by gender
lr_gender_metrics <- lr_preds %>%
  group_by(gender) %>%
  group_modify(~ class_metrics(.x, 
                               truth = diabetes_class, 
                               estimate = .pred_class,
                               estimator = "macro")) %>%
  mutate(model = "Logistic Regression")

rf_gender_metrics <- rf_preds %>%
  group_by(gender) %>%
  group_modify(~ class_metrics(.x, 
                               truth = diabetes_class, 
                               estimate = .pred_class,
                               estimator = "macro")) %>%
  mutate(model = "Random Forest")

xgb_gender_metrics <- xgb_preds %>%
  group_by(gender) %>%
  group_modify(~ class_metrics(.x, 
                               truth = diabetes_class, 
                               estimate = .pred_class,
                               estimator = "macro")) %>%
  mutate(model = "XGBoost")

lgbm_gender_metrics <- lgbm_preds %>%
  group_by(gender) %>%
  group_modify(~ class_metrics(.x, 
                               truth = diabetes_class, 
                               estimate = .pred_class,
                               estimator = "macro")) %>%
  mutate(model = "LightGBM")

all_spa_gender <- bind_rows(
  lr_gender_metrics,
  rf_gender_metrics,
  xgb_gender_metrics,
  lgbm_gender_metrics
)

print(all_spa_gender)

###Subgroup performance by ethnicity
lr_ethnicity_metrics <- lr_preds %>%
  group_by(ethnicity) %>%
  group_modify(~ class_metrics(.x, 
                               truth = diabetes_class, 
                               estimate = .pred_class,
                               estimator = "macro")) %>%
  mutate(model = "Logistic Regression")

rf_ethnicity_metrics <- rf_preds %>%
  group_by(ethnicity) %>%
  group_modify(~ class_metrics(.x, 
                               truth = diabetes_class, 
                               estimate = .pred_class,
                               estimator = "macro")) %>%
  mutate(model = "Random Forest")

xgb_ethnicity_metrics <- xgb_preds %>%
  group_by(ethnicity) %>%
  group_modify(~ class_metrics(.x, 
                               truth = diabetes_class, 
                               estimate = .pred_class,
                               estimator = "macro")) %>%
  mutate(model = "XGBoost")

lgbm_ethnicity_metrics <- lgbm_preds %>%
  group_by(ethnicity) %>%
  group_modify(~ class_metrics(.x, 
                               truth = diabetes_class, 
                               estimate = .pred_class,
                               estimator = "macro")) %>%
  mutate(model = "LightGBM")

all_spa_ethnicity <- bind_rows(
  lr_ethnicity_metrics,
  rf_ethnicity_metrics,
  xgb_ethnicity_metrics,
  lgbm_ethnicity_metrics
)

print(all_spa_ethnicity)

###Subgroup performance by income_level
lr_income_level_metrics <- lr_preds %>%
  group_by(income_level) %>%
  group_modify(~ class_metrics(.x, 
                               truth = diabetes_class, 
                               estimate = .pred_class,
                               estimator = "macro")) %>%
  mutate(model = "Logistic Regression")

rf_income_level_metrics <- rf_preds %>%
  group_by(income_level) %>%
  group_modify(~ class_metrics(.x, 
                               truth = diabetes_class, 
                               estimate = .pred_class,
                               estimator = "macro")) %>%
  mutate(model = "Random Forest")

xgb_income_level_metrics <- xgb_preds %>%
  group_by(income_level) %>%
  group_modify(~ class_metrics(.x, 
                               truth = diabetes_class, 
                               estimate = .pred_class,
                               estimator = "macro")) %>%
  mutate(model = "XGBoost")

lgbm_income_level_metrics <- lgbm_preds %>%
  group_by(income_level) %>%
  group_modify(~ class_metrics(.x, 
                               truth = diabetes_class, 
                               estimate = .pred_class,
                               estimator = "macro")) %>%
  mutate(model = "LightGBM")

all_spa_income_level <- bind_rows(
  lr_income_level_metrics,
  rf_income_level_metrics,
  xgb_income_level_metrics,
  lgbm_income_level_metrics
)

print(all_spa_income_level)

##McNemar's Test
mcnemar.test(table(lr_preds$.pred_class, rf_preds$.pred_class))
mcnemar.test(table(lr_preds$.pred_class, xgb_preds$.pred_class))
mcnemar.test(table(lr_preds$.pred_class, lgbm_preds$.pred_class))
mcnemar.test(table(rf_preds$.pred_class, xgb_preds$.pred_class))
mcnemar.test(table(rf_preds$.pred_class, lgbm_preds$.pred_class))
mcnemar.test(table(xgb_preds$.pred_class, lgbm_preds$.pred_class))

##Bootstrap confidence intervals
set.seed(123)
boot_ci <- function(preds, n = 1000) {
  bootstraps(preds, times = n) %>%
    mutate(f1 = map_dbl(splits, ~ {
      dat <- analysis(.x)
      f_meas(dat, truth = diabetes_class, 
             estimate = .pred_class, 
             estimator = "macro")$.estimate
    })) %>%
    summarise(
      mean_f1 = mean(f1),
      lower = quantile(f1, 0.025),
      upper = quantile(f1, 0.975)
    )
}

lr_ci <- boot_ci(lr_preds)
rf_ci <- boot_ci(rf_preds)
xgb_ci <- boot_ci(xgb_preds)
lgbm_ci <- boot_ci(lgbm_preds)

all_ci <- bind_rows(
  lr_ci %>% mutate(model = "Logistic Regression"),
  rf_ci %>% mutate(model = "Random Forest"),
  xgb_ci %>% mutate(model = "XGBoost"),
  lgbm_ci %>% mutate(model = "LightGBM")
)

print(all_ci)

#Step 4: Model Interpretation
##SHAP Values
# Extract the fitted LightGBM model
lgbm_engine <- extract_fit_engine(lgbm_final_fit)

# Prepare the training data through the recipe
lgbm_preprocessed <- lgbm_final_fit %>%
  extract_recipe() %>%
  bake(new_data = train_data %>% select(-diabetes_class))

# Compute SHAP values
shap <- shapviz(lgbm_engine, 
                X_pred = as.matrix(lgbm_preprocessed))

# Global feature importance (bar chart)
sv_importance(shap, kind = "bar") +
  scale_fill_manual(
    values = c("Class_1" = "gold", "Class_2" = "tomato", "Class_3" = "purple"),
    labels = c("Class_1" = "No Diabetes", 
               "Class_2" = "Pre-Diabetes", 
               "Class_3" = "Type 2")
  ) +
  ggtitle("LightGBM SHAP Feature Importance")

# Beeswarm plot (shows direction of effect)
sv_importance(shap, kind = "beeswarm") +
  ggtitle("LightGBM SHAP Beeswarm Plot")

##Domain Grouping
### Extract SHAP values
shap_vals <- apply(simplify2array(shap$S), c(1,2), mean)

### Define domain groupings
clinical_features <- c("bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp",
                       "heart_rate", "cholesterol_total", "hdl_cholesterol", 
                       "ldl_cholesterol", "triglycerides", "family_history_diabetes",
                       "hypertension_history", "cardiovascular_history")

behavioural_features <- c("smoking_status_Never", "smoking_status_Former",
                          "alcohol_consumption_per_week", 
                          "physical_activity_minutes_per_week",
                          "diet_score", "sleep_hours_per_day", 
                          "screen_time_hours_per_day")

socioeconomic_features <- c("age", "gender_Male", "education_level_Highschool",
                            "education_level_Postgraduate", "income_level_Middle",
                            "income_level_Upper.Middle", "employment_status_Retired",
                            "employment_status_Unemployed", "ethnicity_White",
                            "ethnicity_Black", "ethnicity_Hispanic", "ethnicity_Other")

### Calculate mean absolute SHAP per domain
domain_importance <- data.frame(
  domain = c("Clinical", "Behavioural", "Socioeconomic"),
  importance = c(
    mean(abs(shap_vals[, colnames(shap_vals) %in% clinical_features])),
    mean(abs(shap_vals[, colnames(shap_vals) %in% behavioural_features])),
    mean(abs(shap_vals[, colnames(shap_vals) %in% socioeconomic_features]))
  )
)

print(domain_importance)

### Visualize
ggplot(domain_importance, aes(x = domain, y = importance, fill = domain)) +
  geom_bar(stat = "identity") +
  labs(title = "Mean Absolute SHAP by Domain",
       x = "Domain", y = "Mean |SHAP value|") +
  theme(legend.position = "none")
                              
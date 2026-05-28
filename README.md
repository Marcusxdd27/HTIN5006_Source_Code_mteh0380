Diabetes ML Classification - Source Code
Project
Machine Learning Study on Type 2 Diabetes Classification Using Clinical, Behavioural, and Socioeconomic Indicators
Author
Marcus Teh Wei Jie
Requirements
R 4.5.2 with the following packages:

tidyverse
tidymodels
themis
bonsai
doParallel
scales
shapviz
corrplot

Dataset
Download diabetes_dataset.csv from: https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset
Place it in the same directory as the R scripts before running.
How to Run
Execute scripts in the following order:

day1_data_integrity.R — Data loading and integrity checks
day2_eda.R — Exploratory Data Analysis
day3_preprocessing.R — Data preprocessing and recipe pipeline
day4_model_training.R — Model training
day5_tuning.R — Hyperparameter tuning
day6_evaluation.R — Model evaluation and SHAP analysis

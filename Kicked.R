library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(lme4)
library(parsnip)
library(discrim)
library(kernlab)
library(themis)
#library(smote)
library(kernlab)


trainData <- vroom("training.csv")
testData <- vroom("test.csv")

trainData$IsBadBuy <- as.factor(trainData$IsBadBuy)


# my_recipe <- recipe(IsBadBuy ~ ., data = trainData) |> 
#   step_mutate_at(all_numeric_predictors(), fn = factor) |>  # turn all numeric features into factors
#   #step_other(all_nominal_predictors(), threshold = .001) |>  # combines categorical values that occur
#   #step_dummy(all_nominal_predictors()) |>  # dummy variable encoding
#   #step_mutate_at(ACTION, fn = factor) |> 
#   step_lencode_mixed(all_nominal_predictors(), 
#                      outcome = vars(IsBadBuy)) |> 
#   #step_zv(all_predictors()) |> 
#   step_normalize(all_numeric_predictors()) |> 
#   step_pca(all_predictors(), threshold = 0.8) |> 
#   step_smote(all_outcomes(), neighbors = 10)


my_recipe <- recipe(IsBadBuy ~ ., data = trainData) |> 
  update_role(RefId, new_role = 'ID') |> 
  update_role_requirements('ID', bake = FALSE) |> 
  step_mutate(IsBadBuy = factor(IsBadBuy), skip = TRUE) |> 
  #step_mutate(VNZIP1 = factor(VNZIP1)) |> 
  step_mutate(IsOnlineSale = factor(IsOnlineSale)) |> 
  step_mutate_at(all_nominal_predictors(), fn = factor) |> 
  #step_rm(contains('MMR')) |> 
  step_rm(BYRNO, WheelTypeID, VehYear, VNST, PurchDate, AUCGUART, PRIMEUNIT,
          Model, SubModel, Trim) |> 
  step_corr(all_numeric_predictors(), threshold = 0.7) |> 
  step_other(all_nominal_predictors(), threshold = 0.09) |> 
  step_novel(all_nominal_predictors()) |> 
  step_unknown(all_nominal_predictors()) |> 
  step_dummy(all_nominal_predictors()) |> 
  step_impute_median(all_numeric_predictors())


# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = trainData)

my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 100) %>%
  set_engine("ranger") %>%
  set_mode("classification")


rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

tuning_grid <- grid_regular(mtry(range = c(1, (ncol(baked) - 1))), min_n(),
                            levels = 3)

## Set up K-fold CV
folds <- vfold_cv(trainData, v = 2, repeats = 1)

CV_results <- rf_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = NULL)

## Find best tuning parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

CV_results %>%
  show_best(metric = "roc_auc")

## Predict
final_wf <-
  rf_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = trainData)


testData$MMRCurrentAuctionAveragePrice <- as.character(testData$MMRCurrentAuctionAveragePrice)
testData$MMRCurrentAuctionCleanPrice <- as.character(testData$MMRCurrentAuctionCleanPrice)
testData$MMRCurrentRetailAveragePrice <- as.character(testData$MMRCurrentRetailAveragePrice)
testData$MMRCurrentRetailCleanPrice <- as.character(testData$MMRCurrentRetailCleanPrice)


kicked_rf_predictions <- 
  predict(final_wf,
          new_data = testData,
          type = "prob") |> 
  bind_cols(testData) |> 
  rename(IsBadBuy = .pred_1) |> 
  select(RefId, IsBadBuy)

vroom_write(x = kicked_rf_predictions, 
            file = "./KickedRFPreds.csv", delim = ",")


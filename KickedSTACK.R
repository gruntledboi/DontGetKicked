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
library(dbarts)
library(stacks)


trainData <- vroom("training.csv")
testData <- vroom("test.csv")

trainData$IsBadBuy <- as.factor(trainData$IsBadBuy)


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



## Create a control grid
untunedModel <- control_stack_grid() #If tuning over a grid
tunedModel <- control_stack_resamples() #If not tuning a model

bart_model <- parsnip::bart(trees = 20) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")

bart_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_model)

bart_tuneGrid <- grid_regular(trees(),
                              levels = 4)

## Set up K-fold CV
folds <- vfold_cv(trainData, v = 3, repeats = 1)

tuned_barts <- bart_wf %>%
  tune_grid(resamples = folds,
            grid = bart_tuneGrid,
            metrics = metric_set(roc_auc),
            control = untunedModel)






my_mod <- rand_forest(mtry = tune(),
                      min_n = tune(),
                      trees = 50) %>%
  set_engine("ranger") %>%
  set_mode("classification")


rf_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod)

tuning_grid <- grid_regular(mtry(range = c(1,10)), min_n(), levels = 3)

## Set up K-fold CV
#folds <- vfold_cv(trainData, v = 2, repeats = 1)

forest_models <- rf_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc),
            control = untunedModel)





## Specify with models to include
my_stack <- stacks() |> 
  add_candidates(tuned_barts) |> 
  add_candidates(forest_models)

## Fit the stacked model
stack_mod <- my_stack %>%
  blend_predictions() %>% # LASSO penalized regression meta-learner
  fit_members() ## Fit the members to the dataset


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
            file = "./KickedRFPredsSTACK.csv", delim = ",")



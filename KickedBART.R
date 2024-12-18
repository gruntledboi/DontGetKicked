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



bart_model <- parsnip::bart(trees = 375) %>% # BART figures out depth and learn_rate
  set_engine("dbarts") %>% # might need to install
  set_mode("classification")

bart_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_model)

bart_tuneGrid <- grid_regular(trees(),
                              levels = 6)

## Set up K-fold CV
folds <- vfold_cv(trainData, v = 5, repeats = 1)


tuned_bart <- bart_wf %>%
  tune_grid(resamples = folds,
            grid = bart_tuneGrid,
            metrics = metric_set(accuracy)) #this cross-validates - I just called it something else


#START WORK HERE
bestTune <- tuned_bart %>%
  select_best(metric = "accuracy")

tuned_bart |> show_best(metric = "accuracy")

## Predict
final_wf <-
  bart_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = trainData)


# trainData$MMRAcquisitionAuctionAveragePrice <- as.character(trainData$MMRAcquisitionAuctionAveragePrice)
# trainData$MMRAcquisitionAuctionCleanPrice <- as.character(trainData$MMRAcquisitionAuctionCleanPrice)
# trainData$MMRAcquisitionRetailAveragePrice <- as.character(trainData$MMRAcquisitionRetailAveragePrice)
# trainData$MMRAcquisitonRetailCleanPrice <- as.character(trainData$MMRAcquisitonRetailCleanPrice)
# trainData$MMRCurrentAuctionAveragePrice <- as.character(trainData$MMRCurrentAuctionAveragePrice)
# trainData$MMRCurrentAuctionCleanPrice <- as.character(trainData$MMRCurrentAuctionCleanPrice)
# trainData$MMRCurrentRetailAveragePrice <- as.character(trainData$MMRCurrentRetailAveragePrice)
# trainData$MMRCurrentRetailCleanPrice <- as.character(trainData$MMRCurrentRetailCleanPrice)


testData$MMRCurrentAuctionAveragePrice <- as.character(testData$MMRCurrentAuctionAveragePrice)
testData$MMRCurrentAuctionCleanPrice <- as.character(testData$MMRCurrentAuctionCleanPrice)
testData$MMRCurrentRetailAveragePrice <- as.character(testData$MMRCurrentRetailAveragePrice)
testData$MMRCurrentRetailCleanPrice <- as.character(testData$MMRCurrentRetailCleanPrice)

kicked_bart_predictions <- 
  predict(final_wf,
          new_data = testData,
          type = "prob") |> 
  bind_cols(testData) |> 
  rename(IsBadBuy = .pred_1) |>
  select(RefId, IsBadBuy)

vroom_write(x = kicked_bart_predictions, 
            file = "./KickedRFPredsBART.csv", delim = ",")


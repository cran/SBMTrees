## ----include = FALSE----------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----setup--------------------------------------------------------------------
library(SBMTrees)
library(mitml)
library(lme4)

## ----prediction_sim-----------------------------------------------------------
# Simulate data
data <- simulation_prediction_conti(
   train_prop = 0.5,
   n_subject = 20,
   n_obs_per_sub = 5,
   nonlinear = TRUE,
   residual = "normal",
   randeff = "skewed_MVN",
   seed = 123)


## ----prediction---------------------------------------------------------------
# Fit the predictive model
model <- BMTrees_prediction(
   X_train = data$X_train,
   Y_train = data$Y_train,
   Z_train = data$Z_train,
   subject_id_train = data$subject_id_train,
   X_test = data$X_test,
   Z_test = data$Z_test,
   subject_id_test = data$subject_id_test,
   model = "BMTrees",
   binary = FALSE,
   nburn = 1L, npost = 1L, skip = 1L, verbose = FALSE, seed = 1234
 )

# Posterior expectation for the testing dataset
posterior_predictions <- model$post_predictive_y_test
head(colMeans(posterior_predictions))

## ----prediction_evaluation----------------------------------------------------
point_predictions = colMeans(posterior_predictions)

# Compute MAE
mae <- mean(abs(point_predictions - data$Y_test))
cat("Mean Absolute Error (MAE):", mae, "\n")

# Compute MSE
mse <- mean((point_predictions - data$Y_test)^2)
cat("Mean Squared Error (MSE):", mse, "\n")

# Compute 95% credible intervals
lower_bounds <- apply(posterior_predictions, 2, quantile, probs = 0.025)
upper_bounds <- apply(posterior_predictions, 2, quantile, probs = 0.975)

# Check if true values fall within the intervals
coverage <- mean(data$Y_test >= lower_bounds & data$Y_test <= upper_bounds)
cat("95% Posterior Predictive Interval Coverage:", coverage * 100, "%\n")



plot(data$Y_test, point_predictions, 
     xlab = "True Values", 
     ylab = "Predicted Values", 
     main = "True vs Predicted Values")
abline(0, 1, col = "red") # Add a 45-degree reference line


## ----imputation_sim-----------------------------------------------------------
# Simulate data with missing values
data <- simulation_imputation(NNY = TRUE, NNX = TRUE, 
                                  n_subject = 20, seed = 123)


## ----imputation---------------------------------------------------------------
imputed_model <- sequential_imputation(X = data$data_M[,3:14], Y = data$data_M$Y, Z = data$Z,
   subject_id = data$data_M$subject_id, type = c(0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1),
   outcome_model = "BMLM", binary_outcome = FALSE, model = "BMTrees", nburn = 0,
   npost = 2, skip = 1, verbose = FALSE, seed = 123)

# Extract imputed data
imputed_data <- imputed_model$imputed_data
dim(imputed_data) # Dimensions: posterior samples x observations x variables

## ----imputation_evaluation----------------------------------------------------
# create structure which can be used in mitml
MI_data = list()
for (i in 1:dim(imputed_data)[1]) {
  MI_data[[i]] = cbind(as.data.frame(imputed_data[i,,]), data$Z, data$data_M$subject_id)
  colnames(MI_data[[i]]) = c(colnames(data$data_M[,3:14]), "Y", "Z1", "Z2", "subject_id")
}
MI_data <- as.mitml.list(MI_data)  # Replace with actual datasets
# Fit the model on each imputed dataset
lmm_results <- with(MI_data, lmer(Y ~ X_1 + X_2 + X_3 + X_4 + X_5 + X_6
                                  + X_7 + X_8 + X_9 + X_10 + X_11 + X_12
                                  + (0 + Z1 + Z2 | subject_id)))

# Pool fixed effects using Rubin's Rules
pooled_results <- testEstimates(lmm_results)

# Print pooled results
print(pooled_results)



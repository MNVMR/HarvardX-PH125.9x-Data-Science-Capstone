# Load all libraries
# pheatmap: Generates pretty heatmaps, useful for visualizing data matrices.
library(pheatmap)

# scales: Provides methods for automatically and manually scaling graphs in ggplot2.
library(scales)

# readr: Part of the tidyverse; used for fast and efficient reading of tabular data.
library(readr)

# ggmap: Facilitates the use of Google Maps in data visualization.
library(ggmap)

# ggpubr: Offers tools to create 'ggplot2'-based publication-ready plots.
library(ggpubr)

# hrbrthemes: Provides modern and minimalistic themes for ggplot2.
library(hrbrthemes)

# ggcorrplot: Aids in visualizing a correlation matrix using ggplot2.
library(ggcorrplot)

# GGally: Extends ggplot2 by adding several functions to combine multiple plots.
library(GGally)

# car: Companion to Applied Regression; includes data sets, functions, and scripts.
library(car)

# caret: Provides functions for training and plotting classification and regression models.
library(caret)

# pwr: Tools for power analysis and sample size determination.
library(pwr)

# EnvStats: Functions for environmental statistics, including model fitting and hypothesis testing.
library(EnvStats)

# tidyverse: A collection of R packages designed for data science, including ggplot2, dplyr, and more.
library(tidyverse)

# lmtest: Provides tools for diagnostic testing in linear regression models.
library(lmtest)

# knitr: Allows for dynamic report generation with R, integrating code and its output into the report.
library(knitr)

# fastDummies: Quickly create dummy variables in data frames.
library(fastDummies)

# rmarkdown: Converts R Markdown documents into a variety of formats.
library(rmarkdown)

# markdown: Provides functions for rendering markdown documents in R.
library(markdown)

# tidyr: Facilitates easy tidying of data sets for analysis.
library(tidyr)

# MASS: Includes functions and datasets from the book "Modern Applied Statistics with S."
library(MASS)

# ggplot2: A system for declaratively creating graphics, based on the Grammar of Graphics.
library(ggplot2)

# lattice: Provides high-level data visualization based on a lattice structure.
library(lattice)

# dplyr: A set of tools for efficiently manipulating datasets in R.
library(dplyr)

# olsrr: Tools for building OLS regression models, including diagnostics and model selection.
library(olsrr)

# nnet: Software for feed-forward neural networks with a single hidden layer.
library(nnet)

# neuralnet: Training of neural networks for more complex models.
library(neuralnet)

# corrplot: Visualization of a correlation matrix.
library(corrplot)

# gridExtra: Functions in Grid graphics, particularly for arranging multiple grid-based plots.
library(gridExtra)

# fpc: Functions for clustering and cluster validation.
library(fpc)

# randomForest: Implements the random forest algorithm for classification and regression.
library(randomForest)

# rpart: Recursive partitioning for classification, regression, and survival trees.
library(rpart)

# e1071: Misc functions of the Department of Statistics, Probability Theory Group (formerly E1071), TU Wien.
library(e1071)

# kernlab: Kernel-based machine learning methods, including support vector machines.
library(kernlab)

# rpart.plot: Provides plotting functions for rpart models.
library(rpart.plot)

# rsq: Calculates R-squared values.
library(rsq)

# rfPermute: Estimates permutation-based p-values for random forest importance metrics.
library(rfPermute)

# ada: An implementation of the AdaBoost algorithm.
library(ada)

# glmnet: Lasso and elastic-net regularized generalized linear models.
library(glmnet)

# stats: A variety of statistical functions, including models and tests, that come standard with R.
library(stats)

# Metrics: Provides functions for evaluation of regression and classification models.
library(Metrics)

#========================================================================================================================================================

# Overview of the Dataset
df_house = read.csv("C:/Users/imano/OneDrive/İş masası/House pricing/KC_House_Sales.csv")
head(df_house)

#========================================================================================================================================================

# Checking Na values
cat("Number of NA values:", sum(is.na(df_house)))

#========================================================================================================================================================

# Dropping "id" column
df_house = subset(df_house, select = -id)

# The data in the column "price" must be converted to numeric
df_house$price <- as.numeric(gsub("[\\$,]", "", df_house$price))

df_house$renovated = ifelse(df_house$yr_renovated != 0, 1, 0)

# Cleanup of "date" and creation of "year", "month", and "day" columns
df_house$date <- as.POSIXct(df_house$date, format = "%Y%m%d")
df_house$year <- as.numeric(format(df_house$date, "%Y"))
df_house$month <- as.character(format(df_house$date, "%m"))
df_house$day <- as.numeric(format(df_house$date, "%d"))

df_house$age = ifelse(df_house$renovated == 1,
                      2024 - df_house$yr_renovated,
                      2024 - df_house$yr_built)

#Dropping "Date" column
df_house_original <- df_house
df_house_original[c("date")] <- list(NULL)

# Convert Zipcode to a Categorical variable (char) instead of number
df_house$zipcode = as.character(df_house$zipcode)
head(df_house)

#========================================================================================================================================================

summary(df_house)

#========================================================================================================================================================

# Collect columns that are numeric only
ndf_house = df_house[sapply(df_house, is.numeric)]

# Create a correlation Matrix
cor_matrix = cor(ndf_house)
correlation_df = as.data.frame(cor_matrix)
# New dataframe with variables 'highly' (>0.2) correlated with price
x_high = subset(ndf_house)
# Create a Heatmap
pheatmap(cor_matrix,
         color = colorRampPalette(c("blue", "white", "red"))(20),
         main = "Correlation Matrix Heatmap",
         fontsize = 8,
         cellwidth = 15,
         cellheight = 11,
         display_numbers = TRUE
)

#========================================================================================================================================================

# Remove redundant, unnecessary columns from dataset.
df_house[c("date","sqft_living", "yr_renovated",
           "yr_built")] <- list(NULL)

#========================================================================================================================================================

# Assuming 'data' is your data frame and 'price' is the column with price data
# Loop through all columns except the price column
for (variable_name in names(ndf_house)) {
  if (variable_name != "price") {
    # Create the scatter plot with ggplot
    p <- ggplot(ndf_house, aes_string(x=variable_name, y="price")) +
      geom_point() +  # Add points
      labs(x=variable_name, y="Price") +  # Label axes
      ggtitle(paste("Scatter plot of", variable_name, "vs Price"))  # Add title
    
    # Print the plot
    print(p)
  }
}

#========================================================================================================================================================

# Plot a Histogram of house sales prices
ggplot(df_house, aes(x = price)) +
  geom_histogram(bins = 30, fill = "blue", color = "black") +
  labs(title = "Histogram of House Prices", x = "Price", y = "Count") +
  theme_minimal()

#========================================================================================================================================================

# Boxplot for price by construction grade
ggplot(df_house, aes(x = factor(grade), y = price)) +
  geom_boxplot(fill = "blue", color = "black") +
  labs(title = "Boxplot of Prices by Construction Grade",
       x = "Grade", y = "Price") +
  theme_minimal()

#========================================================================================================================================================

# Boxplot for price by construction bedrooms
ggplot(df_house, aes(x = factor(bedrooms), y = price)) +
  geom_boxplot(fill = "blue", color = "black") +
  labs(title = "Boxplot of Prices by Construction bedrooms",
       x = "bedrooms", y = "Price") +
  theme_minimal()

#========================================================================================================================================================

# Boxplot for price by construction bathrooms
ggplot(df_house, aes(x = factor(bathrooms), y = price)) +
  geom_boxplot(fill = "blue", color = "black") +
  labs(title = "Boxplot of Prices by Construction bathrooms",
       x = "bathrooms", y = "Price") +
  theme_minimal()

#========================================================================================================================================================

# Boxplot for price by construction floors
ggplot(df_house, aes(x = factor(floors), y = price)) +
  geom_boxplot(fill = "blue", color = "black") +
  labs(title = "Boxplot of Prices by Construction floors",
       x = "floors", y = "Price") +
  theme_minimal()

#========================================================================================================================================================

# Boxplot for price by construction year
ggplot(df_house, aes(x = factor(year), y = price)) +
  geom_boxplot(fill = "blue", color = "black") +
  labs(title = "Boxplot of Prices by Construction year",
       x = "year", y = "Price") +
  theme_minimal()

#========================================================================================================================================================

# Boxplot for price by construction month
ggplot(df_house, aes(x = factor(month), y = price)) +
  geom_boxplot(fill = "blue", color = "black") +
  labs(title = "Boxplot of Prices by Construction month",
       x = "month", y = "Price") +
  theme_minimal()

#========================================================================================================================================================

# Boxplot for price by construction day
ggplot(df_house, aes(x = factor(day), y = price)) +
  geom_boxplot(fill = "blue", color = "black") +
  labs(title = "Boxplot of Prices by Construction day",
       x = "day", y = "Price") +
  theme_minimal()

#========================================================================================================================================================

# Boxplot for price by construction renovated
ggplot(df_house, aes(x = factor(renovated), y = price)) +
  geom_boxplot(fill = "blue", color = "black") +
  labs(title = "Boxplot of Prices by Construction renovated",
       x = "renovated", y = "Price") +
  theme_minimal()

#========================================================================================================================================================

# Boxplot for price by construction Waterfront
ggplot(df_house, aes(x = factor(waterfront), y = price)) +
  geom_boxplot(fill = "blue", color = "black") +
  labs(title = "Boxplot of Prices by Construction waterfront",
       x = "waterfront", y = "Price") +
  theme_minimal()

#========================================================================================================================================================

# Scatter plot of properties with mid-point transition color
ggplot(data = df_house, aes(x = long, y = lat, color = price)) +
  geom_point(alpha = 10) +
  scale_color_gradient(low = "blue", high = "red") +
  labs(title = "Geographical Distribution of House Prices in King County",
       x = "Longitude", y = "Latitude", color = "Price") +
  theme_minimal()

#========================================================================================================================================================

set.seed(123)
n<-dim(df_house)[1]
IND<-sample(c(1:n),round(n*0.8))
train.dat<-df_house[IND,]
test.dat<-df_house[-c(IND),]

dim(train.dat)
dim(test.dat)

#========================================================================================================================================================

# Fit a linear regression model based on the training dataset
house_lm<-lm(price ~.,data=train.dat)
summary(house_lm)

# Calculate the Mean Standard Error
MSE <- summary(house_lm)$sigma^2
print(MSE)

#========================================================================================================================================================

# Plotting the Regular Model
par(mfrow=c(2,2))
plot(house_lm)

#========================================================================================================================================================

# Variable Inflation Factor
vif(house_lm)

#========================================================================================================================================================

# Remove redundant, unnecessary columns from dataset. 
df_house[c("lat", "long", "year")] <- list(NULL)
train.dat[c("lat", "long", "year")] <- list(NULL)
test.dat[c("lat", "long", "year")] <- list(NULL)

#========================================================================================================================================================

set.seed(123)
n<-dim(df_house)[1]
IND<-sample(c(1:n),round(n*0.8))
train.dat<-df_house[IND,]
test.dat<-df_house[-c(IND),]

dim(train.dat)
dim(test.dat)

# Refitting the Model after removing Latitude and Longitude
house_lm<-lm(price ~.,data=train.dat)

# Variable Inflation Factor of the adjusted Model 
vif(house_lm)

#========================================================================================================================================================

summary(house_lm)

#========================================================================================================================================================

# Remove predictors from datasets
train.dat <- subset(train.dat, select = -c(sqft_lot15, day, month))
test.dat <- subset(test.dat, select = -c(sqft_lot15, day, month))

# Helper to display regression function with n coefficients
dispRegFunc <- function(reg) {
  coefs <- reg$coefficients
  b0 = coefs[1]
  n <- length(coefs)
  my_formula <- paste0("Y = ", round(b0, digits = 6))
  for (i in 2:n) {
    my_formula <- paste0(my_formula, " + ", round(coefs[i],6), names(coefs)[i])
  }
  my_formula
}

# Refit the model with updated datasets
house_lm <- lm(price ~ ., data = train.dat)
summary(house_lm)

# Display the Regression function of Price
output <- capture.output(dispRegFunc(house_lm))

# Print the output
cat(paste(strwrap(output, width=80), collapse="\n"))

#========================================================================================================================================================

# Test data Predictions
house_lm_test_pred <- predict(house_lm, newdata = test.dat)

house_lm_test_mse <- mean((house_lm_test_pred - test.dat$price)^2)
house_lm_test_rmse <- sqrt(house_lm_test_mse)
house_lm_test_residuals <- test.dat$price - house_lm_test_pred
house_lm_test_rsq <- 1 - var(house_lm_test_residuals) / var(test.dat$price)
house_lm_test_sse <- sum((test.dat$price - house_lm_test_pred)^2)

# Add the predictions to the results
results.df <- data.frame(model = "Linear Regression Model",
                         R.Squared.Train = summary(house_lm)$r.square,
                         R.Squared.Test = house_lm_test_rsq,
                         RMSE.test = house_lm_test_rmse,
                         SSE.test = house_lm_test_sse)

results.df.asc <- results.df[order(results.df$RMSE.test),]
kable(results.df.asc)

#========================================================================================================================================================

# Create a Data Frame for Ridge
set.seed(123)
n<-dim(df_house_original)[1]
IND<-sample(c(1:n),round(n*0.8))
train.dat.o<-df_house_original[IND,]
test.dat.o<-df_house_original[-c(IND),]

dim(train.dat.o)
dim(test.dat.o)

#========================================================================================================================================================

# Ridge Regression

# Extract 'x' and 'y'
x <- data.matrix(dplyr::select(train.dat.o, -price))
y <- train.dat$price

# Perform ridge regression
house_lm_ridge <- glmnet::cv.glmnet(x, y, alpha = 0, nlambda = 100, 
                                    lambda.min.ratio = 0.0001)
best.lambda.ridge <- house_lm_ridge$lambda.min
plot(house_lm_ridge)

print(paste0("Ridge best lambda of ", round(best.lambda.ridge, digits = 3)))

# Generating the results of the model
price.predictors.train <- colnames(dplyr::select(train.dat.o, -price))

ridge_results <- data.frame(
  price.train = train.dat.o$price,
  price.ridge.train = predict(
    house_lm_ridge, s = best.lambda.ridge, 
    newx = data.matrix(train.dat.o[price.predictors.train]))
)

calc_metrics <- function(actual, predicted) {
  sse <- sum((actual - predicted) ^ 2)
  mse <- sse / length(actual)
  rmse <- sqrt(mse) # Calculate RMSE
  sst <- sum((actual - mean(actual)) ^ 2)
  r2 <- 1 - sse / sst
  return(c(SST = sst, SSE = sse, MSE = mse, RMSE = rmse, R2 = r2))
}

# function to each set of predictions
ridge_metrics <- data.frame(
  Model = c("Ridge"),
  do.call(rbind, lapply(
    2:ncol(ridge_results), 
    function(i) calc_metrics(ridge_results$price.train, ridge_results[,i])))
)

# Display the metrics table with RMSE
ridge_metrics %>%
  dplyr::arrange(desc(R2)) %>%
  knitr::kable(caption = "SST, SSE, MSE, RMSE, and R2 of the Model")

#========================================================================================================================================================

# Append to results.df
ridge_train_pred = predict(house_lm_ridge, 
                           s = best.lambda.ridge, 
                           newx = data.matrix(train.dat.o[price.predictors.train]))
ridge_test_pred = predict(house_lm_ridge,
                          s = best.lambda.ridge,
                          newx = data.matrix(test.dat.o[price.predictors.train]))

ridge_train_results = postResample(pred = ridge_train_pred, obs = train.dat.o$price)
ridge_test_results = postResample(pred = ridge_test_pred, obs = test.dat.o$price)
ridge_test_sse = sum((ridge_test_pred - test.dat$price)^2)

# Append to the Results Data Frame
results.df = rbind(results.df,data.frame(model = "Ridge Regression",
                                         R.Squared.Train = unname(ridge_train_results[2]),
                                         R.Squared.Test = unname(ridge_test_results[2]),
                                         RMSE.test = unname(ridge_test_results[1]),
                                         SSE.test = ridge_test_sse))

results.df.asc <- results.df[order(results.df$RMSE.test),]
kable(results.df.asc)

#========================================================================================================================================================

# Lasso Regression
house_lm_lasso <- glmnet::cv.glmnet(x, y, alpha = 1, nlambda = 100, 
                                    lambda.min.ratio = 0.0001)
best.lambda.lasso <- house_lm_lasso$lambda.min
plot(house_lm_lasso)

print(paste0("Lasso best lambda of ", round(best.lambda.lasso, digits = 3)))

# Generating the results of the model
lasso_results <- data.frame(
  price.train = train.dat.o$price,
  price.lasso.train = predict(
    house_lm_lasso, s = best.lambda.lasso, 
    newx = data.matrix(train.dat.o[price.predictors.train]))
)

calc_metrics <- function(actual, predicted) {
  sse <- sum((actual - predicted) ^ 2)
  mse <- sse / length(actual)
  rmse <- sqrt(mse) # Calculate RMSE
  sst <- sum((actual - mean(actual)) ^ 2)
  r2 <- 1 - sse / sst
  return(c(SST = sst, SSE = sse, MSE = mse, RMSE = rmse, R2 = r2))
}

# function to each set of predictions
lasso_metrics <- data.frame(
  Model = c("Lasso"),
  do.call(
    rbind, lapply(
      2:ncol(lasso_results), 
      function(i) calc_metrics(lasso_results$price.train, lasso_results[,i])))
)

# Display the metrics table with RMSE
lasso_metrics %>%
  dplyr::arrange(desc(R2)) %>%
  knitr::kable(caption = "SST, SSE, MSE, RMSE, and R2 of the Model")

#========================================================================================================================================================

# Append to results.df

lasso_train_pred = predict(house_lm_lasso, 
                           s = best.lambda.lasso, 
                           newx = data.matrix(train.dat.o[price.predictors.train]))
lasso_test_pred = predict(house_lm_lasso, 
                          s = best.lambda.lasso, 
                          newx = data.matrix(test.dat.o[price.predictors.train]))

lasso_train_results = postResample(pred = lasso_train_pred, obs = train.dat.o$price)
lasso_test_results = postResample(pred = lasso_test_pred, obs = test.dat.o$price)
lasso_test_sse = sum((lasso_test_pred - test.dat$price)^2)

# Append to the Results Data Frame
results.df = rbind(results.df,data.frame(model = "Lasso Regression",
                                         R.Squared.Train = unname(lasso_train_results[2]),
                                         R.Squared.Test = unname(lasso_test_results[2]),
                                         RMSE.test = unname(lasso_test_results[1]),
                                         SSE.test = lasso_test_sse))

results.df.asc <- results.df[order(results.df$RMSE.test),]
kable(results.df.asc)

#========================================================================================================================================================

# Elastic Net Regression
house_lm_enet <- glmnet::cv.glmnet(x, y, alpha = 0.5, nlambda = 100, 
                                   lambda.min.ratio = 0.0001)
plot(house_lm_enet)
best.lambda.enet <- house_lm_enet$lambda.min

print(paste0("ElasticNet best lambda of ", round(best.lambda.enet, digits = 3)))

# Generating the results of the model
enet_results <- data.frame(
  price.train = train.dat.o$price,
  price.enet.train = predict(
    house_lm_enet, s = best.lambda.enet, 
    newx = data.matrix(train.dat.o[price.predictors.train]))
)

calc_metrics <- function(actual, predicted) {
  sse <- sum((actual - predicted) ^ 2)
  mse <- sse / length(actual)
  rmse <- sqrt(mse) # Calculate RMSE
  sst <- sum((actual - mean(actual)) ^ 2)
  r2 <- 1 - sse / sst
  return(c(SST = sst, SSE = sse, MSE = mse, RMSE = rmse, R2 = r2))
}

# function to each set of predictions
enet_metrics <- data.frame(
  Model = c("ElasticNet"),
  do.call(
    rbind, lapply(
      2:ncol(enet_results), 
      function(i) calc_metrics(enet_results$price.train, enet_results[,i])))
)

# Display the metrics table with RMSE
enet_metrics %>%
  dplyr::arrange(desc(R2)) %>%
  knitr::kable(caption = "SST, SSE, MSE, RMSE, and R2 of the Model")

#========================================================================================================================================================

# Append to results.df

enet_train_pred = predict(house_lm_enet,
                          s = best.lambda.enet,
                          newx = data.matrix(train.dat.o[price.predictors.train]))
enet_test_pred = predict(house_lm_enet,
                         s = best.lambda.enet,
                         newx = data.matrix(test.dat.o[price.predictors.train]))

enet_train_results = postResample(pred = enet_train_pred, obs = train.dat.o$price)
enet_test_results = postResample(pred = enet_test_pred, obs = test.dat.o$price)
enet_test_sse = sum((enet_test_pred - test.dat$price)^2)

# Append to the Results Data Frame
results.df = rbind(results.df,data.frame(model = "Elastic Net Regression",
                                         R.Squared.Train = unname(enet_train_results[2]),
                                         R.Squared.Test = unname(enet_test_results[2]),
                                         RMSE.test = unname(enet_test_results[1]),
                                         SSE.test = enet_test_sse)
)

results.df.asc <- results.df[order(results.df$RMSE.test),]
kable(results.df.asc)

#========================================================================================================================================================

# Build the SVM model
svm_model <- svm(price ~ ., data = train.dat)
print(summary(svm_model))

#========================================================================================================================================================

# Prediction and Performance
svm_predictions <- predict(svm_model, test.dat)
svm_rmse <- sqrt(mean((svm_predictions - test.dat$price)^2))
print(paste("SVM RMSE:", svm_rmse))

#========================================================================================================================================================

svm_train_pred = predict(svm_model,train.dat)
svm_test_pred =  predict(svm_model,test.dat)

svm_train_results = postResample(pred = svm_train_pred, obs = train.dat$price)
svm_test_results = postResample(pred = svm_test_pred, obs = test.dat$price)
svm_test_sse = sum((svm_test_pred - test.dat$price)^2)

# Append to the Results Data Frame
results.df = rbind(results.df,data.frame(model = "SVM Regression",
                                         R.Squared.Train = unname(svm_train_results[2]),
                                         R.Squared.Test = unname(svm_test_results[2]),
                                         RMSE.test = unname(svm_test_results[1]),
                                         SSE.test = svm_test_sse))

results.df.asc <- results.df[order(results.df$RMSE.test),]
kable(results.df.asc)

#========================================================================================================================================================

set.seed(123)
n<-dim(df_house_original)[1]
IND<-sample(c(1:n),round(n*0.8))
train.dat.tree <-df_house_original[IND,]
test.dat.tree <-df_house_original[-c(IND),]

#========================================================================================================================================================

depth_values <- c(2:7)

mse_values = numeric(length(depth_values))
test_rsq_values = numeric(length(depth_values))
train_rsq_values = numeric(length(depth_values))

for (i in seq_along(depth_values)) {
  depth = depth_values[i]
  
  regression_tree_model = rpart(price ~ ., 
                                data = train.dat, 
                                method = "anova", 
                                control=rpart.control(maxdepth=depth))
  predictions_test <- predict(regression_tree_model, newdata = test.dat)
  predictions_train <- predict(regression_tree_model, newdata = train.dat)
  
  mse_values[i] <- mean((predictions_test - test.dat$price)^2)
  test_rsq_values[i] = cor(predictions_test,test.dat$price)^2
  train_rsq_values[i] = cor(predictions_train,train.dat$price)^2
}

# Find the maximum R-squared value
max_rsq <- max(test_rsq_values)
# Get the corresponding depth value
best_depth <- depth_values[which.max(test_rsq_values)]

cat("Best depth:", best_depth, "with R-squared:", max_rsq)

#========================================================================================================================================================

# Fitting decision tree with best depth
dtm = rpart(price ~ ., 
            data = train.dat.tree, 
            method = "anova",
            control=rpart.control(maxdepth=best_depth))

# Print the tree
print(dtm)

# Plot the tree
rpart.plot(dtm, cex = .25, main=paste("Regression Tree - Depth: ", best_depth),
           extra=100,type=4,tweak=2)


#========================================================================================================================================================

# Data frame for plotting RMSE results
plot_data <- data.frame(Depth=depth_values, RMSE=mse_values)

# Plot RMSE results vs Depth of the Tree
ggplot(plot_data, aes(x=Depth, y=RMSE)) +
  geom_point(color = "blue") +
  geom_line(color = "blue") +
  labs(x = "Depth Value", y = "Root Mean Squared Error", 
       title = "Regression Tree Cross Validation (RMSE vs depth)") +
  theme_minimal()

#========================================================================================================================================================

# Data frame for plotting R-squared results
plot_data <- data.frame(Depth=depth_values, TestR2=test_rsq_values)

# Plot R-squared vs Depth of the Tree
ggplot(plot_data, aes(x=Depth, y=TestR2)) +
  geom_point(color="blue") +
  geom_line(color="blue") +
  geom_text(aes(label=round(TestR2, 4)), vjust=-0.5, color="black") +
  labs(x = "Depth Value", y = "Test R-squared", 
       title = "Regression Tree Cross Validation (R2 vs depth)") +
  theme_minimal()

#========================================================================================================================================================

imp = dtm$variable.importance
dt_test_pred <- predict(dtm, newdata=test.dat.tree)
dt_train_pred <- predict(dtm, newdata=train.dat.tree)
dt_test_results = postResample(pred = dt_test_pred, obs = test.dat.tree$price)
dt_train_results = postResample(pred = dt_train_pred, obs = train.dat.tree$price)
dt_test_sse = sum((dt_test_pred - test.dat.tree$price)^2)

# Variable importance may be more reliable 
# if considering other values from cross validation
blue_gradient <- colorRampPalette(c("blue", "white"))(length(imp))
barplot(imp, las=2, main="Variable Importance in Decision Tree", 
        col=blue_gradient, cex.names=0.8, cex.axis=0.7, cex.lab=0.7)

#========================================================================================================================================================

# Append results
results.df = rbind(results.df,data.frame(model = "Decision Tree Regression",
                                         R.Squared.Train = unname(dt_train_results[2]),
                                         R.Squared.Test = unname(dt_test_results[2]),
                                         RMSE.test = unname(dt_test_results[1]),
                                         SSE.test = dt_test_sse))

results.df.asc <- results.df[order(results.df$RMSE.test),]
kable(results.df.asc)

#========================================================================================================================================================

hyperparameter_grid <- expand.grid(
  ntree = c(200, 400, 550),  # Different values for ntree
  minsplit = c(4, 6, 8)  # Different values for minsplit
)

#========================================================================================================================================================

#Skip model tuning for computation time
skip_grid = TRUE
if (skip_grid==TRUE){
  best_rf = randomForest(price ~ .
                         -year-renovated-floors-month-day-condition-bedrooms-sqft_basement-bathrooms-sqft_lot15-age-yr_renovated, 
                         data = train.dat.tree,
                         ntree = 200,
                         mtry=7,
                         minsplit=4,
                         importance=TRUE)
  
  summary(best_rf)
  
} else{
  
  # Initialize model and its RMSE
  best_rf <- NULL
  best_rmse <- Inf
  
  # initialize scores (RMSE, R-square) 
  rmse_values_rf = numeric(nrow(hyperparameter_grid))
  test_rsq_values_rf = numeric(nrow(hyperparameter_grid))
  train_rsq_values_rf = numeric(nrow(hyperparameter_grid))
  
  # Perform Grid Search to tune ntree 
  #   (number of trees) and minsplit (minimum value in node to split)
  for (i in 1:nrow(hyperparameter_grid)) {
    current_model <- randomForest(
      formula = price ~. -year-renovated-floors-month-day-condition-bedrooms-sqft_basement-bathrooms-sqft_lot15-age-yr_renovated,
      data = train.dat.tree,
      ntree = hyperparameter_grid$ntree[i],
      minsplit = hyperparameter_grid$minsplit[i]
    )
    
    # Make predictions
    predictions_test <- predict(current_model, test.dat.tree)
    predictions_train <- predict(current_model, train.dat.tree)
    
    # store results
    train_results = postResample(pred = predictions_train, 
                                 obs = train.dat.tree$price)
    test_results = postResample(pred = predictions_test, 
                                obs = test.dat.tree$price)
    
    # RMSE
    current_rmse <- sqrt(mean((predictions_test - test.dat.tree$price)^2))
    
    rmse_values_rf[i] <- current_rmse
    
    # Rsquared
    test_rsq_values_rf[i] = unname(test_results[2])
    train_rsq_values_rf[i] = unname(train_results[2])
    
    
    # Check if current model is better than the best model so far
    if (current_rmse < best_rmse) {
      best_rmse <- current_rmse
      best_rf <- current_model
      best_minsplit = hyperparameter_grid$minsplit[i]
    }
  }
  
  # Cross validate to tune mtry (number of possible random variables per split)
  ctrl = trainControl(method = "cv",  # Cross-validation method
                      number = 5,    # Number of folds
                      verboseIter = TRUE,  # Display iteration progress
                      summaryFunction = defaultSummary)  # For regression
  
  param_grid = expand.grid(mtry=c(8,10,12))
  
  # Finding model
  best_rf = train(
    price ~ . -year-renovated-floors-yr_renovated-month-day-condition-bedrooms-sqft_basement-bathrooms-sqft_lot15-age, 
    data = train.dat.tree,
    method = "rf",
    trControl = ctrl,
    ntree=200,
    maxdepth=5,
    minsplit=4,
    tuneGrid = param_grid)
}

#========================================================================================================================================================

# Create a Dataframe with Random Forest variable importance
rf_importance <- as.data.frame(importance(best_rf))

# Create a tidy data frame for ggplot
rf_importance$Variable <- rownames(rf_importance)
rf_importance <- rf_importance %>%
  tidyr::gather(Measure, Value, -Variable)

# Plot RF variable importance
ggplot(rf_importance, aes(x = reorder(Variable, Value), y = Value)) +
  geom_col(fill="blue") +
  facet_wrap(~Measure, scales = "free") +
  coord_flip() +
  theme_minimal() +
  labs(x = NULL, y = NULL, title = "Variable Importance in Random Forest Model") +
  theme(legend.position = "none")

#========================================================================================================================================================

# Random Forest Test Results
rf_train_pred = predict(best_rf, newdata = train.dat.tree)
rf_test_pred = predict(best_rf, newdata = test.dat.tree)

rf_train_results = postResample(pred = rf_train_pred, obs = train.dat.tree$price)
rf_test_results = postResample(pred = rf_test_pred, obs = test.dat.tree$price)
rf_test_sse = sum((rf_test_pred - test.dat.tree$price)^2)

# Append to the Results Data Frame
results.df = rbind(results.df,data.frame(model = "Random Forest Tuned Model",
                                         R.Squared.Train = unname(rf_train_results[2]),
                                         R.Squared.Test = unname(rf_test_results[2]),
                                         RMSE.test = unname(rf_test_results[1]),
                                         SSE.test = rf_test_sse))

results.df.asc <- results.df[order(results.df$RMSE.test),]
kable(results.df.asc)

#========================================================================================================================================================
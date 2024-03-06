
library(corrplot)
library(ggplot2)
library(gridExtra)
library(caret)
library(tidyverse)
library(e1071)
library(ggcorrplot)
library(mice)
library(readr)
library(dplyr)
library(DMwR)
library(nnet) 
library(caret)
library(randomForest)
library(rpart)
library(pROC)
library(lightgbm)
library(data.table)


# Load the data
#file_path <- file.choose()
dataset = read.csv("D:\\hku\\financial fraud analytics\\assignment\\10.15\\enron.csv")

################################################Distinguish Attributes###################
print("structure of dataset")
str(dataset)
print("summary statistics")
summary(dataset)



#Extracts numeric variables and stores them in a character vector
numeric_vars <- dataset[, sapply(dataset, is.numeric)]
numeric_var_names <- names(dataset)[sapply(dataset, is.numeric)]
#print(numeric_var_names)

#Data skewness
skewness_values <- lapply(dataset[, sapply(dataset, is.numeric)], function(col) skewness(na.omit(col)))
print(skewness_values)


###############################Univariate Analysis #######################################



# Pie chart of poi
poi_data <- table(dataset$poi)
pie <- pie(poi_data, main="POI Distribution")

# Histograms of 19 variables
plots_list <- list()
for (var_name in numeric_var_names[1:19]) {
  p <- ggplot(data = dataset, aes_string(x = var_name)) + 
    geom_histogram(bins = 30, fill = "blue", color = "black", alpha = 0.7) +
    theme_minimal() +
    ggtitle(var_name)
  plots_list[[var_name]] <- p
}

do.call(gridExtra::grid.arrange, c(plots_list, ncol = 4))

#Scatterplot of all numeric variables and poi
plot_histogram <- function(data, variable_name) {
  ggplot(data, aes_string(x=variable_name, fill="poi")) +
    geom_histogram(position="identity", alpha=0.5, bins=50) +
    labs(title=paste(variable_name, "vs POI "), 
         x=variable_name, 
         y="Count") +
    theme_minimal()
}
# Generates histograms and saves them in a list
plot_list <- list()
for (variable in numeric_var_names) {
  plot_list[[variable]] <- plot_histogram(dataset, variable)
}
#A histogram of 19 cells is displayed using grid.arrange
do.call(grid.arrange, c(plot_list, ncol=4))


#mini_numeric_var_names=numeric_var_names - some attributes - non-numeric attributes

vars_to_remove <- c("total_payments", "total_stock_value", "other", 
                    "exercised_stock_options", "long_term_incentive", 
                    "bonus", "expenses","restricted_stock","deferral_payments","restricted_stock_deferred ")


mini_numeric_var_names <- setdiff(numeric_var_names, vars_to_remove)
print(mini_numeric_var_names)



#######################################Multivariate Analysis######################


mini_numeric_data <- dataset[, mini_numeric_var_names]

# Calculate the correlation matrix
corr_matrix <- cor(mini_numeric_data, use = "pairwise.complete.obs")

corrplot(corr_matrix, method="number")





#########################################Missing value##################################3
original_dataset <- dataset

#The value of a variable with a large number of missing values becomes 0
vars_to_change <- c("loan_advances", "restricted_stock_deferred", 
                    "deferral_payments", "deferred_income", 
                    "long_term_incentive", "director_fees")
dataset <- dataset %>%
  mutate_at(vars(vars_to_change), ~ifelse(is.na(.), 0, .))

# Replace missing values in the rest numeric columns with the median
dataset[] <- lapply(dataset, function(x) if(is.numeric(x)) ifelse(is.na(x), median(x, na.rm = TRUE), x) else x)



################################################univariate-outlier#################################
#Finds duplicate rows
duplicates <- dataset[duplicated(dataset) | duplicated(dataset, fromLast = TRUE), ]
#Print duplicate lines
print(duplicates)

# Outliers are detected for each numeric variable
outliers_list <- lapply(dataset[, sapply(dataset, is.numeric)], function(column) {
  Q1 <- quantile(column, 0.25, na.rm = TRUE)
  Q3 <- quantile(column, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  return(column[column < lower_bound | column > upper_bound])
})
# View the number of outliers
lapply(outliers_list, length)

# Plotting box plots for each variable
plot_boxplot <- function(data, variable_name) {
  ggplot(data, aes_string(y=variable_name)) +
    geom_boxplot(aes(fill="poi")) +
    labs(title=paste(variable_name, "Boxplot with Outliers"), 
         y=variable_name) +
    theme_minimal()
}

plot_list <- lapply(names(dataset)[sapply(dataset, is.numeric)], function(variable) {
  plot_boxplot(dataset, variable)
})

do.call(grid.arrange, c(plot_list, ncol=4))

#Delete the line named "TOTAL"
dataset <- dataset[dataset$X != "TOTAL", ]


###############################################Bivariate-outlier####################
plot_scatter <- function(data, variable_x, variable_y) {
  ggplot(data, aes_string(x=variable_x, y=variable_y, color="poi")) +
    geom_point(alpha=0.7) +
    labs(title=paste(variable_x, "vs", variable_y), 
         x=variable_x, 
         y=variable_y) +
    theme_minimal()
}

plot1 <- plot_scatter(dataset, "total_payments", "salary")
plot2 <- plot_scatter(dataset, "shared_receipt_with_poi", "to_messages")

grid.arrange(plot1, plot2, nrow=2)

#####################################multivariate-outlier##########
#  Standardized data 
scaled_data <- scale(dataset[, sapply(dataset, is.numeric)])

# Use of PCA 
pca_result <- prcomp(scaled_data, center = TRUE, scale. = TRUE)

# Calculated score 
scores <- as.data.frame(pca_result$x)

# Calculate the squared distance for each data point
squared_distances <- rowSums(scores^2)

# Setting thresholds (95% confidence level)
threshold <- quantile(squared_distances, 0.95)

# Detecting outliers
outliers <- which(squared_distances > threshold)
outlier_data <- dataset[outliers, ]
print(outlier_data)

set.seed(1234) 
dataset = read.csv( "D:/hku/financial fraud analytics/assignment/11.13/enron.csv")

# Delete the total column
dataset <- subset(dataset, X != "TOTAL")
# Delete columns named 'X' and 'email address'
dataset <- subset(dataset, select = -c(X,email_address))

#poi change into factor
dataset = dataset %>% mutate_if(is.character,as.factor) 
numeric_data <- dataset[sapply(dataset, is.numeric)]


####################calculate percentage_na#############################
total_rows <- nrow(dataset)
# Number of rows with any missing value
rows_with_na <- sum(apply(dataset, 1, function(x) any(is.na(x))))
# Calculating the percentage
percentage_na <- (rows_with_na / total_rows) * 100
# Print the percentage
print(percentage_na)


#############################################################
####delete some attitude with high ratio of missing value ####
###########################################################3

# Total number of rows in the dataset
total_rows <- nrow(dataset)

# Calculating the percentage of missing values for each column
na_percentage_per_column <- sapply(dataset, function(x) sum(is.na(x)) / total_rows * 100)

# Print the percentages
na_percentage_per_column

dataset <- subset(dataset, select = -c(loan_advances,director_fees,restricted_stock_deferred))


############################################
####fill missing value with Decision Tree####
#############################################

# Find a column with a missing value
columns_with_na <- sapply(dataset, function(x) sum(is.na(x))) > 0
na_columns <- names(columns_with_na[columns_with_na])
na_columns

# A function that fills in the missing value of a numeric variable
fill_missing_with_tree <- function(dataset, variable_name) {
  # Only variables with missing values are processed
  if (sum(is.na(dataset[[variable_name]])) > 0) {
    # Build a regression tree model using variables other than the current variable as predictors
    tree_model <- rpart(formula = paste(variable_name, "~ ."),
                        data = dataset[!is.na(dataset[[variable_name]]), ],
                        method = "anova")
    # Use the model to predict missing values
    predicted_values <- predict(tree_model, newdata = dataset[is.na(dataset[[variable_name]]), ])
    # Fill in the missing values of the forecast
    dataset[is.na(dataset[[variable_name]]), variable_name] <- predicted_values
  }
  
  return(dataset)
}

# Apply this function to every numerical variable with a missing value
for (var in na_columns) {
  if (is.numeric(dataset[[var]])) {
    dataset <- fill_missing_with_tree(dataset, var)
  }
}
# check
summary(dataset)
sapply(dataset, class)
colSums(is.na(dataset))

train = sample(nrow(dataset), 0.7*nrow(dataset), replace = FALSE)
TestSet = dataset[-train,]
TrainSet = dataset[train,]
round(prop.table(table(dataset$poi)),2)
table(dataset$poi)
#####################
####SMOTE############
#####################


perc.over = 100 * (nrow(TrainSet[TrainSet$poi == "False", ]) / nrow(TrainSet[TrainSet$poi == "True", ]) - 1)
perc.over

perc.under = 100
newData <- SMOTE(poi ~ ., TrainSet, perc.over = perc.over,perc.under=perc.under)
table(newData$poi)

#####################
####Random Forest####
#####################
TrainSet=newData

table(TrainSet$poi)

# Set the control parameters for cross validation
fitControl <- trainControl(method = "cv", number = 10)

#The random forest model is trained by train function and cross-validation is applied
model_rf_cv <- train(
  poi ~ ., 
  data = TrainSet, 
  method = "rf", 
  trControl = fitControl, 
  tuneLength = 5, # Try 5 different sets of mtry values
  ntree = 500
)

model_rf_cv

prediction_rf <- predict(model_rf_cv, TestSet)

summary(prediction_rf)
confusionMatrix(prediction_rf,TestSet$poi)

#plot roc
prediction_probabilities <- predict(model_rf_cv, TestSet, type = "prob")
prediction_probabilities
roc_curve <- roc(TestSet$poi, prediction_probabilities[, "True"])
auc_value <- auc(roc_curve)
print(auc_value)
plot(roc_curve, main="ROC Curve-Random Forest")
abline(a=0, b=1, lty=2, col="red")
text(0.6, 0.4, paste("AUC =", round(auc_value, 4)))


#####################
####lightgbm#########
#####################

X_train <- as.matrix(newData[, -which(names(newData) == "poi")])
y_train <- newData$poi

dtrain <- lgb.Dataset(data = X_train, label = as.numeric(y_train) - 1)


param_grid <- expand.grid(
  num_leaves = c(20, 31, 40),
  learning_rate = c(0.01, 0.05, 0.1),
  metric = "binary_logloss",
  objective = "binary",
  n_estimators = 100,
  stringsAsFactors = FALSE
)

results <- list()

for(i in 1:nrow(param_grid)) {
  params <- as.list(param_grid[i,])
  
  cv_results <- lgb.cv(
    params = params,
    data = dtrain,
    nfold = 10,
    nrounds = 100,
    early_stopping_rounds = 10,
    verbose = 0 
  )
  
  results[[i]] <- list(
    params = params,
    best_score = cv_results$best_score,
    best_iter = cv_results$best_iter,
    record_evals = cv_results$record_evals
  )
}

best_result <- results[[which.min(sapply(results, function(x) x$best_score))]]
print(best_result)

best_params <- best_result$params
best_params
lgb_model <- lgb.train(
  params = best_params,
  data = dtrain,
  nrounds = best_result$best_iter
)

X_test <- as.matrix(TestSet[, -which(names(TestSet) == "poi")])
y_test <- TestSet$poi

predictions <- predict(lgb_model, X_test)

prediction_labels <- ifelse(predictions > 0.5, "True", "False")

cm <- confusionMatrix(as.factor(prediction_labels), as.factor(y_test))
print(cm)

roc_curve <- roc(as.numeric(as.factor(y_test)), predictions)
auc_value <- auc(roc_curve)
plot(roc_curve, main = "ROC Curve - LightGBM")
abline(a = 0, b = 1, lty = 2, col = "red")
text(0.6, 0.4, paste("AUC =", round(auc_value, 4)))

summary(TestSet)


##################
###Normalized####
##################
TrainSet=newData
# Normalized function
table(TrainSet$poi)
# Custom normalization functions
normalize <- function(train, test) {
  normalized_train <- train
  normalized_test <- test
  
  num_columns <- sapply(train, is.numeric) 
  
 
  for(col in names(num_columns)[num_columns]) {
    max_val <- max(train[[col]], na.rm = TRUE)
    min_val <- min(train[[col]], na.rm = TRUE)
    
    normalized_train[[col]] <- (train[[col]] - min_val) / (max_val - min_val)
    normalized_test[[col]] <- (test[[col]] - min_val) / (max_val - min_val)
  }
  
  return(list(train = normalized_train, test = normalized_test))
}

norm_sets <- normalize(TrainSet, TestSet)
TrainSetNorm <- norm_sets$train
TestSetNorm <- norm_sets$test

#####################
####Neural Network###
#####################

#Cross-verify models for selecting optimal size and decay parameters
fitControl <- trainControl(method = "cv", number = 10) 

table(TrainSetNorm$poi)
model_nn_cv <- train(
  poi ~ ., 
  data = TrainSetNorm, 
  method = "nnet", 
  trControl = fitControl, 
  tuneGrid = expand.grid(.size = c(3, 4,5, 6,7), .decay = c(0.1, 0.01,0.2,0.02)),
  maxit = 150,
  trace = FALSE  # Set to FALSE to reduce redundant information in the training output
)


# check
print(model_nn_cv)

prediction_nn_cv <- predict(model_nn_cv, TestSetNorm)
confusionMatrix(prediction_nn_cv, TestSetNorm$poi)
prediction_probabilities <- predict(model_nn_cv, TestSetNorm, type = "prob")
roc_curve <- roc(TestSetNorm$poi, prediction_probabilities[, "True"])

auc_value <- auc(roc_curve)
print(auc_value)
plot(roc_curve, main="ROC Curve-Neural Network")
abline(a=0, b=1, lty=2, col="red")
text(0.6, 0.4, paste("AUC =", round(auc_value, 4)))




#####################
####### SVM    ######
#####################

fitControl <- trainControl(method = "cv", number = 10, classProbs = TRUE)

# Define the parameter grid
svmGrid <- expand.grid(
  C = 10^(1:4),
  sigma = 10^(-6:-1)
)

svm_model <- train(
  poi ~ ., 
  data = TrainSetNorm, 
  method = "svmRadial",
  trControl = fitControl,
  tuneGrid = svmGrid
)
svm_model
predictions_class <- predict(svm_model, TestSetNorm, type = "raw")

conf_matrix <- confusionMatrix(predictions_class, TestSetNorm$poi)

print(conf_matrix)

predictions_prob <- predict(svm_model, TestSetNorm, type = "prob")

roc_curve <- roc(response = as.numeric(TestSetNorm$poi == "True"), predictor = predictions_prob$True)
auc_value <- auc(roc_curve)

print(paste("AUC Value:", auc_value))

plot(roc_curve, main = "ROC Curve - SVM")
abline(a = 0, b = 1, lty = 2, col = "red")
text(0.6, 0.4, paste("AUC =", round(auc_value, 4)))


####################################
####fraud scenario identification###
###################################

fake_records <- data.frame(matrix(ncol = 17, nrow = 0))
colnames(fake_records) <- c("salary", "to_messages", "deferral_payments", 
                       "total_payments", "bonus", "deferred_income", 
                       "total_stock_value", "expenses", "from_poi_to_this_person",
                       "exercised_stock_options", "from_messages", "other", 
                       "from_this_person_to_poi", "poi", "long_term_incentive", 
                       "shared_receipt_with_poi", "restricted_stock")

# Scenario 1: False bonuses for executives
fake_records[1, ] <- c(1000000, 700, 2000000, 17000000, 8000000, -500000, 
                  15000000, 100000, 500, 10000000, 100, 7000000, 400, 
                  TRUE, 2000000, 3900, 13000000)


# Scenario 2: Fictitious Customer Program Bonus
fake_records[2, ] <- c(300000, 1500, 2500000, 2500000, 1000000, -1000000, 
                  3400000, 60000, 35, 2300000, 100, 200000, 17, 
                  TRUE, 550000, 1000, 850000)

# Scene 3 Unusually high bonus amounts
fake_records[3, ] <- c(295000, 500, 0, 1500000, 7500000, -300000, 
                  1000000, 50000, 20, 1000000, 40, 500000, 15, 
                  TRUE, 300000, 300, 400000)
# Scenario 4: Fictitious Supplier Bonus
fake_records[4, ] <- c(200000, 800, 400000, 2000000, 1200000, 0, 
                  2500000, 70000, 50, 2000000, 200, 750000, 25, 
                  TRUE, 400000, 800, 500000)

print(fake_records)

# Convert the target variable poi to a factor
fake_records$poi <- as.factor(fake_records$poi)

###########lgbm############
# Converted to a matrix for use in LightGBM
fake_records_matrix <- as.matrix(fake_records[, -which(names(fake_records) == "poi")])
#Prediction using the LightGBM model
fake_predictions <- predict(lgb_model, fake_records_matrix)
fake_prediction_lgbm <- ifelse(fake_predictions > 0.5, "True", "False")
print("LightGBM Predictions:")
print(fake_prediction_lgbm)

###########RF############
# Prediction using the Random Forest Model
fake_predictions_rf <- predict(model_rf_cv, fake_records)

print("Random Forest Predictions:")
print(fake_predictions_rf)


###########neural network############
# normalize fake_records
TrainSet=newData
fake_records_normalized <- fake_records
num_columns <- sapply(TrainSet, is.numeric)

for(col in names(num_columns)[num_columns]) {
  max_val <- max(TrainSet[[col]], na.rm = TRUE) 
  min_val <- min(TrainSet[[col]], na.rm = TRUE) 
  
  fake_records_normalized[[col]] <- (fake_records[[col]] - min_val) / (max_val - min_val)
}

# Now that fake_records_normalized is normalized, it can be used for model prediction
# using a neural network model for prediction
fake_predictions_nn <- predict(model_nn_cv, fake_records_normalized)

print("Neural Network Predictions:")
print(fake_predictions_nn)
###########SVM############
# Class prediction of normalized fake_records using SVM models
fake_predictions_svm <- predict(svm_model, fake_records_normalized, type = "raw")

print("SVM Predictions on Fake Records:")
print(fake_predictions_svm)


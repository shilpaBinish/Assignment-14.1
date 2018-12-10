getwd()
setwd("D:\\R")
disease
d <- read.csv("blog.csv", na.strings = c(""," ","NA"))
d<-d[!duplicated(d),]
library(Amelia)
missmap(d)
sapply(d,function(x) sum(is.na(x)))
x=na.omit(d)
d<-d[!names(d) %in% c("X")]
d
is_outlier <- function(x) {
  return(x < quantile(x, 0.25) - 1.5 * IQR(x) | x > quantile(x, 0.75) + 1.5 * IQR(x))
}
is_outlier
dput(colnames(d))


### Averaging ###
library(data.table)
library(foreach)
library(MatrixModels)

library(xgboost)
library(ranger)

# load and combine dataset
train = fread("BlogFeedback-Train.csv")
test = fread("BlogFeedback-Test.csv")

# error measure
mse = function(y_hat, y) {
  mse = mean((y - y_hat)^2)
  
  return(mse)
}

# create design matrices
train_x = model.Matrix(V281 ~ . - 1, data = train, sparse = F)
train_y = train$V281

test_x = model.Matrix(V281 ~ . - 1, data = test, sparse = F)
test_y = test$V281

train_xgb = xgb.DMatrix(data = as.matrix(train_x), label = train_y)
test_xgb = xgb.DMatrix(data = as.matrix(test_x), label = test_y)

# number of models
n = 5

# fit XGBoost
pred_xgb = foreach(i = 1:n, .combine = cbind) %do% {
  mdl_xgb = xgboost(data = train_xgb, nround = 750, nthread = 4, max_depth = 6, eta = 0.025, subsample = 0.7, gamma = 3)
  
  return(predict(mdl_xgb, test_xgb))
}

# fit random forest
pred_rf = foreach(i = 1:n, .combine = cbind) %do% {
  mdl_rf = ranger(V281 ~ ., data = train, num.trees = 1000, mtry = 120, write.forest = T)
  
  return(predict(mdl_rf, test)$predictions)
}

# weighted average
mse(rowMeans(pred_rf) * 0.25 + rowMeans(pred_xgb) * 0.75, test_y)






### Basic Models ###
library(data.table)
library(MatrixModels)

library(e1071)
library(FNN)
library(glmnet)
library(ranger)
library(xgboost)

# load and combine dataset
train = fread("BlogFeedback-Train.csv")
test = fread("BlogFeedback-Test.csv")

# error measure
mse = function(y_hat, y) {
  mse = mean((y - y_hat)^2)
  
  return(mse)
}

# create design matrices
train_x = model.Matrix(V281 ~ . - 1, data = train, sparse = F)
train_x_sparse = model.Matrix(V281 ~ . - 1, data = train, sparse = T)
train_y = train$V281

test_x = model.Matrix(V281 ~ . - 1, data = test, sparse = F)
test_y = test$V281

train_xgb = xgb.DMatrix(data = as.matrix(train_x), label = train_y)
test_xgb = xgb.DMatrix(data = as.matrix(test_x), label = test_y)

# try kNN
pred_knn = knn.reg(train_x, test_x, train_y, k = 19)$pred
mse(pred_knn, test_y)

# try LASSO
mdl_lasso = cv.glmnet(train_x_sparse, train_y, family = "gaussian", alpha = 1)
pred_lasso = predict(mdl_lasso, newx = test_x)
mse(pred_lasso, test_y)

# try random forest
mdl_rf = ranger(V281 ~ ., data = train, num.trees = 1000, mtry = 120, write.forest = T)
pred_rf = predict(mdl_rf, test)
mse(pred_rf$predictions, test_y)

# try SVM
mdl_svm = svm(V281 ~ V52 + V55 + V61 + V51 + V54 + V21 + V6 + V10, data = train, kernel = "radial", cost = 2, gamma = 0.25)
pred_svm = predict(mdl_svm, test)
mse(pred_svm, test_y)

# try XGboost
mdl_xgb = xgboost(data = train_xgb, nround = 750, nthread = 4, max_depth = 6, eta = 0.025, subsample = 0.7, gamma = 3)
pred_xgb = predict(mdl_xgb, test_xgb)
mse(pred_xgb, test_y)
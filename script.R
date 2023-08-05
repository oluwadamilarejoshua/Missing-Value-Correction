library(forecast)
library(randtests)
library(glmnet)
library(tseries)
library(readxl)
library(tidyverse)
library(caret)


# Data importing and preparation ------------------------------------------

recoded <- read_excel("recoded.xlsx")
# View(recoded)

lambdas <- 10^seq(2, -3, by = -.1)
alpha.grid <- seq(0, 1, length=15)

y = recoded$Inflation
df_x <- as.matrix(recoded[, -c(1, 3)])

df1_ <- data.frame(dates = recoded[,1], Variable = recoded[, 3])
head(df1_)

dim(recoded)

ggplot(data = df1_,aes(y = Inflation, x = Date)) + 
  geom_line() +
  ggtitle("Inflation Series")

# Unit Root Test/ Test for Stationary -------------------------------------

adf.test(y, k=0) # Series contains unit root i.e. not stationary

# Series Differencing -----------------------------------------------------

diff_y <- diff(y, 1, 1)
adf.test(diff_y, k=0) # Series stationary at difference 1

df1 <- data.frame(dates = recoded[2:486,1], Inflation = diff_y)
head(df1)

ggplot(data = df1,aes(y = Inflation, x = Date)) + 
  geom_line() +
  ggtitle("Differenced Inflation Series")

# ---- --------------------------------------------------------------------

#------------------------------- The models -------------------------------

# ----- -------------------------------------------------------------------



# AR(1) Model -------------------------------------------------------------

ar1_model <- Arima(diff_y, order = c(1,0,0))
summary(ar1_model)

ar1_pred = ar1_model$residuals + diff_y
ar1.mse <- sum((diff_y - ar1_pred)^2)/length(diff_y)
ar1.mae <- sum(abs(diff_y - ar1_pred))/length(diff_y)

  # Plotting the graph of observed vs predicted
ar1_df <- data.frame(dates = recoded[2:486,1], Inflation = ar1_pred)
head(ar1_df)

joined.ar_ac <- (rbind(df1, ar1_df))
ar.g <- df1 %>% mutate(Type = "Observed") %>% 
  bind_rows(ar1_df %>% 
              mutate(Type = "AR(1) Predicted"))
View(ar.g)
ar <- cbind(joined.ar_ac, ar.g$Type)
names(ar) <- c("Date", "Inflation", "Type")
head(ar)

ggplot(ar,aes(y = Inflation, x = Date, color = Type)) + 
  geom_line() +
  ggtitle("Observed vs AR(1) Pediction")

# Random Walk Model -------------------------------------------------------

  # Test for random walk
model1 <- lm(diff_y ~ 1)
runs.test(model1$residuals)

  # RW model
rw_model <- Arima(diff_y, order = c(0,1,0))
summary(rw_model)

rw_pred = rw_model$residuals + diff_y
rw.mse <- sum((diff_y - rw_pred)^2)/length(diff_y)
rw.mae <- sum(abs(diff_y - rw_pred))/length(diff_y)

# Plotting the graph of observed vs predicted
rw1_df <- data.frame(dates = recoded[2:486,1], Inflation = rw_pred)
head(rw1_df)

joined.rw_ac <- (rbind(df1, rw1_df))
rw.g <- df1 %>% mutate(Type = "Observed") %>% 
  bind_rows(rw1_df %>% 
              mutate(Type = "RW Predicted"))
rw <- cbind(joined.rw_ac, rw.g$Type)
names(rw) <- c("Date", "Inflation", "Type")
head(rw)

ggplot(rw,aes(y = Inflation, x = Date, color = Type)) + 
  geom_line() +
  ggtitle("Observed vs Random Walk Pediction")


# Lasso Regression --------------------------------------------------------

auto.lasso <- glmnet(df_x, y, alpha = 1)

lasso_model <- cv.glmnet(df_x, y, 
                         alpha = 1, lambda = lambdas,
                         standardize = T, nfolds = 5)
plot(lasso_model)
lasso.best_lambda <- lasso_model$lambda.min
best.lasso_model <- glmnet(df_x, y,
                           alpha = 1, lambda = lasso.best_lambda)
summary(best.lasso_model)
lasso_forecast <- predict(object = best.lasso_model,df_x)

lasso.mse <- sum((y - lasso_forecast)^2)/length(y)
lasso.mae <- sum(abs(y - lasso_forecast))/length(y)
plot(auto.lasso, xvar = "lambda", main = "Lasso Regression")

# Plotting the graph of observed vs predicted
lasso_df <- data.frame(dates = recoded[,1], Inflation = lasso_forecast)
names(lasso_df) <- c("Date", "Inflation")
head(lasso_df)

lasso <- df1_ %>% mutate(Type = "Observed") %>% 
  bind_rows(lasso_df %>% 
              mutate(Type = "Lasso Predicted"))
View(lasso)

ggplot(lasso,aes(y = Inflation, x = Date, color = Type)) + 
  geom_line() +
  ggtitle("Observed vs Lasso Regression Pediction")


# Ridge Regression --------------------------------------------------------

auto.ridge <- glmnet(df_x, y, alpha = 0)

ridge_model <- cv.glmnet(df_x, y, 
                         alpha = 0, lambda = lambdas,
                         standardize = T, nfolds = 5)
summary(ridge_model)
plot(ridge_model)
ridge.best_lambda <- ridge_model$lambda.min
best.ridge_model <- glmnet(df_x, y, 
                              alpha = 0, lambda = ridge.best_lambda)
ridge_forecast <- predict(object = best.ridge_model, df_x)

ridge.mse <- sum((y - ridge_forecast)^2)/length(y)
ridge.mae <- sum(abs(y - ridge_forecast))/length(y)
plot(auto.ridge, xvar = "lambda", main = "Ridge Regression")

# Plotting the graph of observed vs predicted
ridge_df <- data.frame(dates = recoded[,1], Inflation = ridge_forecast)
names(ridge_df) <- c("Date", "Inflation")
head(ridge_df)

ridge <- df1_ %>% mutate(Type = "Observed") %>% 
  bind_rows(ridge_df %>% 
              mutate(Type = "Ridge Predicted"))

head(ridge)

ggplot(ridge,aes(y = Inflation, x = Date, color = Type)) + 
  geom_line() +
  ggtitle("Observed vs Ridge Regression Pediction")


# Elastic Net Regression --------------------------------------------------

control <- trainControl(method = "repeatedcv",
                        number = 5,
                        repeats = 5,
                        search = "random",
                        verboseIter = TRUE)

elastic_model <- train(y ~ .,
                       data = cbind(df_x, y),
                       method = "glmnet",
                       preProcess = c("center", "scale"),
                       tuneLength = 25,
                       trControl = control)
elastic_forecast <- predict(object = elastic_model, df_x)

elastic.mse <- sum((y - elastic_forecast)^2)/length(y)
elastic.mae <- sum(abs(y - elastic_forecast))/length(y)
plot(elastic_model)

# Plotting the graph of observed vs predicted
elastic_df <- data.frame(dates = recoded[,1], Inflation = elastic_forecast)
# names(ridge_df) <- c("Date", "Inflation")
head(elastic_df)

elastic <- df1_ %>% mutate(Type = "Observed") %>% 
  bind_rows(elastic_df %>% 
              mutate(Type = "Elastic Net Predicted"))

head(elastic)

ggplot(elastic,aes(y = Inflation, x = Date, color = Type)) + 
  geom_line() +
  ggtitle("Observed vs Elastic Net  Pediction")

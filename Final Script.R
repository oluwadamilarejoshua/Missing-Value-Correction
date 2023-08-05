

# Loading required packages -----------------------------------------------

library(tidyverse)
library(MASS)
library(caret)
library(norm)
library(VIM)
library(ggplot2)
library(naniar)
library(devtools)
library(missMethods)
library(mice)
library(hot.deck)
library(stats)


# Loading data to the environment -----------------------------------------


full_data <- na.omit(read.csv("../Projects/Yomi FUTA/Thesis/basic_income_dataset_dalia.csv"))
str(full_data)

# Separating the y and Xs as well as removing redundant column ------------


y <- full_data[, 10]
X <- full_data[, -c(2, 10)]


for (column in c(1: ncol(X))) {
  if(typeof(X[, column]) == "character"){
    X[, column] <- as.factor(X[, column])
  }
}

X[["weight"]] <- as.numeric(X[["weight"]])

# Reducing the categories of the y to 2 from 5 ----------------------------


for (obs_ in c(1:length(y))){
  if (y[obs_] == "I would vote for it" || y[obs_] == 
      "I would probably vote for it"){
    y[obs_] <- "Yes"
  }
  else{
    y[obs_] <- "No"
  }
}

y <- as.factor(y)
X <- X[, -c(10, 11)]

revamped <- as.data.frame(cbind(y, X))

str(revamped)

#####################------- DUMMY-FY FUNCTION --------########################

dummyfy <- function(data = revamped){
  w = c(2, 11)
  X <- data[, -1]
  y <- data[, 1]
  dummy_data <- dummyVars(~ country_code + rural + gender + dem_education_level + 
                            dem_full_time_job + dem_has_children +
                            question_bbi_2016wave4_basicincome_awareness + 
                            question_bbi_2016wave4_basicincome_effect + 
                            age_group, data = X)
  dummy_data <- predict(dummy_data, newdata = X)
  the_X <- as.data.frame(cbind(y, dummy_data, X[, w]))
  return(the_X)
}

################--------------------------------------##################




initial_data <- dummyfy(data = revamped)
str(initial_data)




#########----------------------- MODEL FUNCTION -----------------------########

model_evaluate <- function(initial_data = initial_data){
  set.seed(123)
  trainIndices <- sample(1:nrow(initial_data),
                         floor(0.7*nrow(initial_data)))
  trainData <- initial_data[trainIndices, ]
  testData <- initial_data[-trainIndices, ]
  
  model <- glm(y ~ ., data = trainData, family = "binomial")
  
  predictions_prob <- data.frame(predict(model, newdata = testData, type = "response"))
  
  predictions <- predictions_prob %>%
    mutate(predicted = ifelse(predictions_prob > .55, "Yes", "No"))
  
  predictions <- as.data.frame(predictions)
  
  predicted <- as.factor(predictions$predicted)
  
  model_performance <- confusionMatrix(predicted, testData$y)
  
  return(list(model, predictions_prob, predicted, model_performance))
}

###########----------------------------------------------------################


preliminary_model <- model_evaluate(initial_data = initial_data)
pre_coeff <- coefficients(preliminary_model[[1]])
abs_pre_coeff <- abs(pre_coeff)
sorted_pre_coeff <- sort(abs_pre_coeff, decreasing = T)
sorted_names <- names(pre_coeff)[order(abs(pre_coeff), decreasing = TRUE)]
sorted_coefficients <- sorted_pre_coeff[order(abs(pre_coeff), decreasing = TRUE)]
preliminary_model[[4]]


######## Missing Data Plot function -------------------
set.seed(123)

make_simple_MDplot <- function(ds_comp = revamped, ds_mis = revamped) {
  ds_comp$missX <- is.na(ds_mis$`question_bbi_2016wave4_basicincome_awareness`)
  ggplot(ds_comp, aes(x = age, y = weight, col = missX)) +
    geom_point()
}


# Creating the missing values ---------------------------------------------

# Creating missing values using MAR method
x_mar_10 <- delete_MAR_censoring(X, 0.1, colnames(X)[8], colnames(X)[9])
y_mar_10 <- as.data.frame(cbind(y, x_mar_10))
make_simple_MDplot(ds_mis = y_mar_10)

x_mar_20 <- delete_MAR_censoring(X, 0.2, colnames(X)[8], colnames(X)[9])
y_mar_20 <- as.data.frame(cbind(y, x_mar_20))
make_simple_MDplot(ds_mis = y_mar_20)

x_mar_50 <- delete_MAR_censoring(X, 0.5, colnames(X)[8], colnames(X)[9])
y_mar_50 <- as.data.frame(cbind(y, x_mar_50))
make_simple_MDplot(ds_mis = y_mar_50)

# Creating missing values using MCAR method
x_mcar_10 <- delete_MCAR(X, .1, colnames(X)[8])
y_mcar_10 <- as.data.frame(cbind(y, x_mcar_10))
make_simple_MDplot(ds_mis = y_mcar_10)

x_mcar_20 <- delete_MCAR(X, .2, colnames(X)[8])
y_mcar_20 <- as.data.frame(cbind(y, x_mcar_20))
make_simple_MDplot(ds_mis = y_mcar_20)

x_mcar_50 <- delete_MCAR(X, .5, colnames(X)[8])
y_mcar_50 <- as.data.frame(cbind(y, x_mcar_50))
make_simple_MDplot(ds_mis = y_mcar_50)


# Inserting Missing Values by the Imputation methods ----------------------

# Complete case analysis (CCA) ---

df_cca_mar_10 <- na.omit(y_mar_10)
df_cca_mar_20 <- na.omit(y_mar_20)
df_cca_mar_50 <- na.omit(y_mar_50)

df_cca_mcar_10 <- na.omit(y_mcar_10)
df_cca_mcar_20 <- na.omit(y_mcar_20)
df_cca_mcar_50 <- na.omit(y_mcar_50)


# Multiple imputations (MI) ---

imp_mar_10 <- mice(y_mar_10, m = 2)
df_mi_mar_10 <- complete(imp_mar_10)

imp_mar_20 <- mice(y_mar_20, m = 2)
df_mi_mar_20 <- complete(imp_mar_20)

imp_mar_50 <- mice(y_mar_50, m = 2)
df_mi_mar_50 <- complete(imp_mar_50)


imp_mcar_10 <- mice(y_mcar_10, m = 2)
df_mi_mcar_10 <- complete(imp_mcar_10)

imp_mcar_20 <- mice(y_mcar_20, m = 2)
df_mi_mcar_20 <- complete(imp_mcar_20)

imp_mcar_50 <- mice(y_mcar_50, m = 2)
df_mi_mcar_50 <- complete(imp_mcar_50)
str(df_mi_mcar_50)

#------------------------------------------------------
df_mi_mcar_50_dummyfied <- dummyfy(data = df_mi_mcar_50)
model_evaluate(initial_data = df_mi_mcar_50_dummyfied)
#------------------------------------------------------


# Hot deck imputation (HD) ---

hd_10 <- hot.deck(y_mar_10)[["data"]]
df_hd_mar_10 <- hd_10[[1]]
str(df_hd_mar_10)
df_hd_mar_20 <- hot.deck(y_mar_20)[["data"]][[1]]
df_hd_mar_50 <- hot.deck(y_mar_50)[["data"]][[1]]

df_hd_mcar_10 <- hot.deck(y_mcar_10)[["data"]][[1]]
df_hd_mcar_20 <- hot.deck(y_mcar_20)[["data"]][[1]]
df_hd_mcar_50 <- hot.deck(y_mcar_50)[["data"]][[1]]



#------------------------------------------------------
df_hd_mar_10_dummyfied <- dummyfy(data = df_hd_mar_10)
model_evaluate(initial_data = df_hd_mar_10_dummyfied)
#------------------------------------------------------


# KNN Imputation ---

df_knn_mar_10 <- kNN(y_mar_10, 
                     variable = "question_bbi_2016wave4_basicincome_awareness")
summary(df_knn_mar_10)
df_knn_mar_20 <- kNN(y_mar_20, 
                     variable = "question_bbi_2016wave4_basicincome_awareness")
summary(df_knn_mar_20)
df_knn_mar_50 <- kNN(y_mar_50, 
                     variable = "question_bbi_2016wave4_basicincome_awareness")
summary(df_knn_mar_50)


df_knn_mcar_10 <- kNN(y_mcar_10, 
                     variable = "question_bbi_2016wave4_basicincome_awareness")
summary(df_knn_mcar_10)
df_knn_mcar_20 <- kNN(y_mcar_20, 
                     variable = "question_bbi_2016wave4_basicincome_awareness")
summary(df_knn_mcar_20)
df_knn_mcar_50 <- kNN(y_mcar_50, 
                     variable = "question_bbi_2016wave4_basicincome_awareness")
summary(df_knn_mcar_50)



# Modeling and Evaluation of performance ----------------------------------


  # Complete Case Analysis

model_cca_mar_10 <- 
  model_evaluate(initial_data = dummyfy(data = df_cca_mar_10))
model_cca_mar_10[[4]] -> cca_mar_10
model_cca_mar_20 <- 
  model_evaluate(initial_data = dummyfy(data = df_cca_mar_20))
model_cca_mar_20[[4]] -> cca_mar_20
model_cca_mar_50 <- 
  model_evaluate(initial_data = dummyfy(data = df_cca_mar_50))
model_cca_mar_50[[4]] -> cca_mar_50

model_cca_mcar_10 <- 
  model_evaluate(initial_data = dummyfy(data = df_cca_mcar_10))
model_cca_mcar_10[[4]] -> cca_mcar_10
model_cca_mcar_20 <- 
  model_evaluate(initial_data = dummyfy(data = df_cca_mcar_20))
model_cca_mcar_20[[4]] -> cca_mcar_20
model_cca_mcar_50 <- 
  model_evaluate(initial_data = dummyfy(data = df_cca_mcar_50))
model_cca_mcar_50[[4]] -> cca_mcar_50


  # Multiple Imputation

model_mi_mar_10 <- 
  model_evaluate(initial_data = dummyfy(data = df_mi_mar_10))
model_mi_mar_10[[4]] -> mi_mar_10
model_mi_mar_20 <- 
  model_evaluate(initial_data = dummyfy(data = df_mi_mar_20))
model_mi_mar_20[[4]] -> mi_mar_20
model_mi_mar_50 <- 
  model_evaluate(initial_data = dummyfy(data = df_mi_mar_50))
model_mi_mar_50[[4]] -> mi_mar_50

model_mi_mcar_10 <- 
  model_evaluate(initial_data = dummyfy(data = df_mi_mcar_10))
model_mi_mcar_10[[4]] -> mi_mcar_10
model_mi_mcar_20 <- 
  model_evaluate(initial_data = dummyfy(data = df_mi_mcar_20))
model_mi_mcar_20[[4]] -> mi_mcar_20
model_mi_mcar_50 <- 
  model_evaluate(initial_data = dummyfy(data = df_mi_mcar_50))
model_mi_mcar_50[[4]] -> mi_mcar_50


  # Hot Deck Imputation

model_hd_mar_10 <- 
  model_evaluate(initial_data = dummyfy(data = df_hd_mar_10))
model_hd_mar_10[[4]] -> hd_mar_10
model_hd_mar_20 <- 
  model_evaluate(initial_data = dummyfy(data = df_hd_mar_20))
model_hd_mar_20[[4]] -> hd_mar_20
model_hd_mar_50 <- 
  model_evaluate(initial_data = dummyfy(data = df_hd_mar_50))
model_hd_mar_50[[4]] -> hd_mar_50

model_hd_mcar_10 <- 
  model_evaluate(initial_data = dummyfy(data = df_hd_mcar_10))
model_hd_mcar_10[[4]] -> hd_mcar_10
model_hd_mcar_20 <- 
  model_evaluate(initial_data = dummyfy(data = df_hd_mcar_20))
model_hd_mcar_20[[4]] -> hd_mcar_20
model_hd_mcar_50 <- 
  model_evaluate(initial_data = dummyfy(data = df_hd_mcar_50))
model_hd_mcar_50[[4]] -> hd_mcar_50


  # KNN Imputation

model_knn_mar_10 <- 
  model_evaluate(initial_data = dummyfy(data = df_knn_mar_10))
model_knn_mar_10[[4]] -> knn_mar_10
model_knn_mar_20 <- 
  model_evaluate(initial_data = dummyfy(data = df_knn_mar_20))
model_knn_mar_20[[4]] -> knn_mar_20
model_knn_mar_50 <- 
  model_evaluate(initial_data = dummyfy(data = df_knn_mar_50))
model_knn_mar_50[[4]] -> knn_mar_50

model_knn_mcar_10 <- 
  model_evaluate(initial_data = dummyfy(data = df_knn_mcar_10))
model_knn_mcar_10[[4]] -> knn_mcar_10
model_knn_mcar_20 <- 
  model_evaluate(initial_data = dummyfy(data = df_knn_mcar_20))
model_knn_mcar_20[[4]] -> knn_mcar_20
model_knn_mcar_50 <- 
  model_evaluate(initial_data = dummyfy(data = df_knn_mcar_50))
model_knn_mcar_50[[4]] -> knn_mcar_50


# Table of the results ----------------------------------------------------

mar_call <- rep("MAR", times = 4)
mcar_call <- rep("MCAR", times = 4)
m_type <- c(rep(c(mar_call,mcar_call), times = 3))

m_percent <- c(rep("10", times = 8), rep("20", times = 8), rep("50", times = 8))

imput_meth <- c(rep(c("CCA", "MI", "HD", "KNN"), times = 6))

result_tab <- data.frame(m_type, m_percent, imput_meth, c(rep(0, times= 24)),
                         c(rep(0, times= 24)), c(rep(0, times= 24)))
colnames(result_tab) <- c("Missing Type", "Missing Percentatge",
                          "Imputation Method", "Accuracy", "Sensitivity",
                          "Specificity")
head(result_tab)

the_list <- 
  list(cca_mar_10, mi_mar_10, hd_mar_10, knn_mar_10, cca_mcar_10, mi_mcar_10,
       hd_mcar_10, knn_mcar_10,cca_mar_20, mi_mar_20, hd_mar_20, knn_mar_20,
       cca_mcar_20, mi_mcar_20, hd_mcar_20, knn_mcar_20,cca_mar_50, mi_mar_50,
       hd_mar_50, knn_mar_50, cca_mcar_50, mi_mcar_50, hd_mcar_50, knn_mcar_50)

accuracy <- c()
sensitivity <- c()
specificity <- c()

for(i in 1:24){
  assa <- the_list[[i]]
  acc <- assa[[3]][[1]]
  sens <- assa[[4]][[1]]
  spec_ <- assa[[4]][[2]]
  
  accuracy[i] <- acc
  sensitivity[i] <- sens
  specificity[i] <- spec_
}

result_tab$Accuracy <- accuracy
result_tab$Sensitivity <- sensitivity
result_tab$Specificity <- specificity

head(result_tab)
result_tab
# write.csv(result_tab, file = "result.csv")




########-------------------- PART 2 ---------------------###########


# Evaluation of Methods' performances -------------------------------------

CCA_performance_data <- 
  result_tab %>% filter(`Imputation Method` == "CCA")

MI_performance_data <- 
  result_tab %>% filter(`Imputation Method` == "MI")

HD_performance_data <- 
  result_tab %>% filter(`Imputation Method` == "HD")

KNN_performance_data <- 
  result_tab %>% filter(`Imputation Method` == "KNN")

summ_eval <- function(data_ = CCA_performance_data){
  acc = data_$Accuracy
  senc = data_$Sensitivity
  spec = data_$Specificity
  
  acc_sum <- c(mean(acc), sd(acc))
  senc_sum <- c(mean(senc), sd(senc))
  spec_sum <- c(mean(spec), sd(spec))
  
  summa_rized <- cbind(acc_sum, senc_sum, spec_sum)
  row.names(summa_rized) <- c("Mean", "Standard Deviattion")
  colnames(summa_rized) <- c("Accuracy", "Sensitivity", "Specificity")
  return(summa_rized)
}

CCA_Summary <- summ_eval()
CCA_Summary
MI_Summary <- summ_eval(data_ = MI_performance_data)
MI_Summary
HD_Summary <- summ_eval(data_ = HD_performance_data)
HD_Summary
KNN_Summary <- summ_eval(data_ = KNN_performance_data)
KNN_Summary


# Analysis of Variance (ANOVA) --------------------------------------------

acc_anova <- aov(Accuracy ~ `Imputation Method`, data = result_tab)
summary(acc_anova)

sens_anova <- aov(Sensitivity ~ `Imputation Method`, data = result_tab)
summary(sens_anova)

spec_anova <- aov(Specificity ~ `Imputation Method`, data = result_tab)
summary(spec_anova)


# Post-Hoc Test -----------------------------------------------------------

acc_poctHoc_PWT <- pairwise.t.test(result_tab$Accuracy, 
                               result_tab$`Imputation Method`,
                               p.adjust.method = "bonferroni")
acc_poctHoc_PWT
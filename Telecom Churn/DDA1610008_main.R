library(ggplot2)
library(h2o)
library(caTools)

#choose the training set
churn <- read.csv("telecom_nn_train.csv", stringsAsFactors = F)
str(churn)

#attributes have been feature engineered.

#checking for outliers among the continous variables
boxplot(churn$MonthlyCharges)
boxplot(churn$tenure)
boxplot(churn$TotalCharges)

#check for duplicated data
which(duplicated(churn))
churn <- churn[-(which(duplicated(churn))),]

#check for missing values or NAs
sum(is.na(churn))
#6 values. Let's find out how they are distributed among various columns
sapply(churn, function(x)
  sum(is.na(x)))
# All 6 are present in total charges column. Let's see if we can replace these with the mean
summary(as.numeric(churn$TotalCharges))
churn$TotalCharges[which(is.na(churn$TotalCharges))] <-
  mean(churn$TotalCharges, na.rm = TRUE)

#Now lets change the formats of various columns accordingly
str(churn)
#monthly charges, total charges and tenure to be changed to numeric
churn[, c(5, 8, 9)] <-
  sapply(churn[, c(5, 8, 9)], function(x)
    as.numeric(x))




############################################################################
# EDA Plots to explore potential interesting relationships between variables
############################################################################
ggplot(churn, aes(MonthlyCharges, fill = Churn)) + geom_histogram(binwidth = 10) +
  labs(title = "Impact of Monthly Charges on Churn", x = "Monthly Charges", y = "Frequency")
ggplot(churn, aes(tenure, fill = Churn)) + geom_histogram(binwidth = 5) +
  labs(title = "Impact of Tenure on Churn", x = "Tenure", y = "Frequency") #Important
ggplot(churn, aes(TotalCharges, fill = Churn)) + geom_histogram(binwidth =
                                                                  1000) + labs(title = "Impact of Total Charges on Churn", x = "Total Charges", y = "Frequency")  #Important

ggplot(churn, aes(as.factor(PhoneService), fill = Churn)) + geom_bar() +
  labs(title = "Impact of Phone Service on Churn", x = "Phone Service", y = "Frequency") +
  scale_x_discrete(labels = c("No", "Yes"))
ggplot(churn, aes(as.factor(gender), fill = Churn)) + geom_bar() + labs(title =
                                                                          "Impact of Gender on Churn", x = "Gender", y = "Frequency") + scale_x_discrete(labels =
                                                                                                                                                           c("Male", "Female"))
ggplot(churn, aes(factor(SeniorCitizen), fill = Churn)) + geom_bar() + labs(title =
                                                                              "Impact of Senior Citizen on Churn", x = "Senior Citizen", y = "Frequency")

# From the plots it is evident that the dataset is biased. There are fewer datapoints
# which indicate churn and larger number of datapoints which indicate no churn.

#We'll divide the data set into our testing and training samples in the ratio of 3:7 respectively
#From the plots we could see that the data was skewed. Hence we split the data accordingly
summary(factor(churn$Churn))


churn$Churn <- as.factor(churn$Churn)
str(churn)
rows <- sample.split(churn$Churn, SplitRatio = 0.7)
churn_training <- churn[rows,]
summary(as.factor(churn_training$Churn)) 

# Split the remaining data into validation and testing data.
# first extract the remaining data and then split that data
churn_test <- churn[!rows,]

summary(as.factor(churn_test$Churn))

# initialize jvm
h2o.init(nthreads = -1)
columns <- colnames(churn[,-10])
# Convert he data frame to an h2o object
churn_training <- as.h2o(churn_training)
churn_test <- as.h2o(churn_test)


#################################################
#
#             Without Epochs
#
#################################################
EPOCHS = 1 # we use just one pass.

#For the first pass, we'll try different values of hyperparameters to see which ones have
# a larger effect on the accuracy.

#let's run the deeplearning function with default values
# We will use cross validation to train the model across the entire data set starting with nfolds value of 10



initial_model <- h2o.deeplearning(
  x = columns,
  y = c("Churn"),
  training_frame = churn_training,
  epochs = EPOCHS,
  nfolds = 10,
  seed = 5000
)
print(initial_model)
# We can observe a few default values from this
# 1. Loss function is Cross entropy for classification problems
# 2. Default number of hidden layers is 2 with 200 neurons each
# 3. Activation function is of type rectifier (Default)
# 4. Hidden dropout ratio is 0%
# 5. l1 and l2 rates are 0


# Let's tweak these default values. 
#  

args <- list(
  list(
    hidden = c(200, 200, 200), # add one more layer
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.1, 0.1, 0.1), # increase to 10%
    activation = "RectifierWithDropout",    # change from rectifier to RectifierWithDropout
    epochs = EPOCHS
  ),
  list(
    hidden = c(200, 200, 200),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.2, 0.2, 0.2),
    activation = "RectifierWithDropout",
    epochs = EPOCHS
  ),
  list(
    hidden = c(400, 400, 400),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.1, 0.1, 0.1),
    activation = "RectifierWithDropout",
    epochs = EPOCHS
  ),
  list(
    hidden = c(200, 200, 200, 200),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.1, 0.1, 0.1, 0.1),
    activation = "RectifierWithDropout",
    epochs = EPOCHS
  ),
  list(
    hidden = c(200, 200, 200),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.1, 0.1, 0.1),
    activation = "TanhWithDropout",
    epochs = EPOCHS
  )
)

run <- function(extra_params) {
  model <-
    do.call(h2o.deeplearning, modifyList(
      list(
        x = columns,
        y = c("Churn"),
        nfolds = 10,
        distribution = "multinomial",
        seed = 5000,
        l1 = 1e-5,
        balance_classes = TRUE,
        training_frame = churn_training,
        reproducible=TRUE
      ),
      extra_params
    ))
  return(model)
  
}

without_epochs <- lapply(args, run)

#Print each model to evalualte the best

  for (computed_model in without_epochs) {

     print(computed_model@model$cross_validation_metrics)
     cat("\n")
    
  }
  



# Models 1,2 and 4 have lowest log loss as compared to others
# Comparing the RMSE and AUC, it appears that model 4 would be less effective at predicting
# looking at the cross validation results, we can indeed see that it tends to
# have high error rate by classifying most of the customers as people who would churn
# This looks like an overfit and hence we will discard this model
# We'll pick the hyper-params from model 1 and 2


# Let's keep the number of neurons same as model 1 /model 2 i.e.
# Since 4 layers of 200 weren't a good fit, let's try adding another layer just to be sure
# Since tanh with 3 layers of 200 wasn't a good fit, let's also check by increasing the number
# of layers and check the results.

args <- list(
    list(
    hidden = c(200, 200, 200, 200, 200),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.1, 0.1, 0.1, 0.1, 0.1), # inccrease layers
    activation = "RectifierWithDropout",
    epochs = EPOCHS
  ),
  list(
    hidden = c(200, 200, 200, 200, 200),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.1, 0.1, 0.1, 0.1, 0.1),
    activation = "TanhWithDropout", # Different Activation function
    epochs = EPOCHS
  ),
  list(
    hidden = c(200, 200, 200, 200, 200),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.1, 0.1, 0.1, 0.1, 0.1),
    activation = "MaxoutWithDropout", # Different activation function
    epochs = EPOCHS
  )
)

without_epochs_2 <- lapply(args, run)
without_epochs_2



#Print each model to evalualte the best

for (computed_model in without_epochs_2) {
  
  print(computed_model@model$cross_validation_metrics)
  cat("\n")
  
}


# Model 1 seems to be the best among this second set as it has the least log loss and RMSE
# It also does a fairly good job of predicting churn accurately.
# However, it is almost comparable to model 1 and model 2 of the previous iteration
# The addition of one more layer of neurons did not have a significant impact
# Also, Rectifier with output seems to be the most suitable activation function

# Keeping this in mind, let's try to reduce the number of neurons in each layer
# We will keep the number of layers as 3 only. 


args <- list(
  list(
    hidden = c(100, 100, 100 ),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.1, 0.1, 0.1),
    activation = "RectifierWithDropout",
    epochs = EPOCHS
  ),
  list(
    hidden = c(30, 30, 30 ), 
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.1, 0.1, 0.1),
    activation = "RectifierWithDropout",
    epochs = EPOCHS
  ),
  list(
    hidden = c(50, 50, 50 ), 
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.1, 0.1, 0.1),
    activation = "RectifierWithDropout",
    epochs = EPOCHS
  )

)

without_epochs_3 <- lapply(args, run)



#Print each model to evalualte the best

for (computed_model in without_epochs_3) {
  
  print(computed_model@model$cross_validation_metrics)
  cat("\n")
  
}


# The models from reduced neurons seem to have similar overall accuracy
# However, the percentage of actual churn predicted correctly has dropped, while the 
# accurate non churn percentage has increased. This makes it slightly biased towards the
# optimistic side.

# When compared to model 1 and 2 from first iteration and model 1 of second iteration,
# we observe that first iteration was more negatively biased, i.e. greater tendency to 
# classify a customer as someone who will churn.

# The error rate of model 2 for both types of customers is very close to the best 
# specificity and sensitivity of any other model so far. Thus this model seems to have
# a good learning and doesn't seem to overfit. We choose this as the best model without epochs

# Hyper parameters for the best model without epochs

# hidden = c(200, 200, 200, 200),
# loss = "CrossEntropy",
# hidden_dropout_ratio = c(0.1, 0.1, 0.1, 0.1),
# activation = "RectifierWithDropout",
# nfolds = 10,
# distribution = "multinomial",
# l1 = 1e-5


best_model_without_epochs <- without_epochs[2]


# Let's validate it against our test data

for (computed_model in without_epochs[2]) {

  perform <- h2o.performance(model = computed_model,
                             newdata = churn_test)
  print(perform)
}

# Chrun prediction accuracy of 65.5% . 


# ####################################################
#
#             With Epochs
#
# ###################################################

# Let's start with the same initial state as before except that we'll epochs =2
# We'll adopt grid search to get best model
EPOCHS = 2
response <- "Churn"
predictors <- setdiff(names(churn), response)



# Let's start with random values for various hyper-parameters
# We will use activation functions with dropout since we're also using hidden ratio dropout
# we'll also experiment with various values of epochs


args <- list(
  list(
    hidden = c(32, 32, 32, 32),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.1, 0.1, 0.1, 0.1),
    activation = "RectifierWithDropout",
    epochs = 20
  ),
  list(
    hidden = c(32, 32, 32, 32, 32),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.1, 0.1, 0.1, 0.1, 0.1),
    activation = "TanhWithDropout",
    epochs = 10
  ),
  list(
    hidden = c(32, 32, 32, 32, 32),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.1, 0.1, 0.1, 0.1, 0.1),
    activation = "TanhWithDropout",
    epochs = 20
  ),
  list(
    hidden = c(40, 40, 40, 40, 40), #Consider
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.1, 0.1, 0.1, 0.1, 0.1),
    activation = "RectifierWithDropout",
    epochs = 10
  ),
  list(
    hidden = c(32, 32, 32, 32, 32),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.15, 0.15, 0.15, 0.15, 0.15),
    activation = "RectifierWithDropout",
    epochs = 20
  ),
  list(
    hidden = c(32, 32, 32, 32, 32),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.15, 0.15, 0.15, 0.15, 0.15),
    activation = "TanhWithDropout",
    epochs = 10
  ),
  list(
    hidden = c(32, 32, 32, 32, 32), #consider
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.15, 0.15, 0.15, 0.15, 0.15),
    activation = "TanhWithDropout",
    epochs = 20
  ),
  list(
    hidden = c(40, 40, 40, 40, 40), #consider
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.15, 0.15, 0.15, 0.15, 0.15),
    activation = "RectifierWithDropout",
    epochs = 10
  ),
  list(
    hidden = c(32, 32, 32, 32, 32), #consider
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.05, 0.05, 0.05, 0.05, 0.05),
    activation = "RectifierWithDropout",
    epochs = 20
  ),
  list(
    hidden = c(32, 32, 32, 32, 32),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.05, 0.05, 0.05, 0.05, 0.05),
    activation = "TanhWithDropout",
    epochs = 10
  ),
  list(
    hidden = c(32, 32, 32, 32, 32),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.05, 0.05, 0.05, 0.05, 0.05),
    activation = "TanhWithDropout",
    epochs = 20
  ),
  list(
    hidden = c(40, 40, 40, 40, 40),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.05, 0.05, 0.05, 0.05, 0.05),
    activation = "RectifierWithDropout",
    epochs = 10
  )
  
)
run <- function(extra_params) {
  model <-
    do.call(h2o.deeplearning, modifyList(
      list(
        x = predictors,
        y = response,
        nfolds = 5,
        distribution = "multinomial",
        seed = 5000,
        l1 = 1e-5,
        balance_classes = TRUE,
        reproducible=TRUE,
        training_frame = churn_training
      ),
      extra_params
    ))
  return(model)
}

with_epochs_1 <- lapply(args, run)



#Print each model to evalualte the best

for (computed_model in with_epochs_1) {
   print(computed_model@model$cross_validation_metrics)
   cat("======================\n")
   
}



# Model 11,10,4, 3 and  2 show the best balance in terms of predicting churns and non churns
# correctly. They strike a good balance between log loss, AUC, RMSE and churn accuracy among all  


# The models have performed best with neurons 32 or 40 and 5 layers. A majority of the 
# best performing models have used rectifierwithdropout/ tanhwithdroput as activation and as there is no
# significant difference between 10 epochs and 20 epochs

# Let's try reducing epochs
# Hidden dropout ratio will be between 0.05 and 0.15 
# 


args <- list(
  
  list(
    hidden = c(40,40,40,40,40),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.18, 0.18, 0.18, 0.18,0.18),
    activation = "RectifierWithDropout",
    epochs = 5
  ),
  list(
    hidden = c(40,40,40,40,40),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.12, 0.12, 0.12, 0.12,0.12),
    activation = "RectifierWithDropout",
    epochs = 5
  ),
  list(
    hidden = c(40,40,40,40,40),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.15, 0.15, 0.15, 0.15,0.15),
    activation = "RectifierWithDropout",
    epochs = 5
  )
  
)

with_epochs_2 <- lapply(args, run)

for (computed_model in with_epochs_2) {
 print(computed_model@model$cross_validation_metrics)
}

# Changing the number of neurons and epochs has dropped churn accuracy but increased non churn accracy
# Let's try by increasing the number of neurons further by a multiple of the inputs

args <- list(
  list(
    hidden = c(60,60,60,60),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.12, 0.12, 0.12, 0.12),
    activation = "RectifierWithDropout",
    epochs = 5
  ),
  list(
    hidden = c(60,60,60,60,60),
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.12, 0.12, 0.12, 0.12,0.12),
    activation = "RectifierWithDropout",
    epochs = 5
  )
  
)

with_epochs_3 <- lapply(args, run)

for (computed_model in with_epochs_3) {
  print(computed_model@model$cross_validation_metrics)
}

# Its almost comparable to model 4 of our first pass
# Let's increase the number of k fold validations

run <- function(extra_params) {
  model <-
    do.call(h2o.deeplearning, modifyList(
      list(
        x = predictors,
        y = response,
        nfolds = 10,
        distribution = "AUTO",
        seed = 5000,
        l1 = 1e-5,
        balance_classes = TRUE,
        reproducible=TRUE,
        training_frame = churn_training
      ),
      extra_params
    ))
  return(model)
}


with_epochs_4 <- lapply(args, run)

for (computed_model in with_epochs_4) {
  print(computed_model@model$cross_validation_metrics)
}

# Hasn't made too much of a difference.
# Let's try with a  tweaking dropout ratio for the earlier model

args <- list(
    list(
    hidden = c(40, 40, 40, 40, 40), #Consider
    loss = "CrossEntropy",
    hidden_dropout_ratio = c(0.12, 0.12, 0.12, 0.12, 0.12),
    activation = "RectifierWithDropout",
    epochs = 10
  )
  
)

with_epochs_5 <- lapply(args, run)

for (computed_model in with_epochs_5) {
  print(computed_model@model$cross_validation_metrics)
}


# As compared the model 4 of the first pass the performance on validation data is 
# almost similar. However, the log loss, RMSE, AUC are all slightly better.
# This will therefore be better positioned to handle data.

# Let's test this on our test data

for (computed_model in with_epochs_5) {
  perform <- h2o.performance(model = computed_model,
                             newdata = churn_test)
  print(perform)
}

best_model_with_epochs <- with_epochs_2[2]

# 5 layers of 40 neurons each
# 5 epochs
# activation = rectifierwithdropout
# hidden dropout ratio = 12%
# nfolds = 10
# l1 regularization value 0.000010

# Validation results
# Confusion Matrix for F1-optimal threshold:
#   No  Yes    Error       Rate
# No     2005  547 0.214342  =547/2552
# Yes     224  668 0.251121   =224/892
# Totals 2229 1215 0.223868  =771/3444

# Test data results
# Confusion Matrix for F1-optimal threshold:
#   No Yes    Error       Rate
# No     854 239 0.218664  =239/1093
# Yes    103 280 0.268930   =103/383
# Totals 957 519 0.231707  =342/1476

# Churn accuracy of almost 73%. This is almost 8% better than
# the model without epochs.

h2o.shutdown()

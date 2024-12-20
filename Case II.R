#' Author: Ohm Kundurthy
#' Date: Nov 9 2024
# Purpose: Case 2 - National City Bank Customer Propensity Model development exercise
# Code Organized by SEMMA 

# Libs
library(vtreat)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(scales)
library(tidyr)
library(reshape2)
library(GGally)
library(MLmetrics)
library(pROC)
library(caret)
library(rpart.plot)
options(scipen = 999)

# Setwd
setwd("~/Git/GitHub_R/Cases/Spring/II National City Bank/training")

# Read the data for the case 
currentCustomerMktgResults<- read.csv('CurrentCustomerMktgResults.csv')
householdAxiomData<- read.csv('householdAxiomData.csv')
householdCreditData<- read.csv('householdCreditData.csv')
householdVehicleData<- read.csv('householdVehicleData.csv')
prospectiveCustomers<- read.csv('ProspectiveCustomers.csv')

########################################SAMPLE################################
# Examine the data attributes and list of possible values for each attribute; 
summary(currentCustomerMktgResults)
#Review Target Variable
table(currentCustomerMktgResults$Y_AcceptedOffer) 
# Unbalanced Dataset with more observations of DidNotAccept
summary(householdCreditData)
summary(householdAxiomData)
summary(householdVehicleData)
summary(prospectiveCustomers)

#Join data sets using the key attribute
fullDataSet<- left_join(currentCustomerMktgResults, householdAxiomData, by = c('HHuniqueID'))
fullDataSet<- left_join(fullDataSet, householdCreditData, by = c('HHuniqueID'))
fullDataSet<- left_join(fullDataSet, householdVehicleData, by = c('HHuniqueID'))
names(fullDataSet)

prospectiveCustomersJoined <- left_join(prospectiveCustomers, householdAxiomData, by = c('HHuniqueID'))
prospectiveCustomersJoined<- left_join(prospectiveCustomersJoined, householdCreditData, by = c('HHuniqueID'))
prospectiveCustomersJoined<- left_join(prospectiveCustomersJoined, householdVehicleData, by = c('HHuniqueID'))
names(prospectiveCustomersJoined)
#############################Explore########################################################
#Basic EDA
summary(fullDataSet)
names(fullDataSet)

# Get unique values for the input columns in TrainingSet and the corresponding frequency
GroupBy_fullDataSet <- sapply(fullDataSet[, 3:27], function(x) table(x))

# Print the unique values for each column and their frequency like a group by for each column in sql
print(GroupBy_fullDataSet)

###############################MODIFY########################################################
#Exclude duplicate attributes that represent same information 
#LastContactMonth,LastContactDay & DaysPassed represent same information
fullDataSet <- fullDataSet[, !(names(fullDataSet) %in% c("LastContactMonth", "LastContactDay"))]
prospectiveCustomersJoined <- prospectiveCustomersJoined[, !(names(prospectiveCustomersJoined) %in% c("LastContactMonth", "LastContactDay"))]
names(fullDataSet)
names(prospectiveCustomersJoined)

#exclude High Cardinality categorical Variables for exclusion from model
#as attributes with many unique categories (e.g. job , carMake , CarModel, EstRace)
#lead to a large number of dummy variables which 
#increases the model complexity and risk of over fitting
fullDataSet <- fullDataSet[, !(names(fullDataSet) %in% c("carMake", "carModel","EstRace","Job"))]
prospectiveCustomersJoined <- prospectiveCustomersJoined[, !(names(prospectiveCustomersJoined) %in% c("carMake", "carModel","EstRace","Job"))]
head(fullDataSet)

# Convert CallStart and CallEnd to POSIXct
fullDataSet$CallStart <- as.POSIXct(fullDataSet$CallStart, format = "%H:%M:%S")
head(fullDataSet$CallStart)
fullDataSet$CallEnd <- as.POSIXct(fullDataSet$CallEnd, format = "%H:%M:%S")
head(fullDataSet$CallEnd)
# Calculate the duration in seconds
fullDataSet$Duration <- as.numeric(difftime(fullDataSet$CallEnd, fullDataSet$CallStart, units = "secs"))

# Assign NA as the duration in seconds
prospectiveCustomersJoined$Duration <- NA

# Check the result
head(fullDataSet$Duration)
head(prospectiveCustomersJoined$Duration)

#Exclude duplicate attributes that represent same informaiton 
fullDataSet <- fullDataSet[, !(names(fullDataSet) %in% c("CallStart", "CallEnd"))]
names(fullDataSet)
names(prospectiveCustomersJoined)

# second check if any input variables have any unidentified high Correlation pairs

#Build Correlation heatmap of numerical variables
# Step 1: Calculate correlation matrix
cor_matrix <- cor(fullDataSet %>% select_if(is.numeric))

# Step 2: Identify highly correlated variable pairs (with threshold > 0.90)
threshold <- 0.7
high_corr_pairs <- which(abs(cor_matrix) > threshold, arr.ind = TRUE)

# Step 3: Extract column names of highly correlated pairs
high_corr_vars <- data.frame(
  Var1 = rownames(cor_matrix)[high_corr_pairs[, 1]],
  Var2 = colnames(cor_matrix)[high_corr_pairs[, 2]],
  Correlation = cor_matrix[high_corr_pairs]
)

# Remove duplicates (the matrix is symmetric, so each pair will appear twice)
high_corr_vars <- high_corr_vars %>%
  filter(Var1 != Var2) %>%
  distinct()

# Display highly correlated variable pairs
print(high_corr_vars)

# Step 4: Optional - Visualize the correlation heatmap 
cor_melted <- melt(cor_matrix)
cor_melted
ggplot(cor_melted, aes(Var1, Var2, fill = value)) + 
  geom_tile() + 
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  labs(title = "Correlation Heatmap") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 3)

# Step 5: Exclude highly correlated variables #none left
# Here, choose to exclude one variable from each pair
# no domain knowledge emphasis on which one is excluded 
#vars_to_exclude <- unique(high_corr_vars$Var2)
#CorTreatedTrainingSet <- treatedTrainingSet %>% select(-all_of(vars_to_exclude))
#CorTreatedValidationSet <- treatedValidationSet %>% select(-all_of(vars_to_exclude))
# Check remaining columns
#head(CorTreatedTrainingSet)
#print(names(CorTreatedTrainingSet))

###########SAMPLE##################################
###Begin pre-processing###
# Examine names for vtreat usage on trainingset
# Partitioning; get 10% test set
splitPercent <- round(nrow(fullDataSet) %*% .9)

set.seed(12345)
idx      <- sample(1:nrow(fullDataSet), splitPercent)
trainingSet <- fullDataSet[idx, ]
ValidationSet  <- fullDataSet[-idx, ]

names(trainingSet)
inputVariables <- names(trainingSet)[c(3:7, 9:22)]
inputVariables

targetVariable      <- names(trainingSet)[8]
targetVariable

# Examine the levels of Y_AcceptedOffer
levels(as.factor(trainingSet$Y_AcceptedOffer))

table(trainingSet$Y_AcceptedOffer)
table(ValidationSet$Y_AcceptedOffer)
trainingSet$Y_AcceptedOffer <- ifelse(trainingSet$Y_AcceptedOffer == "Accepted", 1, 0)
ValidationSet$Y_AcceptedOffer <- ifelse(ValidationSet$Y_AcceptedOffer == "Accepted", 1, 0)
table(trainingSet$Y_AcceptedOffer)
table(ValidationSet$Y_AcceptedOffer)

# Declare the correct level for success for the use case
successClass        <- 1

# Create a treatment plan using the training set
# for Automated variable processing using vtreat 
# for **categorical** outcomes using designTreatmentsC
plan <- designTreatmentsC(trainingSet, 
                          inputVariables,
                          targetVariable, 
                          successClass)

#  Apply the treatment plan to the training set
treatedTrainingSet <- prepare(plan, trainingSet)

names(treatedTrainingSet)


# Ensure that the unique key (ID) is retained in the treated dataset
# The treated dataset lost the ID column, so add it back
treatedTrainingSet$dataID <- trainingSet$dataID
treatedTrainingSet$HHuniqueID <- trainingSet$HHuniqueID
names(treatedTrainingSet)

#  Apply the same treatment plan & addition of ID to the validation set
treatedValidationSet <- prepare(plan, ValidationSet)
treatedValidationSet$dataID <- ValidationSet$dataID
treatedValidationSet$HHuniqueID <- ValidationSet$HHuniqueID
names(treatedValidationSet)

#  Apply the same treatment plan & addition of ID to the prospects set
nrow(prospectiveCustomersJoined)
treatedProspectivesSet <- prepare(plan, prospectiveCustomersJoined)
treatedProspectivesSet$dataID <- prospectiveCustomersJoined$dataID
treatedProspectivesSet$HHuniqueID <- prospectiveCustomersJoined$HHuniqueID
names(treatedProspectivesSet)

summary(treatedTrainingSet)
summary(treatedValidationSet)
summary(treatedProspectivesSet)

###################################EXPLORE##########################################

names(treatedTrainingSet)

# Get unique values for the first 97 columns in treatedTrainingSet and the corresponding frequency
GroupBy_treatedTrainingSet <- sapply(treatedTrainingSet[, 1:42], function(x) table(x))

# Print the unique values for each column and their frequency like a group by for each column in sql
print(GroupBy_treatedTrainingSet)


# Set a uniform color palette for all visualizations
theme_set(theme_light(base_size = 14) + theme(
  strip.background = element_blank(),
  strip.text = element_text(face = "bold", size = 12),
  legend.position = "bottom",
  legend.title = element_blank(),
  legend.key = element_blank()
))

# Define a uniform color palette
color_palette <- c("#FF6F61", "#6B8E23", "#00BFFF", "#FF6347", "#8A2BE2", "#FFD700")

# Visualization 1: Distribution of the target variable 'Y_AcceptedOffer'
ggplot(treatedTrainingSet, aes(x = factor(Y_AcceptedOffer), fill = factor(Y_AcceptedOffer))) +
  geom_bar() +
  labs(title = "Distribution of Target Variable: Y_AcceptedOffer", x = "Accepted Offer", y = "Count") +
  scale_fill_manual(values = color_palette[1:2])

# Visualization 2: Age distribution of applicants
ggplot(treatedTrainingSet, aes(x = Age, fill = factor(Y_AcceptedOffer))) +
  geom_histogram(bins = 30, alpha = 0.7) +
  labs(title = "Age Distribution by Y_AcceptedOffer", x = "Age", y = "Frequency") +
  scale_fill_manual(values = color_palette[1:2])

# Visualization 3: Job distribution by target variable - optional excluded
#ggplot(treatedTrainingSet, aes(x = Job_catP, fill = factor(Y_AcceptedOffer))) +
# geom_bar() +
# labs(title = "Job Distribution by Y_AcceptedOffer", x = "Job Category", y = "Count") +
# scale_fill_manual(values = color_palette[1:2])

# Visualization 4: Marital status by target variable
ggplot(treatedTrainingSet, aes(x = Marital_catP, fill = factor(Y_AcceptedOffer))) +
  geom_bar() +
  labs(title = "Marital Status by Y_AcceptedOffer", x = "Marital Status", y = "Count") +
  scale_fill_manual(values = color_palette[1:2])

# Visualization 5: Relationship between 'NoOfContacts' and 'Y_AcceptedOffer'
ggplot(treatedTrainingSet, aes(x = NoOfContacts, fill = factor(Y_AcceptedOffer))) +
  geom_histogram(bins = 30, alpha = 0.7) +
  labs(title = "Number of Contacts by Y_AcceptedOffer", x = "Number of Contacts", y = "Frequency") +
  scale_fill_manual(values = color_palette[1:2])

# Visualization 6: Distribution of 'DaysPassed' by target variable
ggplot(treatedTrainingSet, aes(x = DaysPassed, fill = factor(Y_AcceptedOffer))) +
  geom_histogram(bins = 30, alpha = 0.7) +
  labs(title = "Days Passed by Y_AcceptedOffer", x = "Days Passed", y = "Frequency") +
  scale_fill_manual(values = color_palette[1:2])

# Visualization 7: Distribution of 'Education_catP' by target variable
ggplot(treatedTrainingSet, aes(x = Education_catP, fill = factor(Y_AcceptedOffer))) +
  geom_bar() +
  labs(title = "Education Level Distribution by Y_AcceptedOffer", x = "Education Level", y = "Count") +
  scale_fill_manual(values = color_palette[1:2])

# Visualization 8: Monthly contact frequency by target variable
ggplot(treatedTrainingSet, aes(x = factor(LastContactMonth_catP), fill = factor(Y_AcceptedOffer))) +
  geom_bar() +
  labs(title = "Monthly Contact Frequency by Y_AcceptedOffer", x = "Contact Month", y = "Count") +
  scale_fill_manual(values = color_palette[1:2])

# Visualization 9: Target variable against 'RecentBalance'
ggplot(treatedTrainingSet, aes(x = RecentBalance, fill = factor(Y_AcceptedOffer))) +
  geom_histogram(bins = 30, alpha = 0.7) +
  labs(title = "Recent Balance by Y_AcceptedOffer", x = "Recent Balance", y = "Frequency") +
  scale_fill_manual(values = color_palette[1:2])

# Visualization 10: Boxplot of 'RecentBalance' vs 'Y_AcceptedOffer'
ggplot(treatedTrainingSet, aes(x = factor(Y_AcceptedOffer), y = RecentBalance, fill = factor(Y_AcceptedOffer))) +
  geom_boxplot() +
  labs(title = "Recent Balance by Y_AcceptedOffer", x = "Accepted Offer", y = "Recent Balance") +
  scale_fill_manual(values = color_palette[1:2])

# Visualization 11: Distribution of 'PetsPurchases' by target variable
ggplot(treatedTrainingSet, aes(x = PetsPurchases, fill = factor(Y_AcceptedOffer))) +
  geom_bar() +
  labs(title = "Pets Purchases by Y_AcceptedOffer", x = "Pets Purchases", y = "Count") +
  scale_fill_manual(values = color_palette[1:2])

# Visualization 12: Scatter plot of 'Age' vs 'NoOfContacts' colored by target variable
ggplot(treatedTrainingSet, aes(x = Age, y = NoOfContacts, color = factor(Y_AcceptedOffer))) +
  geom_point(alpha = 0.7) +
  labs(title = "Age vs Number of Contacts by Y_AcceptedOffer", x = "Age", y = "Number of Contacts") +
  scale_color_manual(values = color_palette[1:2])

# Visualization 13: Scatter plot of 'Income' vs 'DaysPassed' colored by target variable
ggplot(treatedTrainingSet, aes(x = Income, y = DaysPassed, color = factor(Y_AcceptedOffer))) +
  geom_point(alpha = 0.7) +
  labs(title = "Income vs Days Passed by Y_AcceptedOffer", x = "Income", y = "Days Passed") +
  scale_color_manual(values = color_palette[1:2])

# Visualization 14: Relationship between 'DigitalHabits_5_AlwaysOn' and 'Y_AcceptedOffer'
ggplot(treatedTrainingSet, aes(x = DigitalHabits_5_AlwaysOn, fill = factor(Y_AcceptedOffer))) +
  geom_bar() +
  labs(title = "Digital Habits (Always On) by Y_AcceptedOffer", x = "Always On", y = "Count") +
  scale_fill_manual(values = color_palette[1:2])

# Visualization 15: Plot 'PrevAttempts' vs 'Y_AcceptedOffer'
ggplot(treatedTrainingSet, aes(x = PrevAttempts, fill = factor(Y_AcceptedOffer))) +
  geom_bar() +
  labs(title = "Previous Attempts by Y_AcceptedOffer", x = "Previous Attempts", y = "Count") +
  scale_fill_manual(values = color_palette[1:2])

# Visualization 16: Boxplot of 'Past_Outcome' vs 'Y_AcceptedOffer'
ggplot(treatedTrainingSet, aes(x = factor(Y_AcceptedOffer), y = past_Outcome_catP, fill = factor(Y_AcceptedOffer))) +
  geom_boxplot() +
  labs(title = "Past Outcome by Y_AcceptedOffer", x = "Accepted Offer", y = "Past Outcome") +
  scale_fill_manual(values = color_palette[1:2])


###################################Model###########################################

##########Model#1 - Logistic Regression and identifying the BestFit 
table(treatedTrainingSet$Y_AcceptedOffer)

names(treatedTrainingSet)

# Define input variables as a string
inputVars <- "Communication_catP + Communication_catB + NoOfContacts + DaysPassed + PrevAttempts + 
              past_Outcome_catP + past_Outcome_catB + annualDonations_catP + annualDonations_catB + 
              PetsPurchases + DigitalHabits_5_AlwaysOn + AffluencePurchases + Age + Marital_catP + 
              Marital_catB + Education_catP + Education_catB + DefaultOnRecord + RecentBalance + 
              HHInsurance + CarLoan + carYr + carYr_isBAD + Duration + Communication_lev_NA + 
              Communication_lev_x_cellular + Communication_lev_x_telephone + past_Outcome_lev_NA + 
              past_Outcome_lev_x_failure + past_Outcome_lev_x_other + past_Outcome_lev_x_success + 
              headOfhouseholdGender_lev_x_F + headOfhouseholdGender_lev_x_M + annualDonations_lev_x_ + 
              Marital_lev_x_divorced + Marital_lev_x_married + Marital_lev_x_single + Education_lev_NA + 
              Education_lev_x_primary + Education_lev_x_secondary + Education_lev_x_tertiary"

# Create the formula using paste
formula <- as.formula(paste("Y_AcceptedOffer ~", inputVars))

# Running logistic regression to see relevant input variables 
logReg <- glm(formula, data = treatedTrainingSet, family = 'binomial')
summary(logReg)
#AIC:  2732.4

bestLogReg <- step(logReg, direction='backward')
summary(bestLogReg)

#Following have small p-values, indicating that they are statistically significant in predicting the outcome.
#Y_AcceptedOffer ~ Communication_catP + Communication_catB + 
#NoOfContacts + past_Outcome_catB + annualDonations_catB + 
#  PetsPurchases + Marital_catB + Education_catB + DefaultOnRecord + 
#  HHInsurance + CarLoan + carYr + Duration
#AIC: 2711.2

parsiInputVars <- "Communication_catB + NoOfContacts + past_Outcome_catB + HHInsurance + CarLoan + Duration + Marital_catB "
parsiFormula <- as.formula(paste("Y_AcceptedOffer ~", parsiInputVars))

#Parsimonious LogReg for computationally efficient model with explain-ability
parsiLogReg <- glm(parsiFormula, data = treatedTrainingSet, family = 'binomial')
summary(parsiLogReg)
#AIC: 3033.7

# Get predictions
parsiLogReg <- predict(parsiLogReg, type='response')
head(parsiLogReg)

# Classify
cutoff <- 0.5
logRegPred <- ifelse(parsiLogReg >= cutoff, 1,0)

# Organize w/Actual
results <- data.frame(dataID              = treatedTrainingSet$dataID,
                      actual              = treatedTrainingSet$Y_AcceptedOffer,
                      LogRegprob          = parsiLogReg,
                      logRegPred          = logRegPred)
head(results,10)

results$logRegPred <- factor(results$logRegPred)
results$actual <- factor(results$actual)

levels(trainClass$pred)
levels(trainClass$actual)

# Get a confusion matrix
confMatLogRegTrain <- confusionMatrix(results$logRegPred, results$actual)
confMatLogRegTrain

##########Model#2 - Decision Tree 
head(treatedTrainingSet)
# Fit a decision tree with caret with least computation power
set.seed(1234)

dTreeFormula <- as.formula(paste("as.factor(Y_AcceptedOffer) ~", inputVars))

dTree <- train(
  dTreeFormula,                         # formula using Y_AcceptedOffer as the target variable
  data = treatedTrainingSet,       # data input
  method = "rpart",                # recursive partitioning (trees)
  # Define a range for the complexity parameter (CP) to test
  tuneGrid = data.frame(cp = c(  0.0001,0.001,0.005, 0.01,0.02, 0.05)),
  # Set control parameters for the rpart model
  control = rpart.control(minsplit = 10, minbucket = 5)
)

dTree
#using least RMSE selected optimal model with cp = 0.005.
#trainProbs <- predict(dTree, treatedTrainingSet) 
trainProbs <- predict(dTree, treatedTrainingSet, type = "prob")
head(trainProbs,10)

results$dTreeProb <- trainProbs[, "1"]
head(results,10)

# Get the final pred and actuals
trainClass <- data.frame(pred = colnames(trainProbs)[max.col(trainProbs)],
                         actual = treatedTrainingSet$Y_AcceptedOffer)
head(trainClass, 10)

trainClass$pred <- factor(trainClass$pred)
trainClass$actual <- factor(trainClass$actual)

levels(trainClass$pred)
levels(trainClass$actual)

results$dTreePred <- trainClass$pred
head(results,25)

# Confusion Matrix
confMatdTreeTrain <- confusionMatrix(trainClass$pred,trainClass$actual)
confMatdTreeTrain


#########Model#3 - Random Forest 
rFormula <- as.formula(paste("as.factor(Y_AcceptedOffer) ~", inputVars))


# OverFit a random forest model with Caret
rForestOverFit <- train(rFormula,
                 data = treatedTrainingSet,
                 method = "rf",
                 verbose = FALSE,
                 ntree = 100,
                 tuneGrid = data.frame(mtry = 10)) #num of vars used in each tree
rForestOverFit 
#Results in Accuracy = 1 , Kappa =1 !!!!
#reducing the complexity of the Random Forest model
#Limit the number of trees.
#Limit the number of features used at each split (mtry).


# another Fit for a random forest model after a few iterations
rForest <- train(rFormula,
                  data = treatedTrainingSet,
                  method = "rf",
                  verbose = FALSE,
                  ntree = 50,
                  tuneGrid = data.frame(mtry = 5)) #num of vars used in each tree
rForest

# Other interesting model artifacts
varImp(rForest)
plot(varImp(rForest), top = 20)
predProbs   <- predict(rForest,  
                       treatedTrainingSet, 
                       type = c("prob"))

predAccept <- predict(rForest,  treatedTrainingSet)

head(predProbs)
head(predAccept)

results$rForestProb <- predProbs[, "1"]
results$rForestPred <- predAccept

head(results,25)
nrow(results)

# Confusion Matrix
caret::confusionMatrix(predAccept, 
                       as.factor(treatedTrainingSet$Y_AcceptedOffer))


###################################Assess###########################################

# review model performance on an independent Validation Set

##########Model#1 -assess- Logistic Regression
parsiLogReg <- glm(parsiFormula, data = treatedValidationSet, family = 'binomial')
summary(parsiLogReg)
#AIC: 3033.7

# Get predictions
parsiLogReg <- predict(parsiLogReg, type='response')
head(parsiLogReg)

# Classify
logRegPred <- ifelse(parsiLogReg >= cutoff, 1,0)

vresults <- data.frame(dataID             = treatedValidationSet$dataID,
                      actual              = treatedValidationSet$Y_AcceptedOffer,
                      LogRegprob          = parsiLogReg,
                      logRegPred          = logRegPred)
head(vresults,10)

vresults$logRegPred <- factor(vresults$logRegPred)
vresults$actual <- factor(vresults$actual)


# Get a confusion matrix
confMatLogRegValid <- confusionMatrix(vresults$logRegPred, vresults$actual)
confMatLogRegValid

##########Model#2 - Decision Tree 
trainProbs <- predict(dTree, treatedValidationSet, type = "prob")
head(trainProbs,10)
vresults$dTreeProb <- trainProbs[, "1"]

# Get the final pred and actuals
trainClass <- data.frame(pred = colnames(trainProbs)[max.col(trainProbs)],
                         actual = treatedValidationSet$Y_AcceptedOffer)
head(trainClass, 10)

vresults$dTreePred <- trainClass$pred

trainClass$pred <- factor(trainClass$pred)
trainClass$actual <- factor(trainClass$actual)

levels(trainClass$pred)
levels(trainClass$actual)
head(vresults,10)

# Confusion Matrix
confMatdTreeValid <- confusionMatrix(trainClass$pred,trainClass$actual)
confMatdTreeValid

#########Model#3 - assess- Random Forest 
predProbs   <- predict(rForest,  
                       treatedValidationSet, 
                       type = c("prob"))

predAccept <- predict(rForest,  treatedValidationSet)

vresults$rForestProb <- predProbs[, "1"]
vresults$rForestPred <- predAccept

# Confusion Matrix
caret::confusionMatrix(predAccept, 
                       as.factor(treatedValidationSet$Y_AcceptedOffer))


########################## apply models on  prospective customers Set #############################

##########Model#1 -apply- Logistic Regression
#parsiLogReg <- glm(parsiFormula, data = treatedProspectivesSet, family = 'binomial')
#summary(parsiLogReg)
#AIC: 359

# Get predictions
#parsiLogReg <- predict(parsiLogReg, type='response')
head(parsiLogReg)

# Classify
#logRegPred <- ifelse(parsiLogReg >= cutoff, 1,0)

#presults <- data.frame(dataID             = treatedProspectivesSet$dataID,
                      # actual              = treatedProspectivesSet$Y_AcceptedOffer,
                      # LogRegprob          = parsiLogReg,
                    #   logRegPred          = logRegPred)
#head(presults,10)

#presults$logRegPred <- factor(presults$logRegPred)
#presults$actual <- factor(presults$actual)


# Get a confusion matrix
#confMatLogRegValid <- confusionMatrix(vresults$logRegPred, vresults$actual)
#confMatLogRegValid

##########Model#2 - Decision Tree 
#trainProbs <- predict(dTree, treatedValidationSet, type = "prob")
#head(trainProbs,10)
#vresults$dTreeProb <- trainProbs[, "1"]

# Get the final pred and actuals
#trainClass <- data.frame(pred = colnames(trainProbs)[max.col(trainProbs)],
                         actual = treatedValidationSet$Y_AcceptedOffer)
#head(trainClass, 10)

#vresults$dTreePred <- trainClass$pred

#trainClass$pred <- factor(trainClass$pred)
#trainClass$actual <- factor(trainClass$actual)

#levels(trainClass$pred)
#levels(trainClass$actual)
#head(vresults,10)

# Confusion Matrix
#confMatdTreeValid <- confusionMatrix(trainClass$pred,trainClass$actual)
#confMatdTreeValid

#########Model#3 - assess- Random Forest 
#predProbs   <- predict(rForest,  
 #                      treatedValidationSet, 
#                       type = c("prob"))

#predAccept <- predict(rForest,  treatedValidationSet)

#vresults$rForestProb <- predProbs[, "1"]
#vresults$rForestPred <- predAccept

# Confusion Matrix
#caret::confusionMatrix(predAccept, 
 #                      as.factor(treatedValidationSet$Y_AcceptedOffer))


# combine the results from both training and validation datasets
fullResults <- rbind(results, vresults)
nrow(fullResults)

# Calculate the average of probabilities from 3 models ### ENSEMBLE
fullResults$average <- rowMeans(fullResults[, c("LogRegprob", "dTreeProb", "rForestProb")], na.rm = TRUE)

# Rank `average` in descending order and add the rank as a new column in `fullResults`
fullResults$rank <- rank(-fullResults$average, ties.method = "first")
head(fullResults)

#Selection of top 100 customers 
# Filter fullResults to include only ranks 1 to 100
topCustomers100 <- fullResults[fullResults$rank %in% 1:100, ]
head(topCustomers100,25)

#join back to input dataset for insights on top 100 customers 
topCustomers100Details <- inner_join(topCustomers100, fullDataSet, by = c('dataID'))
head(topCustomers100Details)

write.csv(topCustomers100Details, "~/Git/GitHub_R/personalFiles/topCustomers100Details.csv", row.names = FALSE)

#past_Outcome, RecentBalance,	CarLoan	,Duration are strong indicators of likelihood of acceptance 





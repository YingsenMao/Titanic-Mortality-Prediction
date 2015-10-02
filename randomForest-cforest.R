library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(randomForest)
library(caret)
library(party)

get_accu <- function(y_pred, y_real){
  accu_table <- table(y_pred, y_real)
  accu <- (accu_table[1, 1] + accu_table[2, 2]) / sum(accu_table)
  return (accu)
}


setwd("~/Documents/Kaggle/titanic")
train <- read.csv('train.csv')
test <- read.csv('test.csv')

# Join together the test and train sets for easier feature engineering
test$Survived <- NA
combi <- rbind(train, test)

# Convert to a string
combi$Name <- as.character(combi$Name)

# Engineered variable: Title
combi$Title <- strsplit(combi$Name, split='[,.]')[[1]][2]  # Won't work!
combi$Title <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
combi$Title <- sub(' ', '', combi$Title)
# Inspect new feature
table(combi$Title)
# Combine small title groups
combi$Title[combi$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
combi$Title[combi$Title %in% c('Capt', 'Don', 'Major', 'Sir', 'Jonkheer')] <- 'Sir'
combi$Title[combi$Title %in% c('Dona', 'Lady', 'the Countess')] <- 'Lady'
# Convert to a factor
combi$Title <- factor(combi$Title)

# Engineered variable: Family size
combi$FamilySize <- combi$SibSp + combi$Parch + 1

# Engineered variable: Family
combi$Surname <- sapply(combi$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
combi$FamilyID <- paste(as.character(combi$FamilySize), combi$Surname, sep="")
combi$FamilyID[combi$FamilySize <= 2] <- 'Small'
# Inspect new feature
table(combi$FamilyID)
# Delete erroneous family IDs
famIDs <- data.frame(table(combi$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
combi$FamilyID[combi$FamilyID %in% famIDs$Var1] <- 'Small'
# Convert to a factor
combi$FamilyID <- factor(combi$FamilyID)

Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                data=combi[!is.na(combi$Age),], method="anova")
combi$Age[is.na(combi$Age)] <- predict(Agefit, combi[is.na(combi$Age),])
summary(combi)
summary(combi$Embarked)
which(combi$Embarked == '')
combi$Embarked[c(62,830)] = "S"
combi$Embarked <- factor(combi$Embarked)
summary(combi$Fare)
which(is.na(combi$Fare))
combi$Fare[1044] <- median(combi$Fare, na.rm=TRUE)
combi$FamilyID2 <- combi$FamilyID
combi$FamilyID2 <- as.character(combi$FamilyID2)
combi$FamilyID2[combi$FamilySize <= 3] <- 'Small'
combi$FamilyID2 <- factor(combi$FamilyID2)

set.seed(415)
str(combi)
train <- combi[1:891,]
test <- combi[892:1309,]
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize +
                      FamilyID2, data=train, importance=TRUE, ntree=2000)
pred <- predict(fit, train)
confusionMatrix(pred, train$Survived)  #0.927 
varImpPlot(fit)



fold <- createFolds(train$Survived, 10)
accu_res <- c()
for (i in 1:length(fold)){
  trainSet <- train[-fold[[i]], ]
  testSet <- train[fold[[i]], ]
  fit_dummy <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize +
                              FamilyID2, data=trainSet, ntree=2000)
  pred_dummy <- predict(fit_dummy, testSet, type = 'class')
  accu <- get_accu(pred_dummy, testSet$Survived)
  accu_res <- c(accu_res, accu)
}
mean(accu_res) # 0.8249
sd(accu_res) #0.03629

ctrl <- trainControl(method = 'cv',
                     number = 10)
fit_rf <- train(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize +
               FamilyID2, data=train, method = 'rf', ntrees = 2000,
               tuneGrid=expand.grid(mtry = 3:8),
               trControl = ctrl) #0.8350 mtry = 7

Prediction <- predict(fit_rf, test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "firstforest.csv", row.names = FALSE) #0.7942


set.seed(415)
fit <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
               data = train, controls=cforest_unbiased(ntree=2000, mtry=3))
Prediction <- predict(fit, test, OOB=TRUE, type = "response")
submit <- data.frame(PassengerId = test$PassengerId, Survived = Prediction)
write.csv(submit, file = "cforest.csv", row.names = FALSE)
library(randomForest)
library(caret)

setwd("~/Documents/Kaggle/titanic")
train <- read.table('train_clean_missing.csv', header = TRUE,
                    sep = ',', stringsAsFactors = FALSE)
test <- read.table('test_clean_missing.csv', header = TRUE,
                   sep = ',', stringsAsFactors = FALSE)
train$Survival <- factor(train$Survived, levels = 0:1, 
                         labels = c('dead', 'live'))

ctrl <- trainControl(method = 'repeatedcv',
                     repeats = 10,
                     number = 10)
fit_1 <- train(Survival ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 
               data = train, method = 'rf', ntrees = 2000,
               tuneGrid=expand.grid(mtry = 1:5),
               trControl = ctrl) #0.8391678, mtry = 3

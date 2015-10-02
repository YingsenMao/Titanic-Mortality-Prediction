install.packages('rpart.plot')
library(rpart)
library(rpart.plot)
setwd("~/Documents/Kaggle/titanic")
train <- read.table('train_clean_missing.csv', header = TRUE,
                    sep = ',', stringsAsFactors = FALSE)
head(train)
str(train)
table(train$Survived)
table(train$Survived, train$Sex)
prop.table(table(train$Survived, train$Sex), 2)

summary(train$Age)
train$Child <- ifelse(train$Age > 18, 'adult', 'child' )
table(train$Child)
table(train$Child, train$Sex)

aggregate(Survived ~ Sex + Child, data = train, sum)
aggregate(Survived ~ Sex + Child, data = train, length)
aggregate(Survived ~ Sex + Child, data = train, FUN = function(x) {sum(x) / length(x)})

prop.table(table(train$Pclass, train$Survived), 1)
aggregate(Survived ~ Sex + Pclass, data = train, length)
aggregate(Survived ~ Sex + Pclass, data = train, FUN = function(x) {sum(x) / length(x)})

aggregate(Survived ~ Sex + Child + Pclass, data = train, length)
aggregate(Survived ~ Sex + Child + Pclass, data = train, FUN = function(x) {sum(x) / length(x)})

train$Ticket[train$Pclass == 3]
quantile(train$Fare, seq(0.25, 0.75, by= 0.25))
train$Fare2 <- '31+'
train$Fare2[train$Fare <= 31 & train$Fare >= 14.5] <- '14.5-31'
train$Fare2[train$Fare < 14.5 & train$Fare >= 8] <- '8-14.5'
train$Fare2[train$Fare < 8] <- '-8'
aggregate(Survived ~ Sex + Pclass + Fare2, data = train, length)
aggregate(Survived ~ Sex + Pclass + Fare2, data = train, sum)
aggregate(Survived ~ Sex + Pclass + Fare2, 
          data = train, FUN = function(x) {sum(x) / length(x)})

train$Survival = factor(train$Survived, levels = 0:1, labels = c('live', 'dead'))
str(train)
table(train$Survival)
cfit <- rpart(Survival ~ Sex + Child + Pclass, data = train,
              method = 'class')
print(cfit)
par(mar = rep(0.1, 4))
plot(cfit)
text(cfit)

prp(cfit, extra = 7)

cfit2 <- rpart(Survival ~ Sex + Child + Pclass, data = train,
              method = 'class', control = rpart.control(xval = 10,
                                                        minsplit = 30,
                                                        cp = 0))
cfit2 <- rpart(Survival ~ Sex + Child + Pclass, data = train,
               method = 'class', control = rpart.control(xval = 10,
                                                         cp = 0))
printcp(cfit2)
prp(cfit2, extra = 7, prefix = 'fraction\n')
best_cp = printcp(cfit2)[5, 1]
prune_6 <- prune(cfit2, cp = 0)
prp(prune_6)
plot(prune_6)
text(prune_6)
install.packages('rattle')
library(rattle)
install.packages('rattle')
install.packages('rpart.plot')
install.packages('RColorBrewer')
library(rattle)
library(rpart.plot)
library(RColorBrewer)
fancyRpartPlot(prune_6, tweak = 2.0)


best_cp = printcp(cfit2)[2, 1]
prune_2 <- prune(cfit2, cp = best_cp)
prp(prune_2)

tree <- rpart(Survived ~., data = train, cp = 0.02)
prp(tree, extra = 7, prefix = 'fraction\n')
test <- read.table('test_clean_missing.csv', header = TRUE,
                   sep = ',')
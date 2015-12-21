rm(list = ls(all = TRUE))
library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
library(randomForest)
library(knitr)
setwd("C:\\Users\\sai\\Desktop\\Full\\coursera\\machine learning")
training <- read.csv ("C:\\Users\\sai\\Desktop\\Full\\coursera\\machine learning\\pml-training.csv")
testing <- read.csv("C:\\Users\\sai\\Desktop\\Full\\coursera\\machine learning\\\\pml-testing.csv")
dim(training)
head(training)

inTrain <- createDataPartition(training$classe, p=0.6, list=FALSE)
myTraining <- training[inTrain, ]
myTesting <- training[-inTrain, ]
dim(myTraining);dim(myTesting)



nzv <- nearZeroVar(myTraining, saveMetrics = TRUE)
nzvmyTraining <- myTraining[, nzv$nzv == FALSE]
head(nzvmyTraining)
dim(nzvmyTraining)

nzv <- nearZeroVar(myTesting, saveMetrics = TRUE)
nzvmyTesting <- myTesting[, nzv$nzv == FALSE]
head(nzvmyTesting)
dim(nzvmyTesting)

nzvmyTraining <- nzvmyTraining[c(-1)]

dim(nzvmyTraining)


trainingV3 <- nzvmyTraining
  for(i in 1:length(nzvmyTraining)) {
        if( sum( is.na( nzvmyTraining[, i] ) ) /nrow(nzvmyTraining) >= .7) {
  for(j in 1:length(trainingV3)) {
        if( length( grep(names(nzvmyTraining[i]), names(trainingV3)[j]) ) == 1){
        trainingV3 <- trainingV3[ , -j]
                 }   
             } 
        }
    }

# Set back to the original variable name
myTraining <- trainingV3
rm(trainingV3)

clean1 <- colnames(nzvmyTraining)
clean2 <- colnames(nzvmyTraining[, -58])  # remove the classe column
myTesting <- nzvmyTesting[clean1]         # allow only variables in myTesting that are also in myTraining
testing <- testing[clean2]             # allow only variables in testing that are also in myTraining

dim(myTesting)

dim(testing)

for (i in 1:length(testing) ) {
        for(j in 1:length(myTraining)) {
                if( length( grep(names(myTraining[i]), names(testing)[j]) ) == 1)  {
                        class(testing[j]) <- class(myTraining[i])
                }      
        }      
}

# To get the same class between testing and myTraining
testing <- rbind(myTraining[2, -58] , testing)
testing <- testing[-1,]


dim(testing)
clean1 <- colnames(myTraining)
clean2 <- colnames(myTraining[, -58])  # remove the classe column
myTesting <- myTesting[clean1]         # allow only variables in myTesting that are also in myTraining
testing <- testing[clean2]  


set.seed(123)
modFitA1 <- rpart(classe ~ ., data=myTraining, method="class")
fancyRpartPlot(modFitA1)

library(e1071)
predictionsA1 <- predict(modFitA1, myTesting, type = "class")
cmtree <- confusionMatrix(predictionsA1, myTesting$classe)
cmtree

set.seed(123)
modFitB1 <- randomForest(classe ~ ., data=myTraining)
predictionB1 <- predict(modFitB1, myTesting, type = "class")
cmrf <- confusionMatrix(predictionB1, myTesting$classe)
cmrf
varImpPlot(modFitB1)
plot(modFitB1)

set.seed(123)
fitControl <- trainControl(method = "repeatedcv",
                           number = 5,
                           repeats = 1)

gbmFit1 <- train(classe ~ ., data=myTraining, method = "gbm",
                 trControl = fitControl,
                 verbose = FALSE)


gbmFinMod1 <- gbmFit1$finalModel

gbmPredTest <- predict(gbmFit1, newdata=myTesting)
gbmAccuracyTest <- confusionMatrix(gbmPredTest, myTesting$classe)
gbmAccuracyTest

predictionB2 <- predict(modFitB1, testing, type = "class")
predictionB2

plot(gbmFit1, ylim=c(0.9, 1))

# Write the results to a text file for submission
pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}

pml_write_files(predictionB2)



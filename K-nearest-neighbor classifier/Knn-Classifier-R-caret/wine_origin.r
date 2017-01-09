#package import statement
library(caret)

#Data import statements
dataurl <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
download.file(url = dataurl, destfile = "wine.data")
wine_df <- read.csv("wine.data", header = FALSE)    #load data to wine_df dataframe

str(wine_df) #structure of our data frame

#Data partitioning
set.seed(3033)
intrain <- createDataPartition(y = wine_df$V1, p= 0.7, list = FALSE)
training <- wine_df[intrain,]
testing <- wine_df[-intrain,]

#check dimensions of train & test set
dim(training); dim(testing);  

#check whether any NA value exists or not
anyNA(wine_df)

summary(wine_df) #summary stats of our data

training[["V1"]] = factor(training[["V1"]]) #conversion of V1 integer variable to factor variable

#Training & Preprocessing 
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(3333)
knn_fit <- train(V1 ~., data = training, method = "knn",
 trControl=trctrl,
 preProcess = c("center", "scale"),
 tuneLength = 10)
 
knn_fit #knn classifier

#plot accuracy vs K Value graph 
plot(knn_fit) 

#predict classes for test set using knn classifier
test_pred <- predict(knn_fit, newdata = testing)
test_pred

#Test set Statistics 
confusionMatrix(test_pred, testing$V1 )  #check accuracy

###################################
# Section 1: Importing + cleaning #
###################################

# import and read
setwd("/Users/annayang/Downloads")
data <- read.csv("cleandata2.csv", header=T, na.strings=c("","NA"))
set.seed(1)

# cleaning data
# function to remove N/A COMPLETELY
data[] <- lapply(data, function(x) {
  is.na(levels(x)) <- levels(x) == "N/A"
  x
})

data = na.omit(data)
# Originally 435 observations. Sum of the NA shows 54 NA's. We have removed them
# Now with 379 observations, we have lost 55 observations. We think the additional is the heading.

# removing variables
data <- subset( data, select = -SchoolName )
data <- subset( data, select = -ZipCode )

# combine math,english and writing into one overall score

data$Score = data$Math + data$Eng + data$Writing
data <- subset( data, select = -Math )
data <- subset( data, select = -Eng )
data <- subset( data, select = -Writing )

# format data
data$PercentAsian= as.numeric(sub("%","",data$PercentAsian))
data$PercentHispanic= as.numeric(sub("%","",data$PercentHispanic))
data$PercentBlack= as.numeric(sub("%","",data$PercentBlack))
data$PercentWhite= as.numeric(sub("%","",data$PercentWhite))
data$StudentAttendanceRate= as.numeric(sub("%","",data$StudentAttendanceRate))

# dummy variables
levels(data$StudentAchievementRating)[1] = "AT"
levels(data$StudentAchievementRating)[2] = "ET"
levels(data$StudentAchievementRating)[3] = "MT"
levels(data$StudentAchievementRating)[4] = "NMT"

# shortern Student Achievement Rating name into Achieve
# shortern StudentAttendanceRate into AttendRate
data$Achieve = data$StudentAchievementRating
data$AttendRate = data$StudentAttendanceRate

# remove old columns 
data <- subset( data, select = -StudentAchievementRating )
data <- subset( data, select = -StudentAttendanceRate )

# EDA Analysis 

plot(data$Borough,main="Number of High Schools in Each Borough ",xlab="Borough",ylab="# of schools",
     ylim= c(0,120))

# dummy variables
data$Borough = as.character(data$Borough)
data$Bronx[data$Borough=="Bronx"] = "1"
data$Brooklyn[data$Borough=="Brooklyn"] = "1"
data$Manhattan[data$Borough=="Manhattan"] = "1"
data$Queens[data$Borough=="Queens"] = "1"
data$StatIsl[data$Borough=="Staten Island"] = "1"

# remove Borough since it's been made into dummy variables
data[is.na(data)] <- 0
data <- subset( data, select = -Borough )

# populate dummy variables and remove original column
data$Achieve = as.character(data$Achieve)
data$AT[data$Achieve=="AT"] = "1"
data$ET[data$Achieve=="ET"] = "1"
data$MT[data$Achieve=="MT"] = "1"
data$NMT[data$Achieve=="NMT"] = "1"
data[is.na(data)] <- 0
data <- subset( data, select = -Achieve )

# populate dummy variables and remove original column
data$Quality.Review = as.character(data$Quality.Review)
data$Developing[data$Quality.Review=="Developing"] = "1"
data$Proficient[data$Quality.Review=="Proficient"] = "1"
data$WellDev[data$Quality.Review=="Well Developed"] = "1"
data[is.na(data)] <- 0
data <- subset( data, select = -Quality.Review )

# convert the dummy variables into numeric (currently char from becoming a dummy variable)
data$Bronx = as.numeric(data$Bronx)
data$Brooklyn = as.numeric(data$Brooklyn)
data$Manhattan = as.numeric(data$Manhattan)
data$Queens = as.numeric(data$Queens)
data$StatIsl = as.numeric(data$StatIsl)
data$AT = as.numeric(data$AT)
data$ET = as.numeric(data$ET)
data$MT = as.numeric(data$MT)
data$NMT = as.numeric(data$NMT)
data$Developing = as.numeric(data$Developing)
data$Proficient = as.numeric(data$Proficient)
data$WellDev = as.numeric(data$WellDev)

# separating dataset into training and test set
set.seed(1)
train.index = sample(1:nrow(data),nrow(data)*0.8)
train = data[train.index,]
test = data[-train.index,]
#original train has 303 obs and test would be 76.

# separating original train into train.index and validation 

train.index2 = sample(1:nrow(train),nrow(train)*0.8)
trainset = train[train.index2,]
valid = train[-train.index2,]


#############################
# Section 2: Model Building #
#############################

### Model 1: Linear Regression

modelfull = lm(Score~.,data=trainset)
summary(modelfull)
confint(modelfull)
# coefficents for demographic are negative, 
#perhaps there's an unaccounted demographic which is affecting our outcome


# Feature Selection

#install.packages("leaps")
library(leaps)
model_fwd = regsubsets(Score~., data=data, nvmax=NULL, method="forward")
# Take a look at the process
summary(model_fwd)
plot(model_fwd, scale="adjr2", main="Forward Selection: AdjR2")
model_fwd_summary = summary(model_fwd) # Store summary output
model_fwd_summary
which.max(model_fwd_summary$adjr2) # Display best subset by adjr2
summary(model_fwd)$which[10,]
best_model_fwd = lm(Score~StudentEnrollment+PercentWhite+PercentBlack+PercentHispanic+PercentAsian+AttendRate+Bronx+Manhattan+MT+WellDev, data=data)
summary(best_model_fwd)

## Linear Regression: making predictions 
train.pred.y = predict(best_model_fwd,trainset) # make predictions on training set
error = trainset$Score-train.pred.y # calculate our residuals
sq.error = error^2 # square the residuals
mse = mean(sq.error) # take mean of squared residuals
LRrmse = sqrt(mse) # now we have our RMSE
LRrmse # What is our training RMSE?
# rmse = 101.2303
par(mfrow=c(2,2))
plot(best_model_fwd)

# 2.1: Decision Tree 
####################
par(mfrow=c(1,1))
set.seed(1)

#install.packages("tree")
library(tree)
model = tree(Score~StudentEnrollment+PercentWhite+PercentBlack+PercentHispanic+PercentAsian+AttendRate+Bronx+Manhattan+MT+WellDev,data=trainset); summary(model)

plot(model)
text(model)

pred.y = predict(model,valid)
rmse = sqrt(mean((pred.y-valid$Score)^2))
rmse

best.tree = cv.tree(model,K=10) # K=10 specifying 10-fold cross validation
best.tree

x = best.tree$size
y = best.tree$dev
plot(x,y,main = "Deviance vs Tree Size", xlab="tree size",ylab="deviance",type="b",pch=20,col="blue")

model.pruned = prune.tree(model,best=4)
pred.y = predict(model.pruned,valid)
DTrmse = sqrt(mean((pred.y-valid$Score)^2))
DTrmse

plot(model.pruned)
text(model.pruned)

# 2.2: Random Forest
####################

#install.packages("randomForest")
library(randomForest)
reg.model = randomForest(Score~StudentEnrollment+PercentWhite+PercentBlack+PercentHispanic+PercentAsian+AttendRate+Bronx+Manhattan+MT+WellDev,data=trainset); reg.model
plot(reg.model,main="Error as ntree increases")
set.seed(1)
reg.model = randomForest(Score~StudentEnrollment+PercentWhite+PercentBlack+PercentHispanic+PercentAsian+AttendRate+Bronx+Manhattan+MT+WellDev,data=trainset,ntree=100,importance=TRUE); reg.model
importance(reg.model)
varImpPlot(reg.model,main="Variable Importance Plots")
RFrmse = sqrt(8576.012); RFrmse

### Comparing three models for lowest RMSE ###

`Model RMSE`=1
as.data.frame(cbind(`Model RMSE`,LRrmse,DTrmse,RFrmse))

### Making predictions on test set
set.seed(1)
reg.model2 = randomForest(Score~StudentEnrollment+PercentWhite+PercentBlack+PercentHispanic+
        PercentAsian+AttendRate+Bronx+Manhattan+MT+WellDev,data=test,ntree=100,importance=TRUE); reg.model2

pred.y = predict(reg.model2,test)
RFmse = mean((pred.y-test$Score)^2); RFmse
RFrmse= sqrt(RFmse);RFrmse



#Graphing attendRate vs score to support our argument
x=test$AttendRate
y=test$Score

plot(x,y,main="Attend Rate vs Score", xlab="Attendance Rate", ylab="SAT Score",ylim=c(800,2200))


#creating a baseline
sd(data$Score)

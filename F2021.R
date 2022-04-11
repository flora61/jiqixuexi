library(nnet)
library(gbm)
library(glmnet)
library(caret)
library(MASS)

#-------Q2--------
dat = read.csv(file="C:\\R file\\ML2\\dodgysales.csv", stringsAsFactors=TRUE)
n = nrow(dat) 
set.seed(6041) 
i.train = sample(1:n, floor(.7*n)) 
dat.train = dat[i.train,] 
dat.validation = dat[-i.train,] 

#(a) regression or classification?
# regression

#(b) number of predictors?
ncol(dat)
# 12

#(c) min-max normalisation
minmax.scale = function(x){
  if(!is.factor(x)){
    xs = (x-min(x))/(max(x)-min(x))
  }
  else {
    xs = x
  }
  return(xs)
}
# create a copy
dat.s = as.data.frame(lapply(dat, minmax.scale))
# (i) summaries
summary(dat.s$Sales)
summary(dat.s$BudgOp)
summary(dat.s$Training)
#(ii) two single layer NN
dat.s.train = dat.s[i.train,] 
dat.s.validation = dat.s[-i.train,] 
set.seed(6041)
nn3 = nnet(Sales~., data=dat.s.train, size=3)
set.seed(6041)
nn8 = nnet(Sales~., data=dat.s.train, size=8)
# MSE
mean(nn3$residuals^2)
mean(nn8$residuals^2)
#(iii) predictions for validation set
ytest = dat.s.validation$Sales
pre3 = predict(nn3, dat.s.validation)
pre8 = predict(nn8, dat.s.validation)
# MSE
mean((pre3-ytest)^2)
mean((pre8-ytest)^2)

#(d) GBM
set.seed(6041)
gbo = gbm(Sales~., data=dat.train, distribution='gaussian', n.trees = 100)
pred1 = predict(gbo, dat.train)
pred2 = predict(gbo, dat.validation)
mean((pred1-dat.train$Sales)^2)
mean((pred2-dat.validation$Sales)^2)

#(e) GLM
set.seed(6041)
glmo = glm(Sales~., data = dat.train, family="gaussian")
glmp1 = predict(glmo, dat.train)
glmp2 = predict(glmo, dat.validation)
mean((glmp1-dat.train$Sales)^2)
mean((glmp2-dat.validation$Sales)^2)

#(f) ridge 
set.seed(6041)
xm = model.matrix(Sales~.+0, data = dat.train)
y = dat.train$Sales
ridge = glmnet(xm,y,alpha=0,lambda=.915)
ridge.fit = predict(ridge,xm)
xmv = model.matrix(Sales~.+0, data = dat.validation)
ridgep = predict(ridge,xmv)
mean((ridge.fit-dat.train$Sales)^2)
mean((ridgep-dat.validation$Sales)^2)

#(h) RFE based on Random forest
x = dat.train
x$Sales = NULL
y = dat.train$Sales
set.seed(6041)
subsets <- c(1:10)
ctrl <- rfeControl(functions = rfFuncs,
                   method = "cv",
                   number = 10,
                   verbose = FALSE)
rf.rfe <- rfe(x,y,
              sizes = subsets,
              rfeControl = ctrl)
rf.rfe
dim(dat.train)

#(h) RFE based on backward stepwise elimination
set.seed(6041)
subsets <- c(1:10)
ctrl <- rfeControl(functions = lmFuncs,
                   method = "cv",
                   number = 10,
                   verbose = FALSE)
###############################################
bag.rfe <- rfe(x,y,
              sizes = subsets,
              rfeControl = ctrl)
bag.rfe
dim(dat.train)


#-------Q3--------
library(caret)
library(mlbench) 
data(BreastCancer) 
dat = na.omit(BreastCancer) 
dat$Id = NULL 
set.seed(4061) 
i.train = sample(1:nrow(dat), 600, replace=FALSE) 
dat.train = dat[i.train,] 
dat.validation = dat[-i.train,] 

#(a) random forest and 10-cv use caret
set.seed(4061) 
rf.Control = trainControl(method='cv',number=10)
rf.o = caret::train(Class~., trControl = rf.Control,
                    data=dat.train, method='rf')
rf.p = predict(rf.o, dat.validation)
rf.cm = confusionMatrix(reference=dat.validation$Class, data=rf.p)

#(b) SVM linear kernel
set.seed(4061)
svm.Control = trainControl(method='cv',number=10)
svm.o = caret::train(Class~., trControl = svm.Control,
                    data=dat.train, method='svmLinear')
svm.p = predict(svm.o, dat.validation)
svm.cm = confusionMatrix(reference=dat.validation$Class, data=svm.p)

#(c) SVM radial kernel
set.seed(4061)
svmR.Control = trainControl(method='cv',number=10)
svmR.o = caret::train(Class~., trControl = svmR.Control,
                     data=dat.train, method='svmRadial')
svmR.p = predict(svmR.o, dat.validation)
svmR.cm = confusionMatrix(reference=dat.validation$Class, data=svmR.p)

#(d) compare 3 models
round(cbind(rf.cm$overall, svm.cm$overall, svmR.cm$overall),3)

#(e) most important
varImp(rf.o)
varImp(svm.o)
varImp(svmR.o)









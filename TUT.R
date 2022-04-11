library(class) # contains knn()
library(MASS)  # to have lda()
library(e1071) # svm
library(nnet)  # single layer NNs
library(leaps) # contains regsubsets()
# download dataset
dat = read.csv(file="C:\\R file\\ML2\\dodgysales.csv", stringsAsFactors=TRUE)
dat = iris[sample(1:n),-4] # removing 4th predictor 
# Normalize data to [0,1] using transformation
myscale <- function(x){
  minx = min(x,na.rm=TRUE)
  maxx = max(x,na.rm=TRUE)
  return((x-minx)/(maxx-minx))
}
datss = data.frame(lapply(dat,myscale))
dat[,1:4] = myscale(dat[,1:4])
# or
dats$Salary = (dat$Salary-min(dat$Salary)) / diff(range(dat$Salary))
# or
NC = ncol(dat)
mins = apply(dat[,-NC],2,min)
maxs = apply(dat[,-NC],2,max)
dats = dat
dats[,-NC] = scale(dats[,-NC],center=mins,scale=maxs-mins)

par(mfrow=c(1,2))
# MSE
mean( (y.test-svm.pred)^2 )
mean(nn3$residuals^2)
# RMSE:
sqrt(mean((preds-BostonHousing$medv[-itrain])^2))
# corresponding misclassification rate
acc.glm = 1-sum(diag(tb.glm))/sum(tb.glm)

# S1: logistic and LASSO for scale and unscale.
      # imputing missing data(predictors and variables) fit using overall mean then LASSO

# S2: tuning LASSO

# S3: knn
      sum(diag(tb)) / sum(tb) # overall accuracy
      1 - sum(diag(tb)) / sum(tb) # overall classification error rate
      # logistic regression. GLM
      # LDA and QDA for 2 class and 3 class - exercise 4
      dat$Species = as.factor(ifelse(iris$Species=="virginica",1,0)) # class into numeric
      # boxplot
      # benchmarking on 4 classifiers: knn, glm, lda, qda with 10-cv - exercise 5
      # ROC AUC
      
# S4: tree and prune tree(cv.tree())
      High = ifelse(Carseats$Sales<=8, "No", "Yes") # numeric to classic
      # ROC AUC
      # random forest and bagging
      # importance for RF (Ensemble method plot) - exercise 6 
      # gradient boosting GBM and ROC - Exercise 7
      gb.out = gbm(High~., data=CS, 
                   distribution="bernoulli", # use "gaussian" instead for regression
                   n.trees=5000, # size of the ensemble
                   interaction.depth=1) # depth of the trees, 1 = stumps
      # GBM with caret classification and regression - exercise 8
      rm(list=ls()) #clear the environment
      
# S4+ xgboost with caret
      
# S5a:SVM
      # numeric to classification
      # svm model -> svm tune to find best parameters -> fit model for opt paras
      # plot
      # ROC
      # compare with RF
      
# S5b:SVM linear, polynomial, radial and sigmoid
      # plot with grid
      
# S5c:SVM 2-class: exercise 1. 3-class: exercise 2
      # They must be "recoded" into numerical variables for svm() to analyse their spatial contribution.
      # linear kernel, polynomial kernel and radial
      # test error and ROC
      summary(svmo.lin) # number of support vectors
      pred.lin = predict(svmo.lin, newdata=x.test) # test error, accuracy
      cm.lin = confusionMatrix(y.test, pred.lin)
      c(cm.lin$overall[1], cm.pol$overall[1], cm.rad$overall[1])
      # tune SVM - exercise 2
      # SVM(linear,radial) and RF with caret - exercise 3
      # SVM-based regression, straightforward cv - exercise 4
      
# S6a:NN
      nno = neuralnet(Species~., data=dat, hidden=c(6,5)) # first hidden layer 6, second 5
      nno = nnet(medv/50~., data=BostonHousing, subset=itrain, size=5) # single layer NN, normalize y
      # compare to linear model
      # effect of size on classification performance      
  
# S6b:NN   
      # fit a single-layer, 10-neuron NN: 
      out.nn = neuralnet(f, data=Boston, hidden=c(10), rep=5, linear.output=FALSE)
      # without using an activation function:
      out.nn.lin = neuralnet(f, data=Boston, hidden=c(10), rep=1, linear.output=TRUE)
      # tanh act
      out.nn.tanh = neuralnet(f, data=Boston, hidden=c(10), rep=5, linear.output=FALSE, act.fct='tanh')
      # plot NN - exercise 2
      # single layer NN in (neuralnet) and (nnet). threshold=0.001. - exercise 3
      # FFNN nnet for regression - exercise 4
      nno.ss = nnet(Salary~., data=datss.train, size=10, decay=0, linout=1)
      summary(nno.ss$fitted.values)
      # olden: variable importance and RF - exercise 6
      nno = nnet(medv~., data=dats, size=7, linout=1) # regression
      nno = nnet(Species~., data=dat, size=c(7), linout=FALSE, entropy=TRUE) # classification

# S7a:model selection
      # method=c("exhaustive","backward", "forward", "seqrep")
      reg.full = regsubsets(Salary~., data=Hitters, method="exhaustive")
      summary(reg.full)$which      # tracks covariate selection
      summary(reg.full)$outmat
      RSS = summary(reg.full)$rss
      R2adj = summary(reg.full)$adjr2 # adjusted R2
      # increase desired size to 19
      reg.full = regsubsets(Salary~., data=Hitters, nvmax=19)
      # plot heatmap
      # stepwise selection - exercise 2
      reg.fwd = regsubsets(Salary~., data=Hitters, nvmax=10, method="forward")
      reg.bwd = regsubsets(Salary~., data=Hitters, nvmax=10, method="backward")
      # predict manually from regsubsets - exercise 3
      # with train/test
      itrain = sample(1:n, 150)
      reg.fwd = regsubsets(Salary~., data=dat, nvmax=10, method="forward", subset=itrain)
      # step() backward 
      lm.out = lm(Salary~., data=Hitters) # full model
      step.bck = step(lm.out, direction="backward")
      # compare step() and regsubsets() - exercise 4
      # step() forward with split data
      itrain = sample(1:n, 150)
      lmo = lm(Salary~., data=dat, subset=itrain)
      reg.fwd = step(lmo, direction="forward")
      pred = predict(reg.fwd, newdata=dat[-itrain,]) # predict
      sqrt(mean((pred - dat$Salary[-itrain])^2)) # RMSE
      
# S7b:Feature selection
      # regression with caret - exercise 1      
      # Lasso and 10-cv
      # RFE on linear regression
      # compare to regsubset and 10-cv
      # RFE on random forest

# S8a: 
      # scaled PCA
      # k-means
      
# extra: caret data split
      itrain = createDataPartition(dat$default, p=.7, times=1)[[1]]
      # caret on logistic regression GLM
      # weighted logistic regression
      

#####CA1-1920:####################
# logistic regression, predictions, confusion, misclassification rate,
# random forest, gradient boosting, ROC AUC

#---Q1---
#(a) Diagram A can be the result of sequential dichotomous splits. It is in line with a typical partitioning of 
# the feature space following a tree growing algorithm. 
# Diagram B the "L-shape" region could not be the result of tree-based partitioning. 
# Diagram C and D contain diagonal lines, which could not have been obtained following a tree-growing process, 
# because when growing a tree, only one variable is used at each split. 

#(b) Following the majority vote approach, the model finds that 6 out of 10 predictions yield a value Yi=1. 
#    Hence the predicted label is Yi=1.
#    Following the average probability approach, we calculate the average(estimated) probability pi to be pi=0.445<0.5.
#    As such, with this approach we estimate Yi to be Yi=1.

#---Q2---
library(ISLR)
set.seed(4061) 
n = nrow(Caravan) 
# shuffle dataset first:
dat = Caravan[sample(1:n, n, replace=FALSE), ] 
dat$Purchase = as.factor(as.numeric(dat$Purchase=="Yes")) 
i.train = sample(1:n, round(.7*n), replace=FALSE) 
x.train = dat[i.train, -ncol(dat)] 
y.train = dat$Purchase[i.train] 
x.test = dat[-i.train, -ncol(dat)] 
y.test = dat$Purchase[-i.train] 

#(1) Fit a logistic regression model to the training data. Then generate predictions for the test set. Report the confusion matrix 
set.seed(4061) 
glm.mod = glm(y.train~., data=x.train, family="binomial")
glm.pred = predict(glm.mod, newdata=x.test, type="response")>0.5
# generate confusion matrix
(tb.glm = table(glm.pred, y.test))
# corresponding misclassification rate
(acc.glm = 1-sum(diag(tb.glm))/sum(tb.glm))


#(2) Fit a random forest to the training data, using 100 trees in the ensemble. Then generate predictions for the 
# test set. Report the confusion matrix 
library(randomForest)
# set.seed(4061) 
# rf.mod = randomForest(Purchase~., data=dat[i.train,], ntree=100) 
set.seed(4061) 
rf.mod = randomForest(y.train~., data=x.train, ntree=100) 
rf.pred = predict(rf.mod, newdata=x.test)
(tb.rf = table(rf.pred, y.test))
(acc.rf = 1-sum(diag(tb.rf))/sum(tb.rf))

#(3) Fit a gradient boosting model to the training data, using 100 trees in the ensemble. same predictions, confusion matrix, corresponding misclassification rate
library(gbm)
set.seed(4061) 
gb.y.train = (y.train==1)
gb.mod = gbm(gb.y.train~., data=x.train, distribution="bernoulli", n.trees=100)
gb.pred = predict(gb.mod, newdata=x.test, n.trees=100, type="response")>0.5
(tb.gb = table(gb.pred, y.test))
(acc.gb = 1-sum(diag(tb.gb))/sum(tb.gb))

#(4) Perform an ROC analysis for all three classifiers, and report the corresponding AUC values.  
library(pROC)
glm.pred = predict(glm.mod, newdata=x.test, type="response")
rf.pred = predict(rf.mod, newdata=x.test, type="prob")[,2]
gb.pred = predict(gb.mod, newdata=x.test, n.trees=100, type="response")
auc.glm = roc(y.test, glm.pred)$auc 
auc.rf = roc(y.test, rf.pred)$auc 
auc.gb = roc(y.test, gb.pred)$auc 
c(auc.glm, auc.rf ,auc.gb)
# Area under the curve from GLM: 0.704
# Area under the curve from RF: 0.651
# Area under the curve from GB: 0.730

#(5) Comment on the results you obtained in previous steps; are you satisfied with the performance of these classifiers?  
# It is not necessarily surprising to see GBM outperform the RF model.
# It is however remarkable that the GLM performs comparably to the GBM, in terms of misclassification rates.
# The GBM AUC is however higher than the AUCs of the other two models, hence GBM is the best model overall. 
# Note that we have not included confidence intervals in this assessment. 
# This may be important given the imbalance between classes in the dataset.

#(6) any concerns you may have with respect to the above use of the random forest and gradient boosting models.  
# We are using only a single data split, and not a resampling framework to evaluate model performance, so these results
# may be due to chance.
# The 2 tree-based models use a relatively small number of trees; usually such models use more trees(500 or more).
# This may explain the lack of performance of the RF in particular.
# No turning of the model hyperparameters(eg. mtry for RF or learning rate for the GLM) was carried out, so performance
# of these models could be in fact higher...

#####CA1-1819:####################
# LASSO, tree, random forest, predictions, ROC AUC

library(mlbench) 
data(Sonar)  
N = nrow(Sonar) 
P = ncol(Sonar)-1 
M = 150   
set.seed(1) 
# shuffle dataset first:
mdata = Sonar[sample(1:N),] 
itrain = sample(1:N,M) 
x = mdata[,-ncol(mdata)] 
y = mdata$Class 
xm = as.matrix(x)

#(1) How many observations are there in the test set?  
N-M
# There are N-M=58 observations in the test set.

#(2) Use cv.glmnet() in order to optimise the LASSO for the training set. 
library(glmnet)
lasso.opt = cv.glmnet(xm[itrain,], y[itrain], alpha=1, family='binomial')
lasso.opt$lambda.min
# Value of the optimal regularization parameter is 0.02527557

#(3) Fit the LASSO to the training set, using the optimal value of the regularization parameter found in (2). 
lasso.mod = glmnet(xm[itrain,], y[itrain], alpha=1, family='binomial', lambda=lasso.opt$lambda.min) 
coef(lasso.mod) 

#(4) Fit a classification tree to the training set. How many variables were used in the tree? Name these variables. 
library(tree)
tree.mod = tree(y~., data=x, subset=itrain) 
#tree.mod = tree(y[itrain]~., data=x[itrain,])  same
summary(tree.mod)
sort(summary(tree.mod)$used)
# names(summary(tree.mod))
# summary(tree.mod)$used
# ANSWER: 10 variables were actually used in tree construction:
#  "V11" "V59" "V24" "V20" "V3"  "V16" "V49" "V55" "V31" "V1" 

#(5) Fit a random forest to the training set and provide a plot of variable importance from that fit. 
library(randomForest)
rf.mod = randomForest(y~., data=x, subset=itrain) 
varImpPlot(rf.mod, pch=15) 
# Variable importance from the random forest fit: (plot)

#(6) Generate predictions from the classification tree and random forest from (4)(5). Provide the confusion matrices 
#for these predictions and quote the classification error rates for each model.  
tree.pred = predict(tree.mod, x[-itrain,], 'class') 
rf.pred = predict(rf.mod, x[-itrain,], 'class') 
(tb.tree = table(tree.pred,y[-itrain])) 
(tb.rf = table(rf.pred,y[-itrain])) 

1-sum(diag(tb.tree))/sum(tb.tree) 
1-sum(diag(tb.rf))/sum(tb.rf) 
# Classification error rate from the tree: 29.3%
# Classification error rate from the RF: 25.9%

#(7) Compute and compare the test-set AUC values for the tree and the random forest, using pROC::roc(). Which method is more accurate? 
library(pROC)
tree.p = predict(tree.mod, x[-itrain,], 'vector')[,2] 
rf.p = predict(rf.mod, x[-itrain,], 'prob')[,2] 
auc.tree = roc(y[-itrain],tree.p)$auc 
auc.rf = roc(y[-itrain],rf.p)$auc 
# auc.tree = roc(response=y[-itrain], predictor=tree.p)$auc   same
# auc.rf = roc(response=y[-itrain], predictor=rf.p)$auc
# names(roc(response=y[-itrain], predictor=tree.p))
c(auc.tree,auc.rf) 
# Test AUC from the tree: 0.742
# Test AUC from the RF: 0.874
# Here, the RF is clearly the more accurate of these two classifiers, considering both the 
# classification error rate and the AUC improve when using RF rather than a tree for classification. 

#####CA2-1819:####################
# 10-cv forward stepwise model selection, BIC, random forest, importance

#--------Q3---------
library(leaps) 
library(randomForest) 
dat = read.csv(file="C:\\R file\\ML2\\Q3_dataset.csv") 
X = dat 
X$Y <- NULL 
Y = dat$Y 

#(a)(b)(c)
set.seed(1)
N = nrow(X)
P = ncol(X)
K=10
folds = cut(1:N,10,labels=FALSE)
bics = matrix(NA,nrow=K,ncol=10)
vars = matrix(NA,nrow=K,ncol=P)
for(k in 1:K){
  itrain = which(folds!=k)
  fwd.mod = regsubsets(x=X[itrain,],y=Y[itrain], 
                       nvmax=10, method="forward")
  bics[k,] = summary(fwd.mod)$bic
  vars[k,] = summary(fwd.mod)$which[which.min(bics[k,]),-1]
}
#(b) model size
apply(bics,1,which.min)
# 3 3 3 4 4 3 3 3 3 4
#(d)
100*apply(vars,2,mean)
# X1 X2 X5 X8 are found to be important, 1 2 5 are selected every 
# time, and 8 in 30% of CV loops

#(e)
rf.out = randomForest(Y~., X)
rf.out$mtry

#(f)
rf.out$importance
nms = colnames(X)
nms[order(rf.out$importance, decreasing=TRUE)]

#####CA2-1718:####################
# forward stepwise, BIC, LASSO, optimal, tree, random forest, predictions, ROC AUC

library(mlbench) 
library(leaps)
library(tree)
library(pROC)
data(Sonar)  

N = nrow(Sonar) 
P = ncol(Sonar)-1
M = 150  # size of training set 
set.seed(1) 
mdata = Sonar[sample(1:N),] 
itrain = sample(1:N,M) 
x = as.matrix(mdata[,-ncol(mdata)]) 
y = mdata$Class 

# (1) How many observations are there in the test set? 
N-M
# N is total, M is train set, so test is the rest

#(2) fwd selection plot BIC
x = mdata[,-ncol(mdata)] 
fwd.mod = regsubsets(y[itrain]~., data = x[itrain,], 
                     method="forward", nvmax=P)
names(summary(fwd.mod))
plot(summary(fwd.mod)$bic, t='b', xlab='model size', ylab='BIC')

#(3) size of optimal model
min.bic.ind = which.min(summary(fwd.mod)$bic)
min.bic.ind
summary(fwd.mod)$outmat[min.bic.ind,]

#(4) LASSO optimal regularization parameter
lasso.opt = cv.glmnet(as.matrix(x[itrain,]),y[itrain], alpha=1,family="binomial")    
lasso.opt$lambda.min

#(5) LASSO optimal
lasso.mod = glmnet(as.matrix(x[itrain,]), y[itrain], alpha=1, lambda = lasso.opt$lambda.min,
                   family = 'binomial')
#(a) coefficients
names(lasso.mod)
cbind(round(lasso.mod$beta[,1],3))
lasso.mod$beta[which(lasso.mod$beta!=0)]
#(b) how many variables
sum(lasso.mod$beta!=0)

#(6) tree
tree.model = tree(y[itrain]~., data=(x[itrain,]))
# how many variables used
summary(tree.model)$used  
length(summary(tree.model)$used)  

#(7) random forest and importance
rf.mod = randomForest(y[itrain]~., data=x[itrain,])
varImpPlot(rf.mod)

#(8) predictions
tree.pred = predict(tree.model,  
                    newdata=as.data.frame(x[-itrain,]), type='class')
table(tree.pred,y[-itrain])

#(9) ROC AUC
tree.pred = predict(tree.model, newdata=as.data.frame(x[-itrain,]), type='vector') 
tree.roc = roc(response=y[-itrain], predictor = tree.pred[,-1])
plot(tree.roc)
tree.roc$auc

rf.pred = predict(rf.mod,x[-itrain,], type='prob') 
rf.roc = roc(response=y[-itrain], predictor = rf.pred[,1])
rf.roc$auc

#####FINAL-2021:####################
#Q2: min-max normalisation, 2 single FFNN, MSE, predictions, gradient boosting GBM,
#    GLM, ridge regression model, RFE on random forest with 10-cv, RFE on backward

#Q3: Caret - random forest 10-cv, SVM with linear, SVM with radial, importance

#####FINAL-1718:####################
#Q1: tree with Gini index, boxplots, predictions, pruning tree, random forest OOB

#Q2: LASSO with cv, predictions, scatterplot, random forest 

#Q3: random forest, ROC AUC, KNN





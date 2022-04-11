library(tree)
library(randomForest)
library(ISLR)
library(glmnet)
library(pROC)
library(class)
#----------Q1--------
M = 100 
set.seed(4061) 
dat = iris[sample(1:nrow(iris)),] 
dat[,1:4] = apply(dat[,1:4],2,scale) 
itrain = sample(1:nrow(iris), M) 

#(a) tree with gini index
dat.train = dat[itrain,] 
tree.o = tree(Species~., data=dat.train, split="gini")
summary(tree.o)
#(i)  "Petal.Width"  "Petal.Length" "Sepal.Length" 
#(ii) Number of terminal nodes:  7 
#(iii) Misclassification error rate: 0.04 = 4 / 100 
# draw the tree
plot(tree.o)
text(tree.o,pretty=NULL)

#(b)
# Petal.Width and Petal.Length are used. 

#(c) boxplots
par(mfrow=c(2,2))
for(k in 1:4){
  boxplot(dat[,k]~dat[,5], main=paste(names(dat)[k]))
}
# the Petal information is more clearly separated per species

#(e) prediction for test set
dat.test = dat[-itrain,] 
tree.pred = predict(tree.o, dat.test, type="class")
#(i) confusion matrix
(tb = table(tree.pred,dat.test$Species))
#(ii) prediction error rate
1-sum(diag(tb))/sum(tb)

#(f) pruning the tree
par(mfrow=c(1,1))
cv.o = cv.tree(tree.o, FUN=prune.misclass)
plot(cv.o$size, cv.o$dev, t='b')
opt.size = cv.o$size[which.min(cv.o$dev)]
ptree = prune.misclass(tree.o, best=opt.size)
plot(ptree)
text(ptree)
summary(ptree)

#(g) random forest
rf.o = randomForest(Species~., dat.train)
rf.o
# out-of-bag error rate: OOB estimate of  error rate: 7%

#(h) predictions for test set
rf.p = predict(rf.o, dat.test, type="class")
#(i) confusion matrix
(rf.tb = table(rf.p,dat.test$Species))
#(ii) prediction error rate
1-sum(diag(rf.tb))/sum(rf.tb)

#(i)Why could we not generate an ROC curve for this classification tree? 
# ROC for binary classification only. there are three classes here. 

#----------Q2--------
dat = model.matrix(Apps~., College)[,-1]  
dat <- apply(dat,2,scale) 
set.seed(4061) 
itrain = sample(1:nrow(dat), 500) 

#(a) regression

#(b) LASSO
y = College$Apps
set.seed(4061) 
lasso.opt = cv.glmnet(dat[itrain,], y[itrain], alpha=1)
#(i) opt regular parameter
lasso.opt$lambda.min
#(ii) coefficient estimates
lasso.mod = glmnet(dat[itrain,], y[itrain], alpha=1,
                         lambda=lasso.opt$lambda.min)
coef(lasso.mod)

#(c) predictions for LASSO
dat.test = dat[-itrain,]
ytrue = y[-itrain]
lasso.pred = predict(lasso.mod, dat.test)
#(i) RMSE
sqrt(mean((ytrue-lasso.pred)^2))
#(ii) correlation
cor(lasso.pred, ytrue)
#(iii) scatterplot
plot(lasso.pred, ytrue)
abline(a=0,b=1)

#(d) random forest
dat.train = dat[itrain,]
set.seed(4061) 
rf.mod5 = randomForest(y[itrain]~., dat.train, mtry=5)
rf.mod15 = randomForest(y[itrain]~., dat.train, mtry=15)
# RMSE ################################
#rf.p5 = predict(rf.mod5, dat.test, type="class")
#rf.p15 = predict(rf.mod15, dat.train, type="class")
sqrt(mean((rf.mod5$predicted-rf.mod5$y)^2))
sqrt(mean((rf.mod15$predicted-rf.mod15$y)^2))

#(e) predictions for test set
rf.pred15 = predict(rf.mod15, dat.test, type="class")
# RMSE
sqrt(mean((rf.pred15-y[-itrain])^2))

#----------Q3--------
x = Smarket[,-9] 
y = Smarket$Direction 
set.seed(4061) 
train = sample(1:nrow(Smarket),1000) 

#(a) random forest
dat.train = x[train,]
dat.test = x[-train,]
rf.o = randomForest(y[train]~., dat.train)
summary(rf.o)

#(b) prediction for 250 test and plot ROC, quote AUC
rf.p = predict(rf.o, dat.test, type="prob")
roc.rf = roc(response=y[-train], predictor=rf.p[,1]) #??????
roc.rf$auc
plot(roc.rf)

#(c) KNN ROC AUC
ko = knn(dat.train, dat.test, cl=y[train], k=2)
kop = knn(dat.train, dat.test, cl=y[train], k=2, prob=T)
kopp = attributes(kop)$prob
roc.knn = roc(response=y[-train], predictor=kopp)
roc.knn$auc
plot(roc.knn, add=TRUE, col=4)

#(d) knn from 1 to 10
set.seed(4061) 
M = 1000 
train = sample(1:nrow(Smarket), M) 
knno = numeric(10)
for(k in c(1:10)){
  ko = knn(x[train,], x[-train,], cl=y[train], k=k)
  tbo = table(ko, y[-train])
  knno[k] = sum(diag(tbo))/sum(tbo)
}
plot(knno, t='b')




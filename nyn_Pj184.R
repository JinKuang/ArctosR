## Nguyen - Projet Data Science dec2017##
setwd("C:/Users/ibm/Clouding/OneDrive/esilvwd/FolderR")
require(data.table)
require(dummies)
require(caret)
require(pROC)
require(ggthemes)
require(dismo)
require(ggplot2) : require(ggthemes) ; tuft<-theme_tufte()

projetTrain<-fread("p2train8.csv") 
projetValid<-fread("valid8.csv")
set.seed(0)#tirage au sort pour le sample

# UNIFICATION FICHIERS TRAIN ET VALID#
set.seed(0) ;eval<-sample(1:dim(projetTrain)[1],8000) #fichier eval 
# consolidation train et valid our la dataprep
projetValid[,target:=3] # différencier train et valid - on rajoute la colonne target=3
pp<-rbind(projetTrain,projetValid)
pp<-pp[,-1]#on retire la variable V1 qui contient 80% de valeurs différentes, donc non discriminante

# DONNEES NA#
na.colums<-names(which(sapply(pp,function(x) sum(is.na(x))>0)))
for (c in na.colums) {
pp[is.na(get(c)),c:=median(t(projetTrain[!is.na(get(c)),c,with=FALSE])),with=FALSE]}

# CONVERT CHAR TO FACTOR
chr<-names(which(sapply(pp,class)=="character")) ; for (c in chr) pp[,(c):=as.factor(get(c)),with=FALSE]

# DEFINE pp.train eval valid X y
pp.valid<-pp[target==3]
pp.valid.x<-copy(pp.valid) ;pp.valid.x[,target:=NULL] ;
pp.eval<-pp[eval]
pp.eval.x<-copy(pp[eval]) ;pp.eval.x[,target:=NULL] ; pp.eval.y<-pp.eval$target
pp.train<-pp[target!=3] ; pp.train<-pp.train[-eval]
pp.train.x<-copy(pp.train) ; pp.train.x[,target:=NULL] ; pp.train.y<-pp.train[,target]
pp.TRAIN<-pp[target!=3]
pp.TRAIN.x<-copy(pp.TRAIN) ; pp.TRAIN.x[,target:=NULL] ; pp.TRAIN.y<-pp.TRAIN[,target]

# CONVERT TO matrix & data.frame
pp.mat<-copy(pp)

pp.mat.y<-pp.mat$target ; pp.mat.x<-copy(pp.mat) ; pp.mat.x[,target:=NULL] ; 

pp.mat.valid<-pp.mat[pp.mat.y==3,] 
pp.mat.train<-pp.mat[pp.mat.y!=3,] ; 
pp.mat.eval<-pp.mat.train[eval,] ; 
pp.mat.train<-pp.mat.train[-eval,] ; 

pp.mat.train.y<-pp.mat.train$target 
pp.mat.train.x<-as.matrix(pp.mat.train[,setdiff(colnames(pp.mat.train),"target"),with=FALSE])
pp.mat.eval.y<-pp.mat.eval$target
pp.mat.eval.x<-as.matrix(pp.mat.eval[,setdiff(colnames(pp.mat.eval),"target"),with=FALSE])
pp.mat.valid.y<-pp.mat.valid$target
pp.mat.valid.x<-as.matrix(pp.mat.valid[,setdiff(colnames(pp.mat.valid),"target"),with=FALSE])
pp.mat.TRAIN<-pp.mat[target!=3,] 
pp.mat.TRAIN.y<-pp.mat.TRAIN$target
pp.mat.TRAIN.x<-as.matrix(pp.mat.TRAIN[,setdiff(colnames(pp.mat.TRAIN),"target"),with=FALSE])

pp.matr.train<-model.matrix(~.+0,data=pp.mat.train)
pp.matrf.train<-as.data.frame(pp.matr.train)
pp.matr.train.x<- pp.matr.train[,-1]                                      

pp.matr.TRAIN<-model.matrix(~.+0,data=pp.mat.TRAIN)
pp.matr.TRAIN.x<- pp.matr.TRAIN[,-1]

#RANDOM FOREST

require(randomForest)
require(lift)
# test sur différents nTree
regl<-c(); mtry<-30; for (nTree in c(30,60,100)){
Mod.rf<-randomForest(as.factor(target)~.,data=pp.train,ntree=nTree,mtry=mtry,importance=TRUE) ; 
pred.rf<-predict(Mod.rf, pp.eval.x,type="prob")[,2]
auc<-roc(as.numeric(pp.eval.y),as.vector(predict(Mod.rf, pp.eval.x,type="prob")[,2]))$auc
print(c(nTree,auc)) ;  
regl<-rbind(regl,c(nTree,auc))
Regl<-as.data.table(regl) ; setnames(Regl,c("nTree","auc")) ; write.csv(Regl,"reglnTreeRF.csv")
}

  #Analyse du modèle avec nTree=100
  mtry<-30; nTree<-100
  Mod.rf<-randomForest(as.factor(target)~.,data=pp.train,ntree=nTree,mtry=mtry,importance=TRUE) ; 
  pred.rf<-predict(Mod.rf, pp.eval.x,type="prob")
  predt.rf<-c()
  predt.rf<-as.data.table(pred.rf)
  setnames(predt.rf,c("0","1"))
  str(predt.rf)
  head(predt.rf)
  table(predt.rf$`1`)
  table(predt.rf$`0`)
  
  auc<-roc(as.numeric(pp.eval.y),as.vector(predict(Mod.rf,pp.eval.x,type="prob")[,2]))$auc
  print(c(nTree,auc)) ; 
  #[1] nTree= 100; AUC = 0.5931188
  #voir résultat complets 96
   
  #Application du lift  
  pl<-plotLift(predt.rf,pp.eval.y, cumulative = TRUE, n.buckets = 10)
  #Voir plot 234
  
  #application de predict à valid 
  predv.rf<-predict(Mod.rf, pp.valid.x,type="prob")
  predv.rf<-data.frame(pp.valid.x$id,predict(Mod.rf, pp.valid.x,type="prob")[,2])
  predi.rf<-data.frame(pp.valid.x$id,predv.rf)
  predi.rf<-round(predi.rf,digits = 2)
  predi.rf<-setNames(predv.rf,c("Id","Predict"))
  write.csv(predi.gbmG,"Predict_nguyen_RandFor1.csv",row.names = FALSE)

  
# GBM##
require(dismo)
mod.gbmG <- gbm.step(data=pp.TRAIN, gbm.x = which(names(pp.TRAIN)!="target"), gbm.y = which(names(pp.TRAIN)=="target"), family = "bernoulli", 
                     tree.complexity = 5,learning.rate = 0.01, bag.fraction = 0.6,n.folds=5,max.trees=1500,verbose=0)
#voir 307 results: fitting terminé à 1350 trees

pred.gbmG<-predict(mod.gbmG,n.trees=mod.gbmG$n.trees,pp.valid.x,type="response")
predi.gbmG<-data.frame(pp.valid.x$id,pred.gbmG)
setnames(predi.gbmG,c("Id","Predict"))
write.csv(predi.gbmG,"Predict_nguyen_gbm1.csv",row.names = FALSE)

ro<-round(pred.gbmG,digits = 2)
str(ro)
table(ro)
# voir 307 results: 1 évenement =1 sur 20 000

predeval.gbmG<-predict(mod.gbmG,n.trees=mod.gbmG$n.trees,pp.eval.x,type="response")
auc_gbm<-roc(as.numeric(pp.eval.y),as.vector(predict(mod.gbmG,n.trees=mod.gbmG$n.trees,pp.eval.x,type="response")[,2]))$auc
print(c(auc_gbm)) 


#REGRESSION LOGISTIQUE
library(stats)
GL<-glm(target~.,data=pp.train, family = binomial) 
pred.glm<-predict(GL,newdata=pp.valid.x,type ="response")
predr.glm<-round(pred.glm,digits=2)
table(predr.glm)
# voir results 135
predi.glm<-data.frame(pp.valid.x$id,predr.glm)
predi.glm<-setNames(predi.glm,c("Id","Predict"))
write.csv(predi.glm,"Predict_nguyen_glm1.csv", row.names = FALSE)

predeval.glm<-predict(GL,newdata=pp.eval.x,type ="response")
auc_glm<-roc(as.numeric(pp.eval.y),as.vector(predict(GL,newdata=pp.eval.x,type ="response")))$auc
print(c(auc_glm))  
#AUC = [1] 0.6127446










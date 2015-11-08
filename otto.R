install.packages("MASS")
library(MASS)
install.packages("klaR")
library(klaR)
require("e1071") # for SVM
library("e1071")
require("kernlab") # for k-SVM
library("kernlab")
install.packages("randomForest")
library(randomForest)

#setwd("~/otto-clustering")
setwd("C:/Users/rms15/Desktop/otto-clustering")
#setwd("~/Rice/Kaggle/otto-clustering")

otto.train<-read.csv("train-2.csv")
#nrow(otto.train)
#unique(otto.train$target)
otto.test<-read.csv("test-2.csv")
#nrow(otto.test)

prob.target<-table(otto.train$target)/nrow(otto.train)
###########################################################
##  Model building : QDA
###########################################################
qda.mod1<-MASS:::qda(target~feat_1+feat_2+feat_3+feat_4+feat_5+feat_6+feat_7+feat_8+feat_9+feat_10+feat_11
                     +feat_12+feat_13+feat_14+feat_15+feat_16+feat_17+feat_18+feat_19+feat_20+feat_21
                     +feat_22+feat_23+feat_24+feat_25+feat_26+feat_27+feat_28+feat_29+feat_30+feat_31+
                       feat_32+feat_33+feat_34+feat_35+feat_36+feat_37+feat_38+feat_39+feat_40+feat_41
                     +feat_42+feat_43+feat_44+feat_45+feat_46+feat_47+feat_48+feat_49+feat_50+feat_51+
                       feat_52+feat_53+feat_54+feat_55+feat_56+feat_57+feat_58+feat_59+feat_60+feat_61
                     +feat_62+feat_63+feat_64+feat_65+feat_66+feat_67+feat_68+feat_69+feat_70+feat_71+
                       feat_72+feat_73+feat_74+feat_75+feat_76+feat_77+feat_78+feat_79+feat_80+feat_81
                     +feat_82+feat_83+feat_84+feat_85+feat_86+feat_87+feat_88+feat_89+feat_90+feat_91+
                       feat_92+feat_93, data=otto.train,prior=prob.target)

stepqda.mod1<-stepclass(target~feat_1+feat_2+feat_3+feat_4+feat_5+feat_6+feat_7+feat_8+feat_9+feat_10+feat_11
                        +feat_12+feat_13+feat_14+feat_15+feat_16+feat_17+feat_18+feat_19+feat_20+feat_21
                        +feat_22+feat_23+feat_24+feat_25+feat_26+feat_27+feat_28+feat_29+feat_30+feat_31+
                          feat_32+feat_33+feat_34+feat_35+feat_36+feat_37+feat_38+feat_39+feat_40+feat_41
                        +feat_42+feat_43+feat_44+feat_45+feat_46+feat_47+feat_48+feat_49+feat_50+feat_51+
                          feat_52+feat_53+feat_54+feat_55+feat_56+feat_57+feat_58+feat_59+feat_60+feat_61
                        +feat_62+feat_63+feat_64+feat_65+feat_66+feat_67+feat_68+feat_69+feat_70+feat_71+
                          feat_72+feat_73+feat_74+feat_75+feat_76+feat_77+feat_78+feat_79+feat_80+feat_81
                        +feat_82+feat_83+feat_84+feat_85+feat_86+feat_87+feat_88+feat_89+feat_90+feat_91+
                          feat_92+feat_93,data=otto.train,prior=prob.target,method="qda")
###########################################################
##  Model building : SVM with cross-validation
###########################################################
xtrain<-otto.train[,!(names(otto.train) %in% c("id","target"))]
colnames(xtrain)
svm.mod1<-ksvm(target~feat_1+feat_2+feat_3+feat_4+feat_5+feat_6+feat_7+feat_8+feat_9+feat_10+feat_11
               +feat_12+feat_13+feat_14+feat_15+feat_16+feat_17+feat_18+feat_19+feat_20+feat_21
               +feat_22+feat_23+feat_24+feat_25+feat_26+feat_27+feat_28+feat_29+feat_30+feat_31+
                 feat_32+feat_33+feat_34+feat_35+feat_36+feat_37+feat_38+feat_39+feat_40+feat_41
               +feat_42+feat_43+feat_44+feat_45+feat_46+feat_47+feat_48+feat_49+feat_50+feat_51+
                 feat_52+feat_53+feat_54+feat_55+feat_56+feat_57+feat_58+feat_59+feat_60+feat_61
               +feat_62+feat_63+feat_64+feat_65+feat_66+feat_67+feat_68+feat_69+feat_70+feat_71+
                 feat_72+feat_73+feat_74+feat_75+feat_76+feat_77+feat_78+feat_79+feat_80+feat_81
               +feat_82+feat_83+feat_84+feat_85+feat_86+feat_87+feat_88+feat_89+feat_90+feat_91+
                 feat_92+feat_93,data=otto.train,type="nu-svc",kernel="rbfdot",
               kpar = "automatic",
               cross=5,nu=0.2,prob.model=TRUE)
###########################################################
##  Model building : randomForest
###########################################################
rf.mod1<-randomForest(target~feat_1+feat_2+feat_3+feat_4+feat_5+feat_6+feat_7+feat_8+feat_9+feat_10+feat_11
                      +feat_12+feat_13+feat_14+feat_15+feat_16+feat_17+feat_18+feat_19+feat_20+feat_21
                      +feat_22+feat_23+feat_24+feat_25+feat_26+feat_27+feat_28+feat_29+feat_30+feat_31+
                        feat_32+feat_33+feat_34+feat_35+feat_36+feat_37+feat_38+feat_39+feat_40+feat_41
                      +feat_42+feat_43+feat_44+feat_45+feat_46+feat_47+feat_48+feat_49+feat_50+feat_51+
                        feat_52+feat_53+feat_54+feat_55+feat_56+feat_57+feat_58+feat_59+feat_60+feat_61
                      +feat_62+feat_63+feat_64+feat_65+feat_66+feat_67+feat_68+feat_69+feat_70+feat_71+
                        feat_72+feat_73+feat_74+feat_75+feat_76+feat_77+feat_78+feat_79+feat_80+feat_81
                      +feat_82+feat_83+feat_84+feat_85+feat_86+feat_87+feat_88+feat_89+feat_90+feat_91+
                        feat_92+feat_93,data=otto.train)
varImpPlot(rf.mod1)
summary(trainK)
predK<-predict(trainK,xtest)

###############################################################
## Training accuracy using qda ### 0.673
###############################################################
predict.train<-MASS:::predict.qda(qda.mod1,newdata=otto.train)
length(which(as.character(predict.train$class) == as.character(otto.train$target)))/nrow(otto.train)
##
predict.qda.test<-MASS:::predict.qda(qda.mod1,newdata=otto.test)
str(predict.qda.test)
length(predict.qda.test$class)
#rm(predict.rf)
str(predict.rf)
predict.rf<-data.frame(cbind(otto.test$id,predict.qda.test))
colnames(predict.rf)<-c("id","prediction")
###############################################################
## Training accuracy using rf ### 0.673
###############################################################
predict.rf.train<-predict(rf.mod1,newdata=otto.train)
length(which(as.character(predict.rf.train) == as.character(otto.train$target)))/nrow(otto.train)
#0.99 (overfitted)
rf.mod1$forest
str(rf.mod1)
# tune RF
sample(length(otto.train),100)
kfolds = 1
cum.max.acc=0
accuracy=numeric()
acc.pc = matrix(NA,50,50) 
max.acc=0
for (num_nodes in seq(500,1700,500))
{
  j = 1
  for  (num_mtry in seq(10,60,15))
  {
    print("number of nodes:")
    print(num_nodes)
    print("number of variables sampled for split:")
    print(num_mtry)
    kfolds=1
    for (kfolds in (1:4)) 
    {
      smp <- sample(nrow(cv.data),nrow(cv.data)/8)
      cv.train = cv.data[-smp,]
      cv.test = cv.data[smp,]
      rf.cv.train<-randomForest(target~feat_1+feat_2+feat_3+feat_4+feat_5+feat_6+feat_7+feat_8+feat_9+feat_10+feat_11
                            +feat_12+feat_13+feat_14+feat_15+feat_16+feat_17+feat_18+feat_19+feat_20+feat_21
                            +feat_22+feat_23+feat_24+feat_25+feat_26+feat_27+feat_28+feat_29+feat_30+feat_31+
                              feat_32+feat_33+feat_34+feat_35+feat_36+feat_37+feat_38+feat_39+feat_40+feat_41
                            +feat_42+feat_43+feat_44+feat_45+feat_46+feat_47+feat_48+feat_49+feat_50+feat_51+
                              feat_52+feat_53+feat_54+feat_55+feat_56+feat_57+feat_58+feat_59+feat_60+feat_61
                            +feat_62+feat_63+feat_64+feat_65+feat_66+feat_67+feat_68+feat_69+feat_70+feat_71+
                              feat_72+feat_73+feat_74+feat_75+feat_76+feat_77+feat_78+feat_79+feat_80+feat_81
                            +feat_82+feat_83+feat_84+feat_85+feat_86+feat_87+feat_88+feat_89+feat_90+feat_91+
                              feat_92+feat_93,data=otto.train)
      maxnodes=num_nodes,mtry=num_mtry)

predict.test.cv<-predict(rf.train.cv,newdata=cv.test)
predict.test.cv2 <- as.numeric(as.character(predict.test.cv))
accuracy[kfolds] = length(which(predict.test.cv2 == cv.test$label))/nrow(cv.test)
    }
acc.pc[i,j] = mean(accuracy)
print("Average accuracy for this param set")
print(acc.pc[i,j])
if (max.acc < acc.pc[i,j]) {
  print("Max increased, Best params changed")
  max.acc = acc.pc[i,j]
  best.num_nodes = num_nodes
  best.num_mtry = num_mtry
  print(max.acc)
  print(best.num_nodes)
  print(best.num_mtry)
}
j = j+1
  }
i = i+1
}

predict.rf.test<-predict(rf.mod1,newdata=otto.test)
predict.rf<-data.frame(cbind(otto.test$id,predict.rf.test))
colnames(predict.rf)<-c("id","prediction")
###############################################################
# Create prediction file for test dataset
###############################################################
predict.rf$Class_1 = 0
predict.rf$Class_2 = 0
predict.rf$Class_3 = 0
predict.rf$Class_4 = 0
predict.rf$Class_5 = 0
predict.rf$Class_6 = 0
predict.rf$Class_7 = 0
predict.rf$Class_8 = 0
predict.rf$Class_9 = 0
str(predict.rf)
predict.rf$Class_1[which(predict.rf$prediction == 1)] = 1
predict.rf$Class_2[which(predict.rf$prediction == 2)] = 1
predict.rf$Class_3[which(predict.rf$prediction == 3)] = 1
predict.rf$Class_4[which(predict.rf$prediction == 4)] = 1
predict.rf$Class_5[which(predict.rf$prediction == 5)] = 1
predict.rf$Class_6[which(predict.rf$prediction == 6)] = 1
predict.rf$Class_7[which(predict.rf$prediction == 7)] = 1
predict.rf$Class_8[which(predict.rf$prediction == 8)] = 1
predict.rf$Class_9[which(predict.rf$prediction == 9)] = 1
#colnames(predict.rf)<-c("id","Class_1","Class_2","Class_3","Class_4",
#                        "Class_5","Class_6","Class_7","Class_8","Class_9")

write.csv(predict.rf,"predict_rf.csv")


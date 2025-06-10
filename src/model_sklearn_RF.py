import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import functions as fn
import joblib

##########################################################
# Eğitim 
##########################################################

train_X = pd.read_csv("data/processed/3_class/6000_train_X.csv")
train_y = pd.read_csv("data/processed/3_class/6000_train_y.csv")
train_y = train_y["Alert"]


RF = RandomForestClassifier(max_samples=200,max_leaf_nodes=5) #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
RF.fit(train_X, train_y)
joblib.dump(RF,"models/sklearn/RF.pkl")

##########################################################
# Özel olarak test
##########################################################

from sklearn.metrics import root_mean_squared_error,r2_score,mean_absolute_error,classification_report

test_X = pd.read_csv("data/processed/3_class/6000_test_X.csv")
test_y = pd.read_csv("data/processed/3_class/6000_test_y.csv")

class_names = ["OK","Da","Dw"] # 3 sınıflı eğitimler ve sonuçlar için
# class_names = ["OK","Fault"] # 2 sınıflı eğitimler ve sonuçlar için

pred_train = RF.predict(train_X)
pred_prob_train = RF.predict_proba(train_X)
fn.roc_curve_plot(train_y,pred_prob_train,"RF Train",class_names)
fn.conf_mat(train_y,pred_train,class_names=class_names,Model_Name="RF Train")
print(f"RMSE : {root_mean_squared_error(train_y,pred_train)}\nR2 : {r2_score(train_y,pred_train)}\nMSE : {mean_absolute_error(train_y,pred_train)}")
report = classification_report(train_y,pred_train,target_names=class_names,digits=5)
print(report)

pred = RF.predict(test_X)
pred_prob = RF.predict_proba(test_X)
fn.roc_curve_plot(test_y,pred_prob,"RF Test",class_names)
fn.conf_mat(test_y,pred,class_names=class_names,Model_Name="RF Test")
print(f"RMSE : {root_mean_squared_error(test_y,pred)}\nR2 : {r2_score(test_y,pred)}\nMSE : {mean_absolute_error(test_y,pred)}")
report = classification_report(test_y,pred,target_names=class_names,digits=5)
print(report)


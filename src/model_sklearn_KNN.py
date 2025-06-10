import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import functions as fn
import joblib

##########################################################
# Eğitim 
##########################################################

train_X = pd.read_csv("data/processed/3_class/6000_train_X.csv")
train_y = pd.read_csv("data/processed/3_class/6000_train_y.csv")
train_y = train_y["Alert"]

KNN = KNeighborsClassifier() #https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
KNN.fit(train_X,train_y)
joblib.dump(KNN,"models/sklearn/KNN.pkl")

##########################################################
# Özel olarak test
##########################################################

# from sklearn.metrics import root_mean_squared_error,r2_score,mean_absolute_error,classification_report

# test_X = pd.read_csv("data/processed/3_class/6000_test_X.csv")
# test_y = pd.read_csv("data/processed/3_class/6000_test_y.csv")

# class_names = ["OK","Da","Dw"] # 3 sınıflı eğitimler ve sonuçlar için
# class_names = ["OK","Fault"] # 2 sınıflı eğitimler ve sonuçlar için

# pred = KNN.predict(test_X)
# pred_prob = KNN.predict_proba(test_X)
# fn.roc_curve_plot(test_y,pred_prob,"KNN deneme",class_names)
# fn.conf_mat(test_y,pred,class_names=class_names,Model_Name="KNN deneme")
# print(f"RMSE : {root_mean_squared_error(test_y,pred)}\nR2 : {r2_score(test_y,pred)}\nMSE : {mean_absolute_error(test_y,pred)}")
# report = classification_report(test_y,pred,target_names=class_names,digits=5)
# print(report)
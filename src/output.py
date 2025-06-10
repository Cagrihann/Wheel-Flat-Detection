import pandas as pd
import numpy as np
import joblib
import functions as fn
import os 
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras import models
from model_pythorch_ANN import ANN
from sklearn.metrics import classification_report,r2_score,mean_absolute_error,root_mean_squared_error

train_X = pd.read_csv("data/processed/3_class/6000_train_X.csv")
train_y = pd.read_csv("data/processed/3_class/6000_train_y.csv")
test_X = pd.read_csv("data/processed/3_class/6000_test_X.csv")
test_y = pd.read_csv("data/processed/3_class/6000_test_y.csv")
train_X_tensor = torch.tensor(train_X.values,dtype=torch.float32)
test_X_tensor = torch.tensor(test_X.values,dtype=torch.float32)

ACC = {}

##########################################################
# Tüm Modellerin Eğitilmesi
##########################################################

folder_path =  r"C:\Users\ccari\OneDrive\Masaüstü\DERS\codes\py\Machine Learning\vscode\Bitirme\src"
class_names = ["OK","Da","Dw"]
#class_names = ["OK","Fault"] # 2 sınıflı eğitimler ve sonuçlar için

for file in os.listdir(folder_path):
    if file.startswith("model"):
        file_path = os.path.join(folder_path, file)
        print(f"Basladi: {file}\n")
        os.system(f'python "{file_path}"') # os.system()direkt sisteme yazmak için
        print(f"Bitti\n")

#########################################################
# Sklearn modellerinin ve XGBClassifier'ın test edilmesi 
#########################################################

models_dict = {
    "Random Forest": "models/sklearn/RF.pkl",
    "SCV": "models/sklearn/SVC.pkl",
    "MLP10x10x10": "models/sklearn/MLP10x10x10.pkl",
    "KNN": "models/sklearn/KNN.pkl",
    "GaussianNaiveBayes": "models/sklearn/GNaiveB.pkl",
    "XGBoost": "models/sklearn/XGBC.pkl",
}

for idx,(name, path) in enumerate(models_dict.items()):
    print(f"#############\nModel:{name} Başladi")
    model_N = joblib.load(path)
    test_pred = model_N.predict(test_X)
    train_pred = model_N.predict(train_X)
    test_pred_prob = model_N.predict_proba(test_X)
    auc = fn.roc_curve_plot(test_y,test_pred_prob,name,class_names)
    fn.conf_mat(test_y,test_pred,class_names=class_names,Model_Name=name)
    report = classification_report(test_y,test_pred,output_dict=True)
    ACC[name] = {
        "Accuracy": report["accuracy"],
        "Precision": report["macro avg"]["precision"],
        "Sensivity": report["macro avg"]["recall"],
        "F-measure": report["macro avg"]["f1-score"],
        "ROC area": pd.Series(auc).mean(),
        "R2_test": r2_score(test_y,test_pred),
        "MAE_test": mean_absolute_error(test_y,test_pred),
        "RMSE_test": root_mean_squared_error(test_y,test_pred),
        "R2_train": r2_score(train_y,train_pred),
        "MAE_train": mean_absolute_error(train_y,train_pred),
        "RMSE_train": root_mean_squared_error(train_y,train_pred)
    }
    print(f"tamamlandi\n#############")


##########################################################
# Pytorch ile oluşturulan modelin test edilmesi
##########################################################    

ANN_py = ANN()
ANN_py.load_state_dict(torch.load(f="models/pytorch/ANN10x10x10.pth",weights_only=True))
ANN_py.eval()
with torch.inference_mode():
    torch_test_pred_proba = ANN_py(test_X_tensor) #çünkü softmax kullanıldı
    torch_test_pred_class = torch.argmax(torch_test_pred_proba, dim=1)
    torch_train_pred_proba = ANN_py(train_X_tensor)
    torch_train_pred_class = torch.argmax(torch_train_pred_proba, dim=1)
auc = fn.roc_curve_plot(test_y,torch_test_pred_proba,"PyTorch 10x10x10",class_names)
fn.conf_mat(test_y,torch_test_pred_class,class_names=class_names,Model_Name="PyTorch 10x10x10")
report = classification_report(test_y,torch_test_pred_class,output_dict=True)
ACC["PyTorch 10x10x10"] = {
        "Accuracy": report["accuracy"],
        "Precision": report["macro avg"]["precision"],
        "Sensivity": report["macro avg"]["recall"],
        "F-measure": report["macro avg"]["f1-score"],
        "ROC area": pd.Series(auc).mean(),
        "R2_test": r2_score(test_y,torch_test_pred_class),
        "MAE_test": mean_absolute_error(test_y,torch_test_pred_class),
        "RMSE_test": root_mean_squared_error(test_y,torch_test_pred_class),
        "R2_train": r2_score(train_y,torch_train_pred_class),
        "MAE_train": mean_absolute_error(train_y,torch_train_pred_class),
        "RMSE_train": root_mean_squared_error(train_y,torch_train_pred_class)
    }

##########################################################
# Tensorflow ile oluşturulan modelin test edilmesi
##########################################################    

ANN_tf = models.load_model("models/tensorflow/ANN10x10x10.keras")
tensorflow_test_pred_proba = ANN_tf.predict(test_X) #çünkü softmax kullanıldı
tensorflow_test_pred_class = np.argmax(tensorflow_test_pred_proba, axis=1) #sınıflara ayırma işlemi
tensorflow_train_pred_proba = ANN_tf.predict(train_X) 
tensorflow_train_pred_class = np.argmax(tensorflow_train_pred_proba, axis=1) 
auc = fn.roc_curve_plot(test_y,tensorflow_test_pred_proba,"Tensorflow 10x10x10",class_names)
fn.conf_mat(test_y,tensorflow_test_pred_class,class_names=class_names,Model_Name="Tensorflow 10x10x10")
report = classification_report(test_y,tensorflow_test_pred_class,output_dict=True)
ACC["Tensorflow 10x10x10"] = {
        "Accuracy": report["accuracy"],
        "Precision": report["macro avg"]["precision"],
        "Sensivity": report["macro avg"]["recall"],
        "F-measure": report["macro avg"]["f1-score"],
        "ROC area": pd.Series(auc).mean(),
        "R2_test": r2_score(test_y,tensorflow_test_pred_class),
        "MAE_test": mean_absolute_error(test_y,tensorflow_test_pred_class),
        "RMSE_test": root_mean_squared_error(test_y,tensorflow_test_pred_class),
        "R2_train": r2_score(train_y,tensorflow_train_pred_class),
        "MAE_train": mean_absolute_error(train_y,tensorflow_train_pred_class),
        "RMSE_train": root_mean_squared_error(train_y,tensorflow_train_pred_class)
    }

##########################################################
# Sonuçların kaydedilmesi
##########################################################    

acc = pd.DataFrame.from_dict(ACC, orient='index')
# acc.columns = ['Model', 'Accuracy', 'Precision', 'Sensivity', 'F-measure']
acc.to_excel("output/accuracy_scores.xlsx")
from tensorflow import keras
from tensorflow.keras import layers,optimizers
import pandas as pd

from tensorflow.keras import models


train_X = pd.read_csv("data/processed/3_class/6000_train_X.csv")
train_y = pd.read_csv("data/processed/3_class/6000_train_y.csv")
test_X = pd.read_csv("data/processed/3_class/6000_test_X.csv")
test_y = pd.read_csv("data/processed/3_class/6000_test_y.csv")

ANN = keras.Sequential([
    layers.Input(shape=(5,)),  
    layers.Dense(10, activation='relu'),  
    layers.Dense(10, activation='relu'), 
    layers.Dense(10, activation='relu'), 
    layers.Dense(3, activation='softmax') 
])

optimizer = optimizers.Adam(learning_rate = 0.01)


ANN.compile(optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

history = ANN.fit(train_X,train_y,epochs = 10) 

ANN.save("models/tensorflow/ANN10x10x10.keras") 

##########################################################
# Özel olarak test
##########################################################    

# import functions as fn
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import root_mean_squared_error,r2_score,mean_absolute_error,classification_report

# class_names = ["OK","Da","Dw"] # 3 sınıflı eğitimler ve sonuçlar için
# class_names = ["OK","Fault"] # 2 sınıflı eğitimler ve sonuçlar için

# ANN_tf = models.load_model("models/tensorflow/ANN10x10x10.keras")
# tensorflow_test_pred_proba = ANN_tf.predict(test_X) #çünkü softmax kullanıldı
# tensorflow_test_pred_class = np.argmax(tensorflow_test_pred_proba, axis=1) #sınıflara ayırma işlemi
# fn.roc_curve_plot(test_y,tensorflow_test_pred_proba,"Tensorflow deneme",class_names)
# fn.conf_mat(test_y,tensorflow_test_pred_class,class_names=class_names,Model_Name="Tensorflow deneme")
# print(f"RMSE : {root_mean_squared_error(test_y,tensorflow_test_pred_class)}\nR2 : {r2_score(test_y,tensorflow_test_pred_class)}\nMSE : {mean_absolute_error(test_y,tensorflow_test_pred_class)}")
# report = classification_report(test_y,tensorflow_test_pred_class,target_names=class_names,digits=5)
# print(report)
# loss=history.history['loss']
# epochs=range(1,len(loss)+1)
# plt.figure(figsize=(8,5))
# plt.plot(epochs,loss,label="LOSS")
# plt.xlabel("EPOCH")
# plt.ylabel("LOSS")
# plt.title("LOSSGRAPH")
# plt.legend()
# plt.grid()
# plt.show()
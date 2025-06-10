import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def ayir_kaydet(data,n_class,Name):
    data_y = data["Alert"]
    data_X = data.drop("Alert",axis=1)
    Scaler_X = MinMaxScaler()
    # data_X = Scaler_X.fit_transform(data_X) #direkt tüm girdileri scale etmek sanırım veri sızdırıyormuş...

    train_X,test_X,train_y,test_y = train_test_split(data_X,data_y,test_size=0.33,shuffle=True,random_state=0)

    train_X = Scaler_X.fit_transform(train_X)
    test_X = Scaler_X.transform(test_X)

    train_X = pd.DataFrame(train_X)
    train_y = pd.DataFrame(train_y)
    test_X = pd.DataFrame(test_X)
    test_y = pd.DataFrame(test_y)



    print(f"\n\n#####\n{n_class}\n{Name}\ntrain_X : {train_X.shape}\ntrain_y : {train_y.shape}\n{train_y.value_counts()}\ntest_X : {test_X.shape}\ntest_y : {test_y.shape}\n{test_y.value_counts()}")

    train_X.to_csv(f"data/processed/{n_class}/{Name}_train_X.csv",index=False)
    train_y.to_csv(f"data/processed/{n_class}/{Name}_train_y.csv",index=False)
    test_X.to_csv(f"data/processed/{n_class}/{Name}_test_X.csv",index=False)
    test_y.to_csv(f"data/processed/{n_class}/{Name}_test_y.csv",index=False)

##########################################################
# 3 Sınıflı 
##########################################################
class_3_all_data = pd.read_excel("data/3_class_all_data.xlsx")
class_3_all_data = class_3_all_data.drop(["Station name","Axle"],axis=1)
class_3_all_data = class_3_all_data[class_3_all_data["Speed (km/h)"] > 15]
class_3_all_data = class_3_all_data.dropna()

class_3_all_data["Alert"] = class_3_all_data["Alert"].replace({
    "OK":0,
    "Da":1,
    "Dw":2
})
ayir_kaydet(class_3_all_data,"3_class","all_data")
##########################################################
# 3 Sınıflı OK sayısı 6000 
##########################################################
class_3_all_data = pd.read_excel("data/3_class_all_data.xlsx")
class_3_all_data = class_3_all_data.drop(["Station name","Axle"],axis=1)
class_3_all_data = class_3_all_data[class_3_all_data["Speed (km/h)"] > 15]
class_3_all_data = class_3_all_data.dropna()

class_3_all_data["Alert"] = class_3_all_data["Alert"].replace({
    "OK":0,
    "Da":1,
    "Dw":2
})



data_ok = class_3_all_data[class_3_all_data["Alert"] == 0].sample(n=6000,random_state=0)
data_da = class_3_all_data[class_3_all_data["Alert"] == 1]
data_dw = class_3_all_data[class_3_all_data["Alert"] == 2]

resized_data = pd.concat([data_ok, data_da,data_dw], ignore_index=True)

ayir_kaydet(resized_data,"3_class","6000")
##########################################################
# 2 Sınıflı 
##########################################################
class_2_all_data = pd.read_excel("data/2_class_all_data.xlsx")
class_2_all_data = class_2_all_data.drop(["Station name","Axle"],axis=1)
class_2_all_data = class_2_all_data[class_2_all_data["Speed (km/h)"] > 15]
class_2_all_data = class_2_all_data.dropna()

class_2_all_data["Alert"] = class_2_all_data["Alert"].replace({
    "OK":0,
    "Fault":1
})
ayir_kaydet(class_2_all_data,"2_class","all_data")

##########################################################
# 2 Sınıflı OK sayısı 6000
##########################################################
class_2_all_data = pd.read_excel("data/2_class_all_data.xlsx")
class_2_all_data = class_2_all_data.drop(["Station name","Axle"],axis=1)
class_2_all_data = class_2_all_data[class_2_all_data["Speed (km/h)"] > 15]
class_2_all_data = class_2_all_data.dropna()

class_2_all_data["Alert"] = class_2_all_data["Alert"].replace({
    "OK":0,
    "Fault":1
})

data_ok = class_3_all_data[class_3_all_data["Alert"] == 0].sample(n=6000,random_state=0)
data_da = class_3_all_data[class_3_all_data["Alert"] == 1]

resized_data = pd.concat([data_ok, data_da], ignore_index=True)

ayir_kaydet(resized_data,"2_class","6000")

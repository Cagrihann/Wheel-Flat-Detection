from sklearn.metrics import roc_curve,auc,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def roc_curve_plot(
        y_true, y_prob,Model_Name,class_names
):
    
    """Basitçe ROC eğrisi çizer\n
    N1: tab20b değiştirilerek renkler değiştirilebiliyor\n 
    N2: aç kapa yapıp modelin kaydetmesini veya görseli göstermesini değiştirebilirsin"""
    class_lenght = len(class_names)
    n_class = y_true.nunique()
    tpr = dict()
    fpr = dict()
    roc_auc = dict()
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.get_cmap('tab20b', class_lenght) ####1

    for i in range(class_lenght):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_prob[:, i])  # Burada her bir sınıf için ayrı ayrı işlem yapılır
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=cmap(i), lw=2, label=f"Class {class_names[i]} (AUC = {roc_auc[i]:.5f})")


    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Multi-Class ROC Curve ({Model_Name})")
    plt.legend(loc="lower right")
    plt.grid(True)
    filename = f"output/roc_curve/roc_curve_{Model_Name}.png" 
    plt.savefig(filename) #kaydetme işlemi
    plt.close() # birden fazla kaydederken belleği boşaltıyor 
    # plt.show() # çizdirmek için 
    return roc_auc



def conf_mat(y_true,y_pred,class_names,Model_Name):
    """Çikti isimlerini nparray istiyor
    https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea"""
    cm = confusion_matrix(y_true,y_pred)
    sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=class_names, yticklabels=class_names) #cbar=False ile ısı barı kapalılabilir
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix {Model_Name}")
    filename = f"output/confusion_matrix/confusion_matrix_{Model_Name}.png" 
    plt.savefig(filename)
    plt.close()
    








import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd

def plot_class_distribution(data):
    # Distribution des classes
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Annotations', data=data)
    plt.title('Distribution des classes')
    plt.xlabel('Classe')
    plt.ylabel('Nombre d\'échantillons')
    plt.show()

def plot_confusion_matrix(y_true, y_pred):
    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake_news', 'Good_news'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Matrice de Confusion')
    plt.show()

def plot_classification_report(y_true, y_pred):
    # Rapport de classification
    report = classification_report(y_true, y_pred, target_names=['Fake_news', 'Good_news'], output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    sns.heatmap(df_report.iloc[:-1, :].T, annot=True, cmap="Blues")
    plt.title('Rapport de Classification')
    plt.show()

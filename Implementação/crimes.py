import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.multiclass import OutputCodeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB



#Define o tamanho de K para o KNN
k = int(input("Qual o tamanho de n? "))


#----------------------------------------------------------------
#Ler os dados de treino e teste
#----------------------------------------------------------------
data_train = pd.read_csv(filepath_or_buffer="dados_treinamento_nc_SP_agrupados_REDUZIDO.txt", sep=',', header=None)
data_test = pd.read_csv(filepath_or_buffer="dados_teste_nc_SP_agrupados_REDUZIDO.txt", sep=',', header=None)


#----------------------------------------------------------------
#Imprime tamanho de cada dataset
#----------------------------------------------------------------
print("\nTreino: ", data_train.shape, "tuplas")
print("Teste: ", data_test.shape, "tuplas\n")


#Separando os dados de treinamento
array_train = data_train.values
data_train = array_train[:,0:9]
label_train = array_train[:,10]
#Separando os dados de teste
array_test = data_test.values
data_test = array_test[:,0:9]
label_test = array_test[:,10]

'''
# Scale the Data to Make the NN easier to converge
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(data_train)  
# Transform the training and testing data
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
'''

models = [KNeighborsClassifier(n_neighbors=k),GaussianNB(),BernoulliNB(),tree.DecisionTreeClassifier(),
          svm.SVC(kernel='linear', C=1), OneVsRestClassifier(svm.SVC(kernel='linear')), MLPClassifier(max_iter=700)]

model_names = ["KNN","Naive Bayes - Gaussiano","Naive Bayes - Bernoulli","Decision Tree", "SVM One VS One","SVM One VS All", "Redes Neurais"]
#----------------------------------------------------------------
# Run Each Model
#----------------------------------------------------------------
for model,name in zip(models,model_names):
    print('\n')
    model.fit(data_train, label_train) 
    prediction = model.predict(data_test)
    acc = accuracy_score(label_test, prediction)
    print("Accuracy Using",name,": " + str(acc))





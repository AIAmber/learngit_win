import pandas as pd
import numpy as np  
 
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
   
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
 
data_path = ('./data_train.txt')
test_path = ('./data_text.txt')
 
def load_datasets(data_path):
    filedata = open(data_path, 'r')

    feature = []
    label = []
    
    for line in filedata:
        items = line.strip().split(' ')
        feature.append((items[0:53]))
        label.append(int(items[54]))

    label = np.ravel(label)

    return feature, label

if __name__ == '__main__':
    load_datasets(data_path)

    x_train = feature[0:300000]
    y_train = label[0:300000]

    x_test = feature[300000:400000]
    y_test = label[300000:400000]
    x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size = 0.0)

    print('Start training knn')
    knn = KNeighborsClassifier().fit(x_train, y_train)
    print('Training done')
    answer_knn = knn.predict(x_test)
    print('Prediction done')
     
    print('Start training DT')
    dt = DecisionTreeClassifier().fit(x_train, y_train)
    print('Training done')
    answer_dt = dt.predict(x_test)
    print('Prediction done')
     
    print('Start training Bayes')
    gnb = GaussianNB().fit(x_train, y_train)
    print('Training done')
    answer_gnb = gnb.predict(x_test)
    print('Prediction done')
     
    print('\n\nThe classification report for knn:')
    print(classification_report(y_test, answer_knn))
    print('\n\nThe classification report for DT:')
    print(classification_report(y_test, answer_dt))
    print('\n\nThe classification report for Bayes:')
    print(classification_report(y_test, answer_gnb))
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
feature_path = ('./data_train_feature.txt')
label_path = ('./data_train_label.txt')

feature_test_path = ('test_train_feature.txt')
label_test_path = ('test_train_label.txt')

def reset_data(data_path):
    filedata = open(data_path, 'r')

    feature_init = []
    label_init = []
    
    for line in filedata:
        items = line.strip().split(' ')
        feature_init.append((items[0:53]))
        label_init.append(int(items[54]))

    

    np.savetxt(feature_path, feature_init[0:300000])
    np.savetxt(label_path, label_init[0:300000])
    np.savetxt(feature_test_path, feature_init[300000:400000])
    np.savetxt(label_test_path, label_init[300000:400000])
    '''
    filedata_feature = open(feature_path, 'w')
    filedata_feature.write(feature_init[0:300000])

    filedata_label = open(label_path, 'w')
    filedata_label.write(label_init[0:300000])

    filedata_feature_test = open(feature_test_path, 'w')
    filedata_feature_test.write(feature_init[300000:400000])

    filedata_label_test = open(label_test_path, 'w')
    filedata_label_test.write(label_init[300000:400000])
    '''

def load_datasets(feature_path, label_path):
    reset_data(data_path)

    feature = np.ndarray(shape=(0,41))
    label = np.ndarray(shape=(0,1))

    for file in feature_path:
        df = pd.read_table(file, delimiter=',', na_values='?', header=None)
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imp.fit(df)
        df = imp.transform(df)
        feature = np.concatenate((feature, df))
     
    for file in label_path:
        df = pd.read_table(file, header=None)
        label = np.concatenate((label, df))
         
    label = np.ravel(label)
    return feature, label

if __name__ == '__main__':
    #load_datasets(data_path)

    x_train,y_train = load_datasets(feature_path,label_path)
    x_test,y_test = load_datasets(feature_test_path,label_test_path)
    x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size = 0.25)

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
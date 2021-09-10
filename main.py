# -*- coding: utf-8 -*-

import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering  # cluster hierarquico
from sklearn.ensemble import BaggingClassifier # comites homogeneos
from sklearn.ensemble import AdaBoostClassifier # comites homogeneos
from sklearn.ensemble import VotingClassifier # comites heterogeneos
import os
from PIL import Image
from io import BytesIO ## for Python 3

#pd.set_option('display.max_rows', None) # mostrar prints completos

# fonte das funções process_image e process_image_file: https://gist.github.com/gcardone/c49e3f66dc83be33666d
def process_image(image, blocks=4):
    if not image.mode == 'RGB':
        return None
    feature = [0] * blocks * blocks * blocks
    #feature = [0] * (blocks ** (blocks - 1))
    pixel_count = 0.0
    for pixel in image.getdata():
        ridx = int(pixel[0]/(256/blocks))
        gidx = int(pixel[1]/(256/blocks))
        bidx = int(pixel[2]/(256/blocks))
        idx = ridx + gidx * blocks + bidx * blocks * blocks
        feature[idx] += 1
        pixel_count += 1
    return [x/pixel_count for x in feature]

def process_image_file(image_path):
    image_fp = BytesIO(open(image_path, 'rb').read())
    try:
        image = Image.open(image_fp)
        return process_image(image)
    except IOError:
        return None

def dataset(data, path):
    for pasta in os.listdir(path)[:]:
        instancias = 0
        cluster = 0
        for image_file in os.listdir(path + pasta)[:]:
            process_image = process_image_file(path + pasta + '/' + image_file)
            if process_image != None:
                #data = data.append({'process_image': process_image, 'class': pasta, 'image_file': pasta + '/' + image_file}, ignore_index=True)
                process_image.append(pasta + '/' + image_file)
                process_image.append(pasta)
                #process_image.append(cluster)
                data = data.append(pd.DataFrame([process_image]), ignore_index=True)
                instancias += 1
        cluster += 1
        print (pasta + '\t' + str(instancias) + '\tinstancias' + '\tOK!')
    data = data.rename({64:'image_file', 65:'class'}, axis='columns')
    return data


#data = pd.DataFrame(columns=['process_image', 'class', 'image_file'])
data = pd.DataFrame()
data = dataset(data, './data/')

#X_train, X_test, y_train, y_test = train_test_split(data['process_image'].values.tolist(), data['class'].values.tolist(), test_size=0.33)
X_train, X_test, y_train, y_test = train_test_split(data.drop(['image_file', 'class'], axis=1).values.tolist(), data['class'].values.tolist(), test_size=0.33)
#X_train, X_test, y_train, y_test = train_test_split(data.drop(['image_file', 'class'], axis=1).values.tolist(), data['class'].values.tolist(), test_size=0.10, shuffle=False)

#print (data.iloc[:,:64])
#print (data.drop(['image_file', 'class'], axis=1).values.tolist())
#print (data)

test_print = pd.DataFrame(X_test)
test_print['class'] = pd.DataFrame(y_test)
print (test_print.head(10))

'''
          0         1         2        3         4  ...   60   61        62        63              class
0  0.789652  0.080167  0.000932  0.00002  0.001130  ...  0.0  0.0  0.000000  0.004639          aryastark
1  0.755746  0.095812  0.000043  0.00000  0.000129  ...  0.0  0.0  0.000000  0.002173           nedstark
2  0.311955  0.021491  0.000000  0.00000  0.018398  ...  0.0  0.0  0.059417  0.176705          aryastark
3  0.111844  0.030934  0.000238  0.00000  0.000020  ...  0.0  0.0  0.000000  0.004181          branstark
4  0.685287  0.007301  0.000000  0.00000  0.000119  ...  0.0  0.0  0.000000  0.002387    tyrionlannister
5  0.549407  0.062705  0.000000  0.00000  0.000874  ...  0.0  0.0  0.000040  0.005820   joffreybaratheon
6  0.026360  0.037416  0.000060  0.00000  0.000139  ...  0.0  0.0  0.000238  0.209607  daenerystargaryen
7  0.356227  0.015044  0.000000  0.00000  0.000040  ...  0.0  0.0  0.000000  0.012943  daenerystargaryen
8  0.692355  0.095651  0.000000  0.00000  0.000122  ...  0.0  0.0  0.000000  0.010886    cerseilannister
9  0.718853  0.121018  0.000630  0.00000  0.000000  ...  0.0  0.0  0.000000  0.000062          branstark

[10 rows x 65 columns]
'''

print ('instancias para treinamento:\t', len(X_train))
print ('instancias para teste:\t', len(X_test))

'''
instancias para treinamento:	463
instancias para teste:	        229
'''

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print ('DecisionTreeClassifier:\t', clf.score(X_test, y_test))

clf = GaussianNB()
clf.fit(X_train, y_train)
print ('GaussianNB:\t', clf.score(X_test, y_test))

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
print ('KNeighborsClassifier:\t', clf.score(X_test, y_test))

clf = SVC()
clf.fit(X_train, y_train)
print ('SVC:\t', clf.score(X_test, y_test))

clf = MLPClassifier()
clf.fit(X_train, y_train)
print ('MLPClassifier:\t', clf.score(X_test, y_test))

'''
DecisionTreeClassifier:	 0.4148471615720524
GaussianNB:	             0.28820960698689957
KNeighborsClassifier:	 0.24890829694323144
SVC:	                 0.13973799126637554
MLPClassifier:	         0.27074235807860264
'''


''' ####### TAREFA CLUSTERING (não concluida) #######
clustering = KMeans(n_clusters=3)
clustering.fit(X_train, y_train)
print ('KMeans      OK!')
#print (clustering.labels_)
#print ('y_train: ', y_train)
#print (clustering.predict(X_test))
#print ('y_test: ', y_test)
#print (kmeans.cluster_centers_)
data_print = pd.DataFrame(clustering.predict(X_test))
data_print['class'] = pd.DataFrame(y_test)
print (data_print)
#print ('KMeans: ', kmeans.score(X_test, y_test))

clustering = AgglomerativeClustering(n_clusters=3)
clustering.fit(X_train, y_train)
print ('AgglomerativeClustering      OK!')
#print (clustering.labels_)
#print ('y_train: ', y_train)
#print (clustering.predict(X_test))
#print ('y_test: ', y_test)
#print (kmeans.cluster_centers_)
data_print = pd.DataFrame(clustering.fit_predict(X_test))
data_print['class'] = pd.DataFrame(y_test)
print (data_print)
#print ('KMeans: ', kmeans.score(X_test, y_test))
'''


clf = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=6)
clf.fit(X_train, y_train)
print ('comite homogeneo bagging KNN tam=6:\t', clf.score(X_test, y_test))

clf = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=12)
clf.fit(X_train, y_train)
print ('comite homogeneo bagging KNN tam=12:\t', clf.score(X_test, y_test))

clf = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=24)
clf.fit(X_train, y_train)
print ('comite homogeneo bagging KNN tam=24:\t', clf.score(X_test, y_test))


clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=6)
clf.fit(X_train, y_train)
print ('comite homogeneo bagging DecisionTreeClassifier tam=6:\t', clf.score(X_test, y_test))

clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=12)
clf.fit(X_train, y_train)
print ('comite homogeneo bagging DecisionTreeClassifier tam=12:\t', clf.score(X_test, y_test))

clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=24)
clf.fit(X_train, y_train)
print ('comite homogeneo bagging DecisionTreeClassifier tam=24:\t', clf.score(X_test, y_test))


clf = BaggingClassifier(base_estimator=GaussianNB(), n_estimators=6)
clf.fit(X_train, y_train)
print ('comite homogeneo bagging GaussianNB tam=6:\t', clf.score(X_test, y_test))

clf = BaggingClassifier(base_estimator=GaussianNB(), n_estimators=12)
clf.fit(X_train, y_train)
print ('comite homogeneo bagging GaussianNB tam=12:\t', clf.score(X_test, y_test))

clf = BaggingClassifier(base_estimator=GaussianNB(), n_estimators=24)
clf.fit(X_train, y_train)
print ('comite homogeneo bagging GaussianNB tam=24:\t', clf.score(X_test, y_test))


clf = BaggingClassifier(base_estimator=MLPClassifier(), n_estimators=6)
clf.fit(X_train, y_train)
print ('comite homogeneo bagging MLPClassifier tam=6:\t', clf.score(X_test, y_test))

clf = BaggingClassifier(base_estimator=MLPClassifier(), n_estimators=12)
clf.fit(X_train, y_train)
print ('comite homogeneo bagging MLPClassifier tam=12:\t', clf.score(X_test, y_test))

clf = BaggingClassifier(base_estimator=MLPClassifier(), n_estimators=24)
clf.fit(X_train, y_train)
print ('comite homogeneo bagging MLPClassifier tam=24:\t', clf.score(X_test, y_test))


clf = BaggingClassifier(base_estimator=SVC(), n_estimators=6)
clf.fit(X_train, y_train)
print ('comite homogeneo bagging SVM tam=6:\t', clf.score(X_test, y_test))

clf = BaggingClassifier(base_estimator=SVC(), n_estimators=12)
clf.fit(X_train, y_train)
print ('comite homogeneo bagging SVM tam=12:\t', clf.score(X_test, y_test))

clf = BaggingClassifier(base_estimator=SVC(), n_estimators=24)
clf.fit(X_train, y_train)
print ('comite homogeneo bagging SVM tam=24:\t', clf.score(X_test, y_test))

'''
comite homogeneo bagging KNN tam=6:	                        0.28820960698689957
comite homogeneo bagging KNN tam=12:	                    0.2794759825327511
comite homogeneo bagging KNN tam=24:	                    0.24890829694323144
comite homogeneo bagging DecisionTreeClassifier tam=6:	    0.45414847161572053
comite homogeneo bagging DecisionTreeClassifier tam=12:	    0.5065502183406113
comite homogeneo bagging DecisionTreeClassifier tam=24:	    0.5545851528384279
comite homogeneo bagging GaussianNB tam=6:	                0.314410480349345
comite homogeneo bagging GaussianNB tam=12:	                0.3056768558951965
comite homogeneo bagging GaussianNB tam=24:	                0.30131004366812225
comite homogeneo bagging MLPClassifier tam=6:	            0.27074235807860264
comite homogeneo bagging MLPClassifier tam=12:	            0.2794759825327511
comite homogeneo bagging MLPClassifier tam=24:	            0.27074235807860264
comite homogeneo bagging SVM tam=6:	                        0.13973799126637554
comite homogeneo bagging SVM tam=12:	                    0.13973799126637554
comite homogeneo bagging SVM tam=24:	                    0.13973799126637554
'''

# KNN24 + TREE24
clf = VotingClassifier(estimators=[('KNN24', BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=24)), ('TREE24', BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=24))])
clf.fit(X_train, y_train)
print ('comite heterogeneo KNN24 + TREE24:\t', clf.score(X_test, y_test))

# KNN24 + TREE12
clf = VotingClassifier(estimators=[('KNN24', BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=24)), ('TREE12', BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=12))])
clf.fit(X_train, y_train)
print ('comite heterogeneo KNN24 + TREE12:\t', clf.score(X_test, y_test))

#TREE12 + TREE24
clf = VotingClassifier(estimators=[('TREE12', BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=12)), ('TREE24', BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=24))])
clf.fit(X_train, y_train)
print ('comite heterogeneo TREE12 + TREE24:\t', clf.score(X_test, y_test))

#KNN24 + TREE12 + TREE24
clf = VotingClassifier(estimators=[('KNN24', BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=24)), ('TREE12', BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=12)), ('TREE24', BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=24))])
clf.fit(X_train, y_train)
print ('comite heterogeneo KNN24 + TREE12 + TREE24:\t', clf.score(X_test, y_test))

'''
comite heterogeneo KNN24 + TREE24:	            0.4279475982532751
comite heterogeneo KNN24 + TREE12:	            0.4192139737991266
comite heterogeneo TREE12 + TREE24:	            0.5633187772925764
comite heterogeneo KNN24 + TREE12 + TREE24:	    0.537117903930131
'''


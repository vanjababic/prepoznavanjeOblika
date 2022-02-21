# -*- coding: utf-8 -*-
"""
@author: babic
"""
#%% ucitavanje biblioteka

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

#%% ucitavanja podataka

train = pd.read_csv('cookies_train.csv')
test = pd.read_csv('cookies_test.csv')

print(train.isnull().sum().sum())

X = train.iloc[:,:-1]
y = train.iloc[:,-1]
y_test = test.iloc[:,-1]
print(X.shape)
labels_y = y.unique()
print(labels_y)

print(y.groupby(by=y).count())
print(y_test.groupby(by=y_test).count())

#%%

df_class = train.set_index('class').groupby('class').sum()

df_cookies = df_class.loc[['Cookies']]
df_cookies = df_cookies.T
df_pasteries = df_class.loc[['Pastries']]
df_pasteries = df_pasteries.T
df_pizzas = df_class.loc[['Pizzas']]
df_pizzas = df_pizzas.T

df_cookies.plot(kind='bar', figsize=(25,15))
df_pasteries.plot(kind='bar', figsize=(25,15))
df_pizzas.plot(kind='bar', figsize=(25,15))
    
#%% unakrsna validacija, biranje najboljih parametara KNN

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

acc = []
acc_matching = []
acc_jaccard = []
acc_dice = []
acc_kulsinski = []

for k in range(1, 11):
    for m in ['matching', 'jaccard', 'dice', 'kulsinski']:
        indexes = kf.split(X, y)
        acc_tmp = []
        fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
        for train_index, test_index in indexes:
            classifier = KNeighborsClassifier(n_neighbors=k, metric=m)
            classifier.fit(X.iloc[train_index,:], y.iloc[train_index])
            y_pred = classifier.predict(X.iloc[test_index,:])
            acc_tmp.append(accuracy_score(y.iloc[test_index], y_pred))
            fin_conf_mat += confusion_matrix(y.iloc[test_index], y_pred, labels=labels_y)
        print('za parametre k=', k, ' i m =', m, ' tacnost je: ', np.mean(acc_tmp), ' a mat. konf. je:')
        print(fin_conf_mat)
        acc.append(np.mean(acc_tmp))
        if(m == 'matching'):
            acc_matching.append(np.mean(acc_tmp))
        if(m == 'jaccard'):
            acc_jaccard.append(np.mean(acc_tmp))
        if(m == 'dice'):
            acc_dice.append(np.mean(acc_tmp))
        if(m == 'kulsinski'):
            acc_kulsinski.append(np.mean(acc_tmp))
            
plt.figure(figsize=(12, 6))
plt.plot(range(1, 11), acc_matching, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate for matching')
plt.xlabel('K Value')
plt.ylabel('Acc')

plt.figure(figsize=(12, 6))
plt.plot(range(1, 11), acc_jaccard, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate for jaccard')
plt.xlabel('K Value')
plt.ylabel('Acc')

plt.figure(figsize=(12, 6))
plt.plot(range(1, 11), acc_dice, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate for dice')
plt.xlabel('K Value')
plt.ylabel('Acc')

plt.figure(figsize=(12, 6))
plt.plot(range(1, 11), acc_kulsinski, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate for kulsinski')
plt.xlabel('K Value')
plt.ylabel('Acc')

print('najbolja tacnost je u iteraciji broj: ', np.argmax(acc))
print('najlosija tacnost je u iteraciji broj: ', np.argmin(acc))



#%% testiranje KNN

classifier = KNeighborsClassifier(n_neighbors=1, metric='kulsinski')
classifier.fit(X, y)
y_pred = classifier.predict(test.iloc[:,:-1])
conf_mat = confusion_matrix(test.iloc[:,-1], y_pred, labels=labels_y)
print(conf_mat)
print('procenat pogodjenih uzoraka: ', accuracy_score(test.iloc[:,-1], y_pred))
print('preciznost mikro: ', precision_score(test.iloc[:,-1], y_pred, average='micro'))
print('preciznost makro: ', precision_score(test.iloc[:,-1], y_pred, average='macro'))
print('osetljivost mikro: ', recall_score(test.iloc[:,-1], y_pred, average='micro'))
print('osetljivost makro: ', recall_score(test.iloc[:,-1], y_pred, average='macro'))
print('f mera mikro: ', f1_score(test.iloc[:,-1], y_pred, average='micro'))
print('f mera makro: ', f1_score(test.iloc[:,-1], y_pred, average='macro'))
print(labels_y)

#%% unakrsna validacija, biranje najboljih parametara SVM

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
acc = []
for c in [1, 10, 20, 30, 40, 50]:
    for F in ['linear', 'rbf', 'poly']:
        for mc in ['ovo','ovr']:
            indexes = kf.split(X, y)
            acc_tmp = []
            fin_conf_mat = np.zeros((len(np.unique(y)),len(np.unique(y))))
            for train_index, test_index in indexes:
                classifier = SVC(C=c, kernel=F, decision_function_shape=mc)
                classifier.fit(X.iloc[train_index,:], y.iloc[train_index])
                y_pred = classifier.predict(X.iloc[test_index,:])
                acc_tmp.append(accuracy_score(y.iloc[test_index], y_pred))
                fin_conf_mat += confusion_matrix(y.iloc[test_index], y_pred, labels=labels_y)
            print('za parametre C=', c, ', kernel=', F, ' i pristup ', mc, ' tacnost je: ', np.mean(acc_tmp),
                  ' a mat. konf. je:')
            print(fin_conf_mat)
            acc.append(np.mean(acc_tmp))
print('najbolja tacnost je u iteraciji broj: ', np.argmax(acc))

#%% testiranje SVM

classifier = SVC(C=30, kernel='rbf', decision_function_shape='ovr')
classifier.fit(X, y)
y_pred = classifier.predict(test.iloc[:,:-1])
conf_mat = confusion_matrix(test.iloc[:,-1], y_pred, labels=labels_y)
print(conf_mat)
print('procenat pogodjenih uzoraka: ', accuracy_score(test.iloc[:,-1], y_pred))
print('preciznost mikro: ', precision_score(test.iloc[:,-1], y_pred, average='micro'))
print('preciznost makro: ', precision_score(test.iloc[:,-1], y_pred, average='macro'))
print('osetljivost mikro: ', recall_score(test.iloc[:,-1], y_pred, average='micro'))
print('osetljivost makro: ', recall_score(test.iloc[:,-1], y_pred, average='macro'))
print('f mera mikro: ', f1_score(test.iloc[:,-1], y_pred, average='micro'))
print('f mera makro: ', f1_score(test.iloc[:,-1], y_pred, average='macro'))


#%%




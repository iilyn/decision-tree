# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 10:49:10 2022

@author: neural.net_
"""
from sklearn.datasets import load_wine
from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

'Initialize Tree Instance'
#  DecisionTreeClassifier is capable of both binary (labels = [-1, 1]) 
#  and multiclass (labels = [0, …, K-1]) classification.
tree_ = tree.DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)
'Load Data'
data =  load_wine(as_frame=True)
X, y  = load_wine(return_X_y=True)
print('--------Dataset Info--------' )
print('Features: ', X.shape[1])
print('Number samples: ', X.shape[0])

'Split Data into Training and Test Dataset'
test_ratio = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
print('\n--------Split Info--------' )
print('Number Training samples: ', X_train.shape[0])
print('Number Test samples: ', X_test.shape[0])

'Train model'
tree_ = tree_.fit(X_train, y_train)

'Test model'
y_pred = tree_.predict(X_test)
# y_prob = tree_.predict_proba(X_test)   # predicts the probabilities of each class
correct = sum(y_pred == y_test)
cm = confusion_matrix(y_test, y_pred)
print('\n--------Classification Result--------' )
print('Correctly classified: ' + str(correct) + '/' + str(X_test.shape[0]))
print('Percentage: ' + str(correct/X_test.shape[0]))
print('Confusion matrix:')
print(cm)
tree.plot_tree(tree_)



        

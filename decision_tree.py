# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 15:54:10 2022

@author: neural.net_
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
'Load Data'
data =  load_wine(as_frame=True)
X, y  = load_wine(return_X_y=True)

'Split Data into Training and Test Dataset'
test_ratio = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2, impurity_measure='IG'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.impurity_measure = impurity_measure
        if impurity_measure =='gini':
            self.impurity = self._gini
        else:
            self.impurity = self._information_gain
    
    def _check_criteria(self, depth):
        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        return False
    
    def _generate_split(self, X, th):
        idx_l = np.argwhere(X <= th).flatten()
        idx_r = np.argwhere(X > th).flatten()
        return idx_l, idx_r
        
    def _gini(self, X, y, th):
        idx_l, idx_r = self._generate_split(X, th)
        n, n_l, n_r = len(y), len(idx_l), len(idx_r)
        if n_l == 0 or n_r == 0: 
            return 1
        else:
            G_r= 1 -  np.sum(np.square(np.bincount(y[idx_r])/n_r ))
            G_l= 1 -  np.sum(np.square(np.bincount(y[idx_l])/n_l ))
            G = n_r/n*G_r + n_l/n*G_l
        return G
    
    def _information_gain(self, X_f, y, th):
        def _entropy(y):
            distribution = np.bincount(y.flatten()) / len(y)
            entropy = -np.sum([p * np.log2(p) for p in distribution if p > 0])
            return entropy
        idx_l, idx_r = self._generate_split(X_f, th)
        n, n_l, n_r = len(y), len(idx_l), len(idx_r)
        if n_l == 0 or n_r == 0: 
            return 0
        else:
            H_marg = _entropy(y)
            H_cond = (n_l/n)*_entropy(y[idx_l]) + (n_r/n)*_entropy(y[idx_r])
            IG = H_marg - H_cond
        return IG
        
    def _optimal_split(self, X, y, features):
        if self.impurity_measure == 'gini':
            split = {'score': 1, 'feat': None, 'thresh': None}
        else:
            split = {'score': 0, 'feat': None, 'thresh': None}
            
        for f in features:
            X_f = X[:, f]
            thresholds  = np.unique(X_f)
            for th in thresholds:
                I = self.impurity(X_f, y, th)
                
                if self.impurity_measure == 'gini':
                    if I  < split['score']:
                        split['score'] = I 
                        split['feat'] = f
                        split['thresh'] = th
                else:
                    if I  > split['score']:
                        split['score'] = I 
                        split['feat'] = f
                        split['thresh'] = th  
        return split['feat'], split['thresh']
    
    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))
     # stopping criteria
        if self._check_criteria(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)
    # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._optimal_split(X, y, rnd_feats)
    # grow children recursively
        left_idx, right_idx = self._generate_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)
        
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        
    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
    

clf = DecisionTree(max_depth=10, impurity_measure='IG')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
correct = sum(y_pred == y_test)
cm = confusion_matrix(y_test, y_pred)
print('\n--------Classification Result--------' )
print('Correctly classified: ' + str(correct) + '/' + str(X_test.shape[0]))
print('Percentage: ' + str(correct/X_test.shape[0]))
print('Confusion matrix:')
print(cm)

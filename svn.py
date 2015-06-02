# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn
from sklearn import decomposition
from sklearn.neighbors import NearestNeighbors
from sklearn import neighbors, datasets
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

root = "/Users/mdmiah/Desktop/Kaggle/Digit Recognizer/";
path_train = root+"Data/train.csv";
path_test = root+"Data/test.csv";
path_out_knn = root+"results_knn.csv";
path_out_svn = root+"results_svn.csv";
path_out_randomForest = root+"results_randomForest.csv";
path_out_nn = root+"results_nn.csv";

print "The path is "+path_train;

# Load data
train = pd.read_csv(path_train);
test = pd.read_csv(path_test);
print "Data has been loaded from files";

# Parameters
w = 28; # dimensions of image

# Normalize
#train.values[:,1:] = train.values[:,1:].astype('float32')/255.0;
#test.values[:,1:]  =  test.values[:,1:].astype('float32')/255.0;

# PCA to use
print "Computing pricipal components...";
p = decomposition.PCA(n_components=50)
p.fit(train.values[:,1:])
X = p.fit_transform(train.values[:,1:])
y = train.values[:,0]
Xtest = p.transform(test.values)

# Without PCA
#X = train.values[:,1:]

# Perform grid search to find best parameters...
param_grid = [
    {'C': [1e0, 1e1, 1e2], 'gamma': [1e-5, 1e-6, 1e-7], 'kernel': ['rbf']},
    ]
print "Performing grid search..."
clf = GridSearchCV(svm.SVC(C=1), param_grid, n_jobs=-1);
clf.fit(X, y); # Use a subsample of the data to find the best parameters
print clf.best_estimator_;
C = clf.best_estimator_.C
gamma = clf.best_estimator_.gamma
kernel = clf.best_estimator_.kernel

# SVN
print "SVN...";
clf = svm.SVC(kernel=kernel, gamma=gamma, C=C);
clf.fit(X, y)
print str(clf.n_support_.sum()) + " support vectors"

# Cross-tabulate results
print "Cross-tabulating..."
Z = clf.predict(X)
print pd.crosstab(y, Z, rownames=['actual'], colnames=['preds'])

print "Cross-validation..."
score = cross_val_score(clf, X, y, cv=5)
score_mean = score.mean()
print score_mean

# Run classifier on the test data
print "Classifying test data..."
Ztest = clf.predict(Xtest)

# Writing result on test data to file...
print "Saving test data classifications into "+path_out_svn+"..."
Zout = pd.DataFrame(Ztest);
Zout.index += 1;
Zout.to_csv(path_out_svn, sep=',', encoding='utf-8', index_label=["ImageId"],\
    header=["Label"]);





# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
#import sklearn
from sklearn import decomposition
#from sklearn.neighbors import NearestNeighbors
#from sklearn import neighbors, datasets
#from sklearn import svm
#from sklearn.grid_search import GridSearchCV
#from sklearn.metrics import classification_report
#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.cross_validation import cross_val_score
#from pybrain.structure import FeedForwardNetwork
#from pybrain.structure import LinearLayer, SigmoidLayer
#from pybrain.structure import FullConnection
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
#from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
#from scipy import diag, arange, meshgrid, where
#from numpy.random import multivariate_normal
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

root = "/Users/mdmiah/Desktop/Kaggle/Digit Recognizer/";
path_train = root+"Data/train.csv";
path_test = root+"Data/test.csv";
path_out_knn = root+"results_knn.csv";
path_out_svn = root+"results_svn.csv";
path_out_randomForest = root+"results_randomForest.csv";
path_out_nn = root+"results_nn.csv";

print "The path is "+path_train

# Load data
train = pd.read_csv(path_train)
test = pd.read_csv(path_test)
print "Data has been loaded from files"

# Parameters
#w = 28; # dimensions of image

# Normalize
#train.values[:,1:] = train.values[:,1:].astype('float32')/255.0;
#test.values[:,1:]  =  test.values[:,1:].astype('float32')/255.0;

# PCA to use
print "Computing pricipal components..."
p = decomposition.PCA(n_components=20)
p.fit(train.values[:,1:])
X = p.fit_transform(train.values[:,1:])
y = train.values[:,0]
Xtest = p.transform(test.values)

# Without PCA
#X = train.values[:,1:]

## Perform grid search to find best parameters...
#param_grid = [
#    {'C': [1e0, 1e1, 1e2], 'gamma': [1e-5, 1e-6, 1e-7], 'kernel': ['rbf']},
#    ]
#print "Performing grid search..."
#clf = GridSearchCV(_________, param_grid, n_jobs=-1);
#clf.fit(X, y); # Use a subsample of the data to find the best parameters
#print clf.best_estimator_;
#_______ = clf.best_estimator_._______

# Load training dataset in PyBrain's format
alldata = ClassificationDataSet(X.shape[1], 1, nb_classes=10)
for n in xrange(X.shape[0]):
    alldata.addSample(X[n], y[n])

# ------------------------- MLP Backpropagation start -------------------------
tstdata_temp, trndata_temp = alldata.splitWithProportion(0.25)

tstdata = ClassificationDataSet(X.shape[1], 1, nb_classes=10)
for n in xrange(tstdata_temp.getLength()):
    tstdata.addSample( tstdata_temp.getSample(n)[0], tstdata_temp.getSample(n)[1] )

trndata = ClassificationDataSet(X.shape[1], 1, nb_classes=10)
for n in xrange(trndata_temp.getLength()):
    trndata.addSample( trndata_temp.getSample(n)[0], trndata_temp.getSample(n)[1] )

trndata._convertToOneOfMany( )
tstdata._convertToOneOfMany( )

fnn = buildNetwork( trndata.indim, 100, 100, 100, trndata.outdim, outclass=SoftmaxLayer )

trainer = BackpropTrainer( fnn, dataset=trndata, learningrate=0.2, lrdecay=0.99999, \
            momentum=0.01, verbose=True, weightdecay=0.00001)
trainer.trainEpochs( 6 )

trnresult = percentError( trainer.testOnClassData(),
                          trndata['class'] )
tstresult = percentError( trainer.testOnClassData(
       dataset=tstdata ), tstdata['class'] )

print "  train error: %5.2f%%" % trnresult, \
      "  test error: %5.2f%%" % tstresult

Z =  np.asarray( trainer.testOnClassData(dataset=alldata) )
# -------------------------- MLP Backpropagation end --------------------------

## NN
#print "NN...";
#clf = ___________________
#clf.fit(X, y)
#print str(clf.n_support_.sum()) + " support vectors"
#Z = clf.predict(X)

# Cross-tabulate results
print "Cross-tabulating..."
print pd.crosstab(y, Z, rownames=['actual'], colnames=['preds'])

#print "Cross-validation..."
#score = cross_val_score(clf, X, y, cv=5)
#score_mean = score.mean()
#print score_mean

# Load test data in PyBrain's format
testdata = ClassificationDataSet(Xtest.shape[1], 1, nb_classes=10)
for i in xrange(Xtest.shape[0]):
    testdata.addSample(Xtest[i], [0])
testdata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy

## Run classifier on the test data
#Ztest = clf.predict(Xtest)
print "Classifying test data..."
Ztest = fnn.activateOnDataset(testdata)
Ztest = Ztest.argmax(axis=1)  # the highest output activation gives the class

## Writing result on test data to file...
print "Saving test data classifications into "+path_out_nn+"..."
Zout = pd.DataFrame(Ztest);
Zout.index += 1;
Zout.to_csv(path_out_nn, sep=',', encoding='utf-8', index_label=["ImageId"],\
    header=["Label"]);






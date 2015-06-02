# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:22:37 2015

@author: Muhammed Onu Miah

Experimenting on predicting on MNIST data
Using OpenCV to extract HOG (Histogram of Oriented Gradients)
Using Random Forests from SKLearn to predict

"""

# Import packages
import argparse
import numpy as np
import pandas as pd
# OpenCV
import cv2
from cv2 import HOGDescriptor
# Machine Learning: Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# Machine Learning: Predictor
from sklearn.ensemble import RandomForestClassifier
# Machine Learning: Evaluation
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# Visualisation
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
# Other
import random

# Paths
root = "/Users/mdmiah/Desktop/Kaggle/Digit Recognizer/"
path_train = root+"Data/train.csv"
path_test = root+"Data/test.csv"
path_out_1 = root+"results_opencv1.csv"
path_out = path_out_1

print "The path is "+path_train

# Load data using Pandas (returns DataFrame)
train = pd.read_csv(path_train, dtype="uint8")
test = pd.read_csv(path_test, dtype="uint8")
yTrain = train.values[:,0]
dataTrain = train.values[:, 1:]
dataTest = test.values[:, :]
print "Data has been loaded from files using Pandas"

# Load data using NumPy (returns uint8 2d-array - slower)
#train = np.genfromtxt(path_train, delimiter=",", dtype="uint8", skip_header=1)
#test  = np.genfromtxt(path_test,  delimiter=",", dtype="uint8", skip_header=1)
#yTrain = train[:, 0]
#dataTrain = train[:, 1:]
#dataTest  = test[:, :]
#print "Data has been loaded from files using NumPy"

# Parameters
trainSize = dataTrain.shape[0]
testSize = dataTest.shape[0]
w = 28; # dimensions of the image

# ---------------------------------- OpenCV ----------------------------------

# Clear anything from previous runs of the script
cv2.destroyAllWindows()

# Load an image and show it
#print "Displaying a few images..."
#for i in random.sample(range(0,784), 10):
#    image = dataTrain[i].reshape(w,w)
#    cv2.imshow("image", image)
#    cv2.waitKey(0) & 0xFF
#    cv2.destroyAllWindows()

# show a histogram
#hist = cv2.calcHist([image], [0], None, [256], [0, 256])
#plt.figure()
#plt.title("Grayscale Histogram")
#plt.xlabel("Bins")
#plt.ylabel("# of Pixels")
#plt.plot(hist)
#plt.xlim([0, 256])

# -------------------------------- OpenCV: HoG --------------------------------

# Compute the histogram of oriented gradients
# http://docs.opencv.org/modules/gpu/doc/object_detection.html
hog = HOGDescriptor(_winSize=(w,w), _blockSize=(4,4), _blockStride=(2,2), _cellSize=(2,2), _nbins=9)

hogTrain = np.zeros((trainSize, hog.getDescriptorSize()), dtype="float32")
for i in xrange(0,trainSize):
    image = dataTrain[i].reshape(w,w)
    h = hog.compute(image)
    hogTrain[i,:] = h[:,0]

hogTest = np.zeros((testSize, hog.getDescriptorSize()), dtype="float32")
for i in xrange(0,testSize):
    image = dataTest[i].reshape(w,w)
    h = hog.compute(image)
    hogTest[i,:] = h[:,0]

print "Extracted HoG features (", h.size, " per image)"

# -------------------- Machine Learning: Feature Selection --------------------

selector = SelectKBest(chi2, k=500)
XTrain = selector.fit_transform(hogTrain, yTrain)
XTest = selector.transform(hogTest)

print "Performed feature selection"

# ------------------------ Machine Learning: Predictor ------------------------

# Random Forest
print "Random Forest..."
clf = RandomForestClassifier(n_estimators=200, bootstrap=False, n_jobs=-1)
clf.fit(XTrain, yTrain)

# ------------------------ Machine Learning: Evaluation -----------------------

# Cross-tabulate results
print "Cross-tabulating..."
Z = clf.predict(XTrain)
print pd.crosstab(yTrain, Z, rownames=['actual'], colnames=['preds'])

print "Cross-validation..."
score = cross_val_score(clf, XTrain, yTrain, cv=5)
score_mean = score.mean()
print score_mean

# ---------------------------------- Output ----------------------------------

# Run classifier on the test data
print "Classifying test data..."
Ztest = clf.predict(XTest)

# Writing result on test data to file...
print "Saving test data classifications into "+path_out+"..."
Zout = pd.DataFrame(Ztest)
Zout.index += 1
Zout.to_csv(path_out, sep=',', encoding='utf-8', index_label=["ImageId"],\
    header=["Label"])

# Clean up and end
cv2.destroyAllWindows()
print "Done"


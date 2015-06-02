# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

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
path_out_randomForest = root+"results_randomForest.csv";

print "The path is "+path_train;

# Load data
mnist = pd.read_csv(path_train);
test = pd.read_csv(path_test);
print "Data has been loaded from files";

# Parameters
w = 28; # dimensions of image

# Normalize
#mnist.values[:,1:] = mnist.values[:,1:].astype('float32')/255.0;
#test.values[:,1:] = test.values[:,1:].astype('float32')/255.0;

# Draw a number
#print "single image..."
#n1 = 3;
#pix = mnist.values[n1][1:];
#pix = pix.astype('float32')/255;
#pix = pix.reshape([28,28]);
#plt.imshow(pix); # Plot image
#plt.colorbar(); # Plot color bar
#plt.show(); # Output image


# Draw multiple numbers next to each other
#print "multiple images..."
#n_temp = 2;
#imax = 8; jmax = 16;
#pix = np.zeros([w*imax, w*jmax]);
#for i in range(0,imax):
#    for j in range(0,jmax):
#        pix_temp = mnist.values[ n_temp ][1:].astype('float32').reshape([28,28])/255;
#        pix[w*i:w*(i+1) , w*j:w*(j+1)] = pix_temp;
#        n_temp += 1;
#plt.imshow(pix); # Plot image
#plt.show(); # Output image

# Averages of each number
#print "mean of each number..."
#meanpix = mnist.groupby('label').mean().astype('float32')/255.0;
#n_temp = 0;
#imax = 2; jmax = 5;
#pix = np.zeros([w*imax, w*jmax]);
#for i in range(0,imax):
#    for j in range(0,jmax):
#        pix_temp = meanpix.values[ n_temp ].reshape([28,28]);
#        pix[w*i:w*(i+1) , w*j:w*(j+1)] = pix_temp;
#        n_temp += 1;
#plt.imshow(pix); # Plot image
##plt.colorbar(); # Plot color bar
#plt.show(); # Output image

# PCA & Images reconstructed from first few PCA
print "reconstructed pca images...";
figure = plt.figure();
pca_index = 1;
pca_components = [1,2,3,5,10,20,30];
for i in pca_components:
    p = decomposition.PCA(n_components=i).fit(mnist.values[:,1:]);
    transformed = p.transform(mnist.values[:,1:]);
    reconstructed = p.inverse_transform(transformed);
    
    for n in range(0,10):
        reconstructed_mean = reconstructed[ mnist.values[:,0]==n ,:].mean(axis=0);
        f = figure.add_subplot(len(pca_components), 10, pca_index)\
                  .imshow( np.reshape(reconstructed_mean, (28, 28)),\
                           interpolation='nearest', cmap=plt.cm.hot);
        for xlabel_i in f.axes.get_xticklabels():
            xlabel_i.set_visible(False);
            xlabel_i.set_fontsize(0.0);
        for xlabel_i in f.axes.get_yticklabels():
            xlabel_i.set_fontsize(0.0);
            xlabel_i.set_visible(False);
        for tick in f.axes.get_xticklines():
            tick.set_visible(False);
        for tick in f.axes.get_yticklines():
            tick.set_visible(False);
        pca_index += 1;
plt.show();







## PCA to use
#p = decomposition.PCA(n_components=50)
#p.fit(mnist.values[:,1:])
#X = p.transform(mnist.values[:,1:])
#y = mnist.values[:,0]
#Z = []
#Xtest = p.transform(test.values)
#Ztest = []
#
##
#K = [1, 3, 5];
#for k in K:
#    # we create an instance of Neighbours Classifier and fit the data.
#    clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform');
#    clf.fit(X, y);
#    Z.append(clf.predict(X));
#    Ztest.append(clf.predict(Xtest));
#print "on training..."
#print K
#print sum((Z!=y).T)/float(y.shape[0])
## Writing result on test data to file...
#Zout = pd.DataFrame(Ztest[5]);
#Zout.index += 1;
#Zout.to_csv(path_out_knn, sep=',', encoding='utf-8', index_label=["ImageId"],\
#    header=["Label"]);





# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:26:57 2020

@author: USER
"""
import os
import gc
import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display
import cv2

import pandas as pd
import numpy as np

from PIL import Image

from skimage.feature import hog
from skimage.color import rgb2grey

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from sklearn.metrics import roc_curve, auc


def create_features(img):
    # flatten three channel color image
    color_features = img.flatten()
    # convert image to greyscale
    grey_image = rgb2grey(img)
    # get HOG features from greyscale image
    hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    # combine color and hog features into a single array
    flat_features = np.hstack([color_features, hog_features])
    return flat_features

def create_feature_matrix():
    features_list = []
    directory='D:/Transformations/test'
    i=1
    for filename in os.listdir(directory):
        print(i)
        img = Image.open(os.path.join(directory, filename)) 
        img=np.array(img)
        
        # get features for image
        image_features = create_features(img)
        features_list.append(image_features)
        i+=1
    # convert list of arrays into a matrix
    gc.collect()
    feature_matrix = np.array(features_list)
    return feature_matrix

# run create_feature_matrix on our dataframe of images
feature_matrix = create_feature_matrix()
print(feature_matrix.shape)
scaler=StandardScaler()#instantiate
scaler.fit(feature_matrix) # compute the mean and standard which will be used in the next command
X_scaled=scaler.transform(feature_matrix)
pca=PCA(n_components=6) 
#pca=PCA()
pca.fit(X_scaled) 
X_pca=pca.transform(X_scaled) 
pca_variance = pca.explained_variance_


plt.figure(figsize=(8, 6))
plt.bar(range(6), pca_variance, alpha=0.5, align='center', label='individual variance')
plt.legend()
plt.ylabel('Variance ratio')
plt.xlabel('Principal components')
plt.show()
pca=PCA(n_components=3) 
#pca=PCA() z
pca.fit(X_scaled) 
X_pca=pca.transform(X_scaled) 
pca_variance = pca.explained_variance_
print("shape of X_pca", X_pca.shape)

ex_variance=np.var(X_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio )

'''plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,5], c=('red', 'blue', 'black', 'green', 'yellow', 'cyan'))
plt.show()'''

plt.matshow(pca.components_[:,:7],cmap='viridis')
plt.yticks([0,1,2],['1st Comp','2nd Comp','3rd Comp'],fontsize=10)
plt.colorbar()
#plt.xticks(range(),cancer.feature_names,rotation=65,ha='left')
plt.tight_layout()
plt.show()
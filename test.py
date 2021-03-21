import os

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

import numpy as np

# load tf model
# model = keras.models.load_model('data/CLABSI_MdlOpt_Focal_2')

# random data
# data = tf.random.normal([16, 168, 135], 0, 1, tf.float32)

import sklearn
import shap
from sklearn.model_selection import train_test_split

# print the JS visualization code to the notebook
# shap.initjs()

# train a SVM classifier
X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
svm = sklearn.svm.SVC(kernel='rbf', probability=True)
svm.fit(X_train, Y_train)

# use Kernel SHAP to explain test set predictions
explainer = shap.KernelExplainer(svm.predict_proba, X_train, link="logit")
shap_values = explainer.shap_values(X_test, nsamples=100)

print(shap_values)

# plot the SHAP values for the Setosa output of the first instance
# shap.force_plot(explainer.expected_value[0], shap_values[0][0,:], X_test.iloc[0,:], link="logit")